# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    sft_video_llm.py \
    --dataset_name=mfarre/simplevideoshorts \
    --video_cache_dir="/optional/path/to/cache/" \
    --model_name_or_path=Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size=1 \
    --output_dir=video-llm-output \
    --bf16=True \
    --tf32=True \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=4 \
    --optim="adamw_torch_fused" \
    --logging_steps=1 \
    --log_level="debug" \
    --log_level_replica="debug" \
    --save_strategy="steps" \
    --save_steps=300 \
    --learning_rate=8e-5 \
    --max_grad_norm=0.3 \
    --warmup_ratio=0.1 \
    --lr_scheduler_type="cosine" \
    --report_to="wandb" \
    --push_to_hub=False \
    --torch_dtype=bfloat16 \
    --gradient_checkpointing=True
"""

import json
import os
import random
import zipfile
import fcntl
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
import wandb
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Global flag to track if we've printed visual tokens info
_visual_tokens_printed = False

def extract_videos_from_single_process(dataset_name: str, video_cache_dir: str) -> str:
    """Extract videos from dataset zip if not already extracted from main process."""
    videos_dir = os.path.join(video_cache_dir, "videos")
    lock_file = os.path.join(video_cache_dir, ".download_lock")
    expected_zip_path = os.path.join(video_cache_dir, "videos.zip")
    
    # Check if videos directory already exists and has content
    if os.path.exists(videos_dir) and os.listdir(videos_dir):
        print(f"Videos already extracted at: {videos_dir}")
        return videos_dir
    
    # Use file-based locking for distributed coordination
    os.makedirs(video_cache_dir, exist_ok=True)
    
    # Try to acquire lock (only one process will succeed)
    try:
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Double-check after acquiring lock
            if os.path.exists(videos_dir) and os.listdir(videos_dir):
                print(f"Videos already extracted at: {videos_dir}")
                return videos_dir
            
            # Download and extract
            print("Downloading videos.zip from HuggingFace dataset repository...")
            zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="videos.zip", 
                repo_type="dataset",
                local_dir=video_cache_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"Extracting videos from {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(video_cache_dir)
            
            os.remove(zip_path)
            print(f"Videos extracted successfully to: {videos_dir}")
            
            # Count and print extracted videos
            train_dir = os.path.join(videos_dir, "train")
            val_dir = os.path.join(videos_dir, "val")
            
            train_count = len([f for f in os.listdir(train_dir) if f.endswith('.mp4')]) if os.path.exists(train_dir) else 0
            val_count = len([f for f in os.listdir(val_dir) if f.endswith('.mp4')]) if os.path.exists(val_dir) else 0
            
            print(f"Extraction complete! Total videos - Train: {train_count}, Validation: {val_count}")
            
    except (IOError, OSError):
        # Lock acquisition failed, wait for other process to finish
        print("Another process is downloading videos, waiting...")
        while not (os.path.exists(videos_dir) and os.listdir(videos_dir) and not os.path.exists(expected_zip_path)):
            print(f"Waiting for extraction to complete... (checking videos_dir: {os.path.exists(videos_dir)}, has_content: {os.path.exists(videos_dir) and os.listdir(videos_dir) if os.path.exists(videos_dir) else False}, zip_deleted: {not os.path.exists(expected_zip_path)})")
            time.sleep(5)
        print("Videos ready!")
    
    return videos_dir


def prepare_custom_dataset(example: dict[str, Any], use_system_message: bool, video_cache_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Prepare custom dataset example for training (specifically for findingdory dataset)."""
    video_path = example["video"]
    qa_pairs = json.loads(example["qa"])

    # Convert relative path to absolute path using the cache directory
    # video_path is something like "videos/train/ep_id_1157.mp4"
    absolute_video_path = os.path.join(video_cache_dir, video_path)
    
    # Verify the video file exists
    if not os.path.exists(absolute_video_path):
        raise FileNotFoundError(f"Video file not found: {absolute_video_path}")

    if use_system_message:
        system_message = "You are an expert and intelligent question answering agent. You will be shown a video that was collected by a robot yesterday while navigating around a house and picking and placing objects. Each frame in the video has a unique frame index in the top left corner of the video along with the time of day information. Your job is to help the robot complete a task today by looking at the video and finding the frame indices that the robot should move to. Note: The robot uses a magic grasp action to pick up an object, where a gripper goes close to the object and the object gets magically picked up. When deciding which frame indices to choose, make sure you choose the frame indices that are closest to the object/place."
    else:
        system_message = ""

    qa_pair = qa_pairs[0]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": absolute_video_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": qa_pair["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": qa_pair["answer"]}]},
    ]

    return {"messages": messages}


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    global _visual_tokens_printed
    
    texts = []
    video_inputs = []
    input_texts = []

    for i, example in enumerate(examples):
        video_path = next(
            content["video"]
            for message in example["messages"]
            for content in message["content"]
            if content.get("type") == "video"
        )

        texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
        input_texts.append("<|im_start|>".join(texts[-1].split("<|im_start|>")[:-1]))
        video_input = process_vision_info(example["messages"])[1][0]
        video_inputs.append(video_input)

    inputs = processor(text=texts, videos=video_inputs, return_tensors="pt", padding=True)
    inputs_only_tokens = processor(text=input_texts, videos=video_inputs, return_tensors="pt", padding=True)

    # Handle visual tokens based on processor type
    visual_tokens = (
        [151652, 151653, 151656]
        if isinstance(processor, Qwen2VLProcessor)
        else [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token),
            processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        ]
    )

    # Count visual tokens in labels before masking
    if not _visual_tokens_printed:
        visual_token_count = sum((inputs["input_ids"] == visual_token_id).sum().item() for visual_token_id in visual_tokens)
        print(f"Number of visual token elements in data tensor: {visual_token_count}")
        _visual_tokens_printed = True

    labels = inputs["input_ids"].clone()
    
    # Mask out padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Create a mask for input tokens using batched operations
    input_mask = inputs_only_tokens["input_ids"] != processor.tokenizer.pad_token_id
    input_lengths = input_mask.sum(dim=1)
    batch_size, seq_length = labels.shape
    mask_indices = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).to(labels.device)
    labels = torch.where(mask_indices < input_lengths.unsqueeze(1), -100, labels)
    # Mask out visual tokens
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels

    return inputs

def preprocess_logits_for_metrics(logits, labels):
    """
    Store only the argmax of the logits to save memory.
    """
    if type(logits) == tuple:
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def get_compute_metrics(processor):
    def extract_assistant_response(text):
        """Extract only the assistant's response from the full model output."""
        if "assistant\n" in text:
            return text.split("assistant\n", 1)[1].strip()
        return text  # Return original text if pattern not found

    def compute_metrics(pred):
        """
        Compute metrics for the model.
        """
        predictions = torch.tensor(pred.predictions)
        labels = torch.tensor(pred.label_ids)
        correct, total = 0, 0
        for i in range(len(predictions)):
            label_i, pred_i = labels[i], predictions[i]
            pred_i = pred_i[label_i != -100]
            label_i = label_i[label_i != -100]
        
            # Convert to text
            pred_text = processor.tokenizer.decode(pred_i, skip_special_tokens=True).strip()
            label_text = processor.tokenizer.decode(label_i, skip_special_tokens=True).strip()

            pred_text = extract_assistant_response(pred_text)
            label_text = extract_assistant_response(label_text)

            if pred_text == label_text:
                correct += 1
            total += 1

        print(pred_text.encode("utf-8"))
        print(label_text.encode("utf-8"))
        return {"accuracy": correct / total}
    return compute_metrics


@dataclass
class CustomScriptArguments(ScriptArguments):
    r"""
    Arguments for the script.

    Args:
        video_cache_dir (`str`, *optional*, defaults to `"/tmp/videos/"`):
            Video cache directory.
    """

    video_cache_dir: str = field(default="/tmp/videos/", metadata={"help": "Video cache directory."})
    train_samples: int = field(default=-1, metadata={"help": "Maximum number of samples to use for training."})
    eval_samples: int = field(default=-1, metadata={"help": "Maximum number of samples to use for evaluation."})
    use_system_message: bool = field(default=False, metadata={"help": "Whether to use a system message."})

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train")
    eval_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="validation")

    # Setup model
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(),
        # quantization_config=get_quantization_config(model_args),
        attn_implementation=model_args.attn_implementation,
        use_cache=False,
    )

    model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Configure model modules for gradients
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    
    # Prepare train dataset
    total_examples = len(dataset)
    if script_args.train_samples != -1:
        # Calculate step size to spread samples across dataset
        step = max(1, total_examples // script_args.train_samples)
        # Generate indices spread throughout the datase t
        indices = list(range(0, total_examples, step))[:script_args.train_samples]
        dataset = dataset.select(indices)
        print(f"Using {len(dataset)} samples spread throughout the dataset (step size: {step})")
    else:
        print(f"Using all {len(dataset)} samples for training.")

    # Filter eval dataset to specified eval_samples
    total_examples = len(eval_dataset)
    if script_args.eval_samples != -1:
        # Calculate step size to spread samples across dataset
        step = max(1, total_examples // script_args.eval_samples)
        # Generate indices spread throughout the dataset
        indices = list(range(0, total_examples, step))[:script_args.eval_samples]
        eval_dataset = eval_dataset.select(indices)
        print(f"Using {len(eval_dataset)} samples spread throughout the dataset (step size: {step})")
    else:
        print(f"Using all {len(eval_dataset)} samples for evaluation.")

    assert "findingdory" in script_args.dataset_name, "Dataset name must be findingdory"
    
    print(f"Loading dataset name: {script_args.dataset_name}")
    print(f"Using system message: {script_args.use_system_message}")
    
    # Extract videos from huggingface dataset
    extract_videos_from_single_process(script_args.dataset_name, script_args.video_cache_dir)
    prepared_train_dataset = [prepare_custom_dataset(example, use_system_message=script_args.use_system_message, video_cache_dir=script_args.video_cache_dir) for example in dataset]
    prepared_eval_dataset = [prepare_custom_dataset(example, use_system_message=script_args.use_system_message, video_cache_dir=script_args.video_cache_dir) for example in eval_dataset]
    
    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="findingdory_sft_video_llm")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_train_dataset,
        eval_dataset=prepared_eval_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        processing_class=processor.tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=get_compute_metrics(processor)
    )

    # Train model
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    
    # Save final model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
