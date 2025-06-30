#!/usr/bin/env python3
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
python evaluate_video_llm.py \
    --checkpoint_dir=video-llm-output \
    --dataset_name=yali30/findingdory-val-subsampled-48-qwen \
    --model_name_or_path=Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_eval_batch_size=1 \
    --bf16 \
    --torch_dtype=bfloat16 \
    --max_samples=10 \
    --output_file=evaluation_results.json
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
import torch
import tqdm
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

from trl import get_kbit_device_map

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def download_video(url: str, cache_dir: str) -> str:
    """Download video if not already present locally."""
    os.makedirs(cache_dir, exist_ok=True)  # Create cache dir if it doesn't exist
    filename = url.split("/")[-1]
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}") from e


def prepare_custom_dataset(example: dict[str, Any], use_system_message: bool) -> dict[str, list[dict[str, Any]]]:
    """Prepare custom dataset example for evaluation (specifically for findingdory dataset)."""
    video_path = example["video_path"]
    qa_pairs = json.loads(example["qa"])
    task_id = example["id"].split("_")[-1]

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
                {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": qa_pair["question"]},
            ],
        },
    ]

    return {
        "messages": messages,
        "ground_truth": qa_pair["answer"],
        "task_id": task_id
    }


def prepare_dataset(example: dict[str, Any], cache_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Prepare dataset example for evaluation."""
    video_url = example["video_url"]
    timecoded_cc = example["timecoded_cc"]
    qa_pairs = json.loads(example["qa"])

    system_message = "You are an expert in movie narrative analysis."
    base_prompt = f"""Analyze the video and consider the following timecoded subtitles:

{timecoded_cc}

Based on this information, please answer the following questions:"""

    selected_qa = random.sample(qa_pairs, 1)[0]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": download_video(video_url, cache_dir), "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": f"{base_prompt}\n\nQuestion: {selected_qa['question']}"},
            ],
        },
    ]

    return {
        "messages": messages,
        "ground_truth": selected_qa["answer"]
    }


@dataclass
class EvaluationArguments:
    """Arguments for the evaluation script."""
    
    checkpoint_dir: str = field(metadata={"help": "Path to the trained model checkpoint."})
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    dataset_name: str = field(metadata={"help": "The name of the dataset to use."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use."})
    video_cache_dir: str = field(default="/tmp/videos/", metadata={"help": "Video cache directory."})
    max_samples: int = field(default=-1, metadata={"help": "Maximum number of samples to use for evaluation."})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Batch size per device during evaluation."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 16-bit (mixed) precision."})
    torch_dtype: Optional[str] = field(default=None, metadata={"help": "Override the default `torch.dtype` and load the model with specified dtype."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether to trust remote code."})
    output_file: str = field(default="evaluation_results.json", metadata={"help": "Path to save evaluation results."})


# Simple exact match metric
def calculate_exact_match(pred_text, ground_truth):
    """Calculate exact match score (1.0 if texts match exactly, 0.0 otherwise)."""
    return 1.0 if pred_text.strip() == ground_truth.strip() else 0.0


def extract_assistant_response(text):
    """Extract only the assistant's response from the full model output."""
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text  # Return original text if pattern not found


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained video LLM model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model name or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")
    parser.add_argument("--video_cache_dir", type=str, default="/tmp/videos/", help="Video cache directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples for evaluation")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--bf16", action="store_true", help="Use BFloat16 precision")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--torch_dtype", type=str, default=None, help="Override torch dtype")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--apply_liger", action="store_true", help="Apply liger kernels to the model")
    parser.add_argument("--split", type=str, default="validation", help="Split to evaluate on")
    parser.add_argument("--use_system_message", type=bool, default=False, help="Use system message")
    
    args = parser.parse_args()
    
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.torch_dtype:
        torch_dtype = getattr(torch, args.torch_dtype)
    
    print(f"Using device: {device}, dtype: {torch_dtype}")
    
    # Load base model and processor
    print(f"Loading base model from {args.model_name_or_path}...")
    model_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
    
    model = AutoModelForVision2Seq.from_pretrained(args.model_name_or_path, **model_kwargs)
    
    # Load trained checkpoint
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    try:
        # Check if this is a PEFT model
        if os.path.exists(os.path.join(args.checkpoint_dir, "adapter_config.json")):
            print("Loading as PEFT model...")
            model = PeftModel.from_pretrained(model, args.checkpoint_dir)
            model = model.merge_and_unload()  # Merge adapter weights for better inference performance
        else:
            # Load as full model
            model = AutoModelForVision2Seq.from_pretrained(args.checkpoint_dir, **model_kwargs)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Falling back to base model...")
        
    # apply liger kernel patch
    if args.apply_liger:
        print("Applying liger kernels to the loaded model !")
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        from transformers.modeling_utils import PreTrainedModel
        if isinstance(model, PreTrainedModel):
            # Patch the model with liger kernels. Use the default kernel configurations.
            _apply_liger_kernel_to_instance(model=model)
        elif hasattr(model, "get_base_model") and isinstance(model.get_base_model(), PreTrainedModel):
            # Patch the base model with liger kernels where model is a PeftModel. Use the default kernel configurations.
            _apply_liger_kernel_to_instance(model=model.get_base_model())
        else:
            logger.warning(
                "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
            )
    else:
        print("Liger Kernels NOT applied to loaded model !")
        
    # Put model in evaluation mode
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name} --- split: {args.split}...")
    dataset = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split)
    
    # Calculate total examples and indices to sample
    total_examples = len(dataset)
    print(f"Found total samples: {total_examples}")
    if args.max_samples != -1:
        # Calculate step size to spread samples across dataset
        step = max(1, total_examples // args.max_samples)
        # Generate indices spread throughout the dataset
        indices = list(range(0, total_examples, step))[:args.max_samples]
        dataset = dataset.select(indices)
        print(f"Using {len(dataset)} samples spread throughout the dataset (step size: {step})")
    else:
        print(f"Using all {len(dataset)} samples for evaluation.")
    
    # Prepare dataset
    print("Preparing dataset for evaluation...")
    if "findingdory" in args.dataset_name:
        # karmesh fix to sample 26 examples only for overfit test
        # allowed_tasks = ["1"]
        # dataset = dataset.filter(lambda x: x["id"].split('_')[-1] in allowed_tasks and x["id"].split('_')[1][0] == "5" and len(x["id"].split('_')[1]) == 3)
        print(f"Loading dataset name: {args.dataset_name}")
        print(f"Using system message: {args.use_system_message}")
        prepared_examples = [prepare_custom_dataset(example, args.use_system_message) for example in dataset]
    else:
        prepared_examples = [prepare_dataset(example, args.video_cache_dir) for example in dataset]
    
    # Evaluation loop
    print("Starting evaluation...")
    results = []
    exact_match_scores = []
    
    for i, example in enumerate(tqdm.tqdm(prepared_examples)):
        try:
            # Process the input
            messages = example["messages"]
            # print("Sample: ", example)
            ground_truth = example["ground_truth"]
            task_id = example["task_id"]
            
            # Get video path from messages
            video_path = next(
                content["video"]
                for message in messages
                for content in message["content"]
                if content.get("type") == "video"
            )
            
            # Convert to model inputs
            with torch.no_grad():
                inputs = processor(
                    text=processor.apply_chat_template(messages, tokenize=False),
                    videos=process_vision_info(messages)[1][0],
                    return_tensors="pt",
                ).to(device)
                
                # Generate output
                output_obj = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    output_logits=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=True
                )
                output_ids = output_obj.sequences
                                    
                # Decode output
                output_text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                
                # Extract only the assistant's response
                output_text = extract_assistant_response(output_text)
                
                # Calculate exact match score
                exact_match = calculate_exact_match(output_text, ground_truth)
                exact_match_scores.append(exact_match)
                
                # Store results
                example_result = {
                    "example_id": i,
                    "task_id": task_id,
                    "video": os.path.basename(video_path),
                    "ground_truth": ground_truth,
                    "model_output": output_text,
                    "exact_match": exact_match
                }
                results.append(example_result)
                
                # Print results
                print(f"\nExample {i}:")
                print(f"Video: {os.path.basename(video_path)}")
                print(f"Task ID: {task_id}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Model Output: {output_text}")
                print(f"Exact Match: {exact_match:.1f}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append({
                "example_id": i,
                "error": str(e)
            })
    
    # Calculate and print average exact match score
    avg_exact_match = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
    
    print("\n" + "=" * 50)
    print(f"Overall Exact Match Score: {avg_exact_match:.4f}")
    print("=" * 50)
    
    # Save results
    final_results = {
        "individual_results": results,
        "average_exact_match": avg_exact_match
    }
    
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main() 