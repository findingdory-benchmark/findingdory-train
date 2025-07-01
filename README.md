# FindingDory VLM Training

Minimal repository for fine-tuning and evaluating Vision-Language Models (VLMs) on FindingDory dataset.

## Setup

After cloning the repository, simply run the setup script:

```bash
./setup.sh
```

This will handle all the necessary dependencies and environment configuration.

## Usage

### Fine-tuning VLM

To fine-tune a Vision-Language Model on FindingDory data, use the provided SLURM script:

```bash
sbatch slurm_files/launch_findingdory_sft.sh
```

> **Note**: This script uses a subsampled version of the FindingDory dataset where all original videos are evenly downsampled to 96 frames. Dataset available at: [yali30/findingdory-subsampled-96](https://huggingface.co/datasets/yali30/findingdory-subsampled-96)
> 
> The original dataset with full-length videos can be found at: [yali30/findingdory](https://huggingface.co/datasets/yali30/findingdory)

### Evaluation

#### Parallel Checkpoint Evaluation

Run parallel evaluation jobs over multiple checkpoints from various experiment runs:

```bash
sbatch slurm_files/launch_findingdory_ckpt_evals.sh
```

> **Best Model**: We provide our best trained model checkpoint here: [yali30/findingdory-qwen2.5-VL-3B-finetuned](https://huggingface.co/yali30/findingdory-qwen2.5-VL-3B-finetuned)

#### Metrics Aggregation

Aggregate exact/relaxed accuracy metrics across various checkpoints:

```bash
python findingdory/evaluate_llm_outputs.py
```

**Evaluation Metrics:**
- **Exact Accuracy**: Performs exact string matching between the predicted and ground-truth frame lists. The VLM is fine-tuned to predict the list of all frame indices that will exactly solve the task in consideration. [[Implementation]](https://github.com/findingdory-benchmark/findingdory-train/blob/main/findingdory/evaluate_llm_outputs.py#L98-L100)
- **Relaxed Accuracy**: Considers a predicted sublist as correct if any frame index in the predicted sublist belongs to the ground-truth frame lists. For multi-goal tasks, we compute the relaxed accuracy over each sublist in the predicted list. [[Implementation]](https://github.com/findingdory-benchmark/findingdory-train/blob/main/findingdory/evaluate_llm_outputs.py#L111-L112)