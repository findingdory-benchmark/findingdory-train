#!/bin/bash
#SBATCH --job-name=mmbench_sft
#SBATCH --output=/srv/flash1/yali30/code/findingdory-release/findingdory-train/slurm_logs/sft_qwen_train_jun_30/full-ft-96-frames-lr5e6-epoch5-3B-%j.out
#SBATCH --error=/srv/flash1/yali30/code/findingdory-release/findingdory-train/slurm_logs/sft_qwen_train_jun_30/full-ft-96-frames-lr5e6-epoch5-3B-%j.err
#SBATCH --gpus=a40:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --exclude=chomps,ephemeral-3,walle,friday,cyborg,starrysky,hk47,jill,xaea-12,johnny5,calculon,puma
#SBATCH --qos="short"
#SBATCH --partition=kira-lab
#SBATCH --requeue
#SBATCH --signal=USR1@100

MAIN_ADDR=$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

REPO_ROOT_DIR=/srv/flash1/yali30/code/findingdory-release/findingdory-train

export TRANSFORMERS_CACHE=$REPO_ROOT_DIR/models
export HF_DATASETS_CACHE=$REPO_ROOT_DIR/hf_datasets
export WANDB_API_KEY=
export TRITON_CACHE_DIR=$REPO_ROOT_DIR/triton_cache

source /coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate findingdory

cd $REPO_ROOT_DIR/trl

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    $REPO_ROOT_DIR/findingdory/sft_video_llm.py \
    --dataset_name yali30/findingdory-subsampled-96 \
    --video_cache_dir=hf_cache_dir/findingdory-subsampled-96 \
    --dataset_train_split train \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --logging_steps 1 \
    --log_level debug \
    --log_level_replica debug \
    --save_strategy steps \
    --save_steps 200 \
    --report_to wandb \
    --push_to_hub False \
    --output_dir runs/jun_30/full-ft-96-frames-lr5e6-epoch5-3B \
    --optim adamw_torch_fused \
    --learning_rate 5e-6 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 1 \
    --do_eval True \
    --eval_strategy steps \
    --eval_steps 1000 \
    --eval_on_start True \
    --per_device_eval_batch_size 1 \
    --bf16_full_eval True \
    --eval_samples 64