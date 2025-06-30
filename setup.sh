# Run from root directory of findingdory-train project folder
conda_env_name=findingdory

# Create conda env and install habitat-sim
mamba create -n $conda_env_name python=3.9 cmake=3.27.4 -y
mamba activate $conda_env_name

# Install torch and other dependencies
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install importlib-metadata
pip install deepspeed==0.16.5
pip install wandb
pip install peft
pip install qwen-vl-utils
pip install flash_attn==2.7.4.post1
pip install transformers==4.51.0
pip install matplotlib

# Clone and install the huggingface/trl package
git clone --branch v0.16.1 --depth 1 https://github.com/huggingface/trl.git
cd trl
pip install -e .
cd ..