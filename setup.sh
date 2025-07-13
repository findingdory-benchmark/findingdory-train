# Run from root directory of findingdory-train project folder
conda_env_name=findingdory

# Create conda env.
conda create -n $conda_env_name python=3.9 cmake=3.27.4 -y
conda activate $conda_env_name

# make sure conda_env_name is activated, if not end the script
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$conda_env_name" ]; then
    echo "Error: Conda environment $conda_env_name is not activated."
    exit 1
fi

# Install torch and other dependencies
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install pyproject.toml dependencies
pip install -e .
