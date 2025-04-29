#!/bin/bash
#SBATCH --job-name=snap
#SBATCH --partition=ghx4
#SBATCH --account=bcjw-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --mem=128G # running into OOM error
#SBATCH --cpus-per-task=16

# module load python/3.11.6
source ~/.bashrc
module load cuda/12.4.0
export CXX=/usr/bin/g++-12
export CC=/usr/bin/gcc-12

# the line below moves model ckpts to the /scratch/ partition, with lot of space but large loading time
# change this to some ~/ location of the model you are actively working with
export HF_DATASETS_CACHE="/work/nvme/bcjw/asharma13/hf_cache"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH

export PYTHONPATH=/u/asharma13/ai_efficiency/MiniKV:$PYTHONPATH

source /u/asharma13/ai_efficiency/MiniKV/.venv/bin/activate
