#!/bin/bash
#SBATCH --job-name=snapkv_llama3
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --account=bcjw-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --mem=128G # running into OOM error
#SBATCH --cpus-per-task=16

# module load python/3.11.6
source ~/.bashrc
module load cuda/12.4.0

# the line below moves model ckpts to the /scratch/ partition, with lot of space but large loading time
# change this to some ~/ location of the model you are actively working with
export HF_DATASETS_CACHE="/u/asharma13/.cache"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/u/asharma13/.conda/envs/minikv/lib:$LD_LIBRARY_PATH   # wherever the right libstdc++.so.6 is

export PYTHONPATH=/u/asharma13/ai_efficiency/MiniKV:$PYTHONPATH

conda deactivate
conda activate minikv
