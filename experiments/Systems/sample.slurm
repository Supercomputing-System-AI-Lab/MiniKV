#!/bin/bash
#SBATCH --job-name=MiniKV-system
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcjw-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --mem=180G # running into OOM error
#SBATCH --constraint=perf&nvperf

# module load python/3.11.6
source ~/.bashrc
module load cuda/12.4.0

# the line below moves model ckpts to the /scratch/ partition, with lot of space but large loading time
# change this to some ~/ location of the model you are actively working with
export HF_DATASETS_CACHE="/u/ndani/.cache"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/scratch/bcjw/ndani/minikv/lib:$LD_LIBRARY_PATH   # wherever the right libstdc++.so.6 is

export PYTHONPATH=/u/ndani/MiniKV-Dev:$PYTHONPATH

conda deactivate
conda activate minikv
