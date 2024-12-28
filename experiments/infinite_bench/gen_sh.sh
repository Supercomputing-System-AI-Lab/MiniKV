#!/bin/bash

# Directory to store the generated Slurm scripts
output_dir="scripts/snapKV"

# Create the directory if it doesn't exist
mkdir -p $output_dir

# Base Slurm script template
slurm_template='#!/bin/bash
#SBATCH --job-name=MiniKV
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcjw-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --mem=128G # running into OOM error
#SBATCH --cpus-per-task=16

# module load python/3.11.6
source ~/.bashrc
module load cuda/12.4.0

# the line below moves model ckpts to the /scratch/ partition, with lot of space but large loading time
# change this to some ~/ location of the model you are actively working with
export HF_DATASETS_CACHE="/scratch/bcjw/ding3/hf_home"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/u/ding3/.conda/envs/minikv/lib:$LD_LIBRARY_PATH   # wherever the right libstdc++.so.6 is

export PYTHONPATH=/u/ding3/MiniKV-Dev:$PYTHONPATH

conda deactivate
conda activate minikv_new

TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")

export TOKENIZERS_PARALLELISM=false

MODEL="llama3-8b-instruct"
NUM_EVAL=-1

for task in ${TASKS[@]}; do
echo $task
python "run_infinitebench.py" \
    --task $task \
    --model $MODEL \
    --data_dir ./data \
    --output_dir ./results \
    --rewrite \
    --verbose \
    --num_eval_examples $NUM_EVAL \
    --full_model False \
    --use_snap True \
    --prompt_sparsity_ratio heavy_ratio_placeholder \
    --quant_bits 16
done'

# Loop to generate Slurm scripts with different heavy_ratio values
for i in {1..10}; do
  # Calculate the heavy_ratio value
  heavy_ratio=$(echo "scale=1; $i/10" | bc)
  
  # Replace placeholders in the template
  slurm_script=$(echo "$slurm_template" | sed "s/layer_x/layer_$i/g" | sed "s/heavy_ratio_placeholder/$heavy_ratio/g")
  
  # Write the generated script to a file
  echo "$slurm_script" > "${output_dir}/job${i}.slurm"
done

echo "Generated Slurm scripts in $output_dir"
