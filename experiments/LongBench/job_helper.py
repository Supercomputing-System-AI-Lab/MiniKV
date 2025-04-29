import os, sys, json
import itertools
import numpy as np
from tqdm import tqdm

os.system("rm slurm_jobs/*")

with open("sample.slurm", "r") as f:
    original = f.read()

total_budget = [0.04]
model = ['llama2-7b-chat-4k', 'llama2-13b-chat-4k', 'llama3-8b-instruct', 'llama3-3b-instruct', 'llama3-1b-instruct', 'mistral-7B-instruct-v0.2']
use_snaps = [True]
strategy = ['uniform']
quant_bits = [16]

for idx, (b, m, s, u, q) in enumerate(tqdm(itertools.product(total_budget, model, strategy, use_snaps, quant_bits), desc="Creating SLURM jobs")):
    if u == False and q == 16:
        continue
    
    if u == True:
        job = f"python pred_minikv.py --model {m} --e --full_model False --use_snap {u} --prompt_sparsity_ratio {b} --eviction_strategy {s} --quant_bits {q} --group_size 16 --residual_length 128"
    else:
        job = f"python pred_minikv.py --model {m} --e --full_model False --use_snap {u} --heavy_ratio {b/2} --recent_ratio {b/2} --eviction_strategy {s} --quant_bits {q} --group_size 16 --residual_length 128"
    
    job_content = original + job
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(job_content)


# also make the full model jobs
# start_idx = len(os.listdir("slurm_jobs"))  # get the current index for new jobs
# for m in tqdm(model):
#     # Create full model jobs for each model
#     job = f"python pred_minikv.py --model {m} --e --full_model True"
    
#     job_content = original + job
#     with open(f"slurm_jobs/job_{start_idx}.slurm", "w") as f:
#         f.write(job_content)
    
#     start_idx += 1  # Increment the index for the next job
