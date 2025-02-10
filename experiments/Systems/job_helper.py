import os, sys, json
import itertools
import numpy as np
from tqdm import tqdm

os.system("rm slurm_jobs/*")

with open("sample.slurm", "r") as f:
    original = f.read()

job_list = [
    #  Peak memory usage and throughput vs batch size
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 2 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 4 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 8 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 16 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 32 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 34 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 36 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 38 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 40 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    "python experiments/Systems/system_benchmark.py --batch-size 42 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix batch",
    
    # latency vs prompt length
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 1024 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 2048 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 4096 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 8192 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 16384 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 32768 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 36864 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 40960 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
    "python experiments/Systems/system_benchmark.py --batch-size 1 --prompt-length 45056 --output-length 1024 --ratio 0.25 --bits 2 --prefix input",
]

for idx, job in enumerate(tqdm(job_list, desc="Creating SLURM jobs")):
    job_content = original + job
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(job_content)