# MiniKV


<p align="center">
<a href="https://2025.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2025-FF6600.svg"></a>
<a href="https://arxiv.org/pdf/2411.18077"><img src="https://img.shields.io/badge/Arxiv-2411.18077-B31B1B.svg"></a>
<a href="https://supercomputing-system-ai-lab.github.io/projects/minikv/"><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
</p>
<img width="700" height="401" alt="image" src="https://github.com/user-attachments/assets/fbc2dc1b-ca74-42e6-a0bd-98078b40ff81" />

## Overview
**MiniKV** is a lightweight, training‑free 2‑bit KV cache compression pipeline for LLM inference:
- Achieves >80% compression of the KV cache while retaining accuracy on long‑context tasks
- Hardware‑Accelerated Triton Kernel calculates signals for downstream KV cache eviction

### Key features
-   **Adaptive Quantization:** 2‑bit KV cache quantization with adaptive selection policies to maintain accuracy under high compression ratios.
-   **Plug‑and‑Play Integration:** Works seamlessly with existing LLM inference stacks—no retraining or fine‑tuning required.
-   **Hardware‑Accelerated Kernel:** Memory‑efficient kernels (FlashAttention‑compatible) in Triton for long-context inference.
  
## Requirements
Currently tested on an Nvidia GH200 with gcc/g++ 12.3.0, CUDA 12.4, triton 3.3.0, `transformers==4.47.1`

## Installation
0. Install uv for package management. You can use pip/conda as well, but uv is much faster.
```
pip install uv
```

1. Install MiniKV
```
git clone https://github.com/Supercomputing-System-AI-Lab/MiniKV.git
cd MiniKV
# create a uv virtual env at your desired location
uv venv .venv --python 3.9

source .venv/bin/activate
uv pip install setuptools IPython wheel ninja numpy
uv pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

2. Install quant package from the [KIVI repo](https://github.com/jy-yuan/KIVI/tree/main/quant)
```
cd quant
TORCH_CUDA_ARCH_LIST="9.0" uv pip install -e . --no-build-isolation # based on your GPU
```

3. Install Triton from source (we do this for a stable install on GH200, if working on a platform with pre-built wheels you should do a uv pip install here)
```
# follow triton installation instructions, usually under `install from source` in readme.md
git clone https://github.com/triton-lang/triton.git triton_install
cd triton_install
git checkout v3.3.0
uv pip install -r python/requirements.txt --no-build-isolation
uv pip install -e python --no-build-isolation
```

4. Install flash attention from souce
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
uv pip install packaging wheel pip --no-build-isolation
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="9.0" uv pip install -v -e . --no-build-isolation # increasing max_jobs speeds up install but increases memory usage
```

5. Install the selection kernel for MiniKV
```
git clone https://github.com/jpli02/selection_kernel.git selection_kernel_code
cd selection_kernel_code
TORCH_CUDA_ARCH_LIST="9.0" uv pip install -v -e . --no-build-isolation
```

## Quick Start
### Setup env
1. `cd experiments/LongBench/`
2. Include minikv source files in the PYTHONPATH.
```bash
export PYTHONPATH=$(pwd)/../../:$PYTHONPATH
```

### Running pred_minikv.py
1. To run **MiniKV**: H2O + quantization
   1. set `--use_snap False` to enable the H2O selection mechanism during pre-filling
   2. set `--heavy_ratio, --recent_ratio, --eviction_strategy` to control the eviction strategy
   3. set `--use_eviction_flash` to either enable the selective flash-attention kernel (True) or use the quadratic attention map to get the cumulative attention score (False)
   4. set `--quant_bits, group_size, residual_length` to control the quantization parameters. We use (quant_bits, group_size, residual_length) = (2,16,128) in the paper.

   An example
    ```bash
    python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy uniform/pyramid --use_eviction_flash False/True --quant_bits 2 --group_size 16 --residual_length 128
    ```
    
    Example usage for **Llama3.1-8b-instruct**
    ```bash
    python pred_minikv.py --model llama3-8b-instruct --e --full_model False --use_snap False --heavy_ratio 0.2655 --recent_ratio 0.2655 --eviction_strategy uniform --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128
    ```

2. To run snapKV
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --quant_bits 16
```

Example usage for **Llama3.1-8b-instruct**
```bash
python pred_minikv.py --model llama3-8b-instruct --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --quant_bits 16
```

1. Uncompressed model
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model True
```

Example usage for **Llama3.1-8b-instruct**
```bash
python pred_minikv.py --model llama3-8b-instruct --e --full_model True
```


1. To run snapKV + quantization (results not reported in the paper)
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --eviction_strategy uniform/pyramid --quant_bits 2 --group_size 16 --residual_length 128
```

Example usage for **Llama3.1-8b-instruct**
```bash
python pred_minikv.py --model llama3-8b-instruct --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --eviction_strategy uniform --quant_bits 2 --group_size 16 --residual_length 128
```

### Create sbatch jobs
1. `job_helper.py` creates sbatch files for running multiple experiments.
2. Jobs are saved in `slurm_jobs/` directory.
3. To run eval, ```bash launch_jobs.sh```

### Running InfiniteBench

1. `cd experiments/infinite_bench/`
2. follow the description in `experiments/infinite_bench/README.md`.

## BibTeX
```
@article{sharma2024minikv,
  title={Minikv: Pushing the limits of llm inference via 2-bit layer-discriminative kv cache},
  author={Sharma, Akshat and Ding, Hangliang and Li, Jianping and Dani, Neel and Zhang, Minjia},
  journal={arXiv preprint arXiv:2411.18077},
  year={2024}
}
```

## Acknowledgement

-   We gratefully acknowledge the developers of [SnapKV](https://github.com/FasterDecoding/SnapKV) and [KiVi](https://github.com/jy-yuan/KIVI/tree/main)
-   We are also inspired by the [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [Triton FlashAttention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
