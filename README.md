# MiniKV


<p align="center">
<a href="https://2025.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2025-FF6600.svg"></a>
<a href="https://arxiv.org/pdf/2411.18077"><img src="https://img.shields.io/badge/Arxiv-2411.18077-B31B1B.svg"></a>
<a href="https://supercomputing-system-ai-lab.github.io/projects/minikv/"><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
</p>

## Requirements
Currently tested with `transformers==4.37.0` and `cuda 12.4.0`

## Installation
We tested on Nvidia A100 using Ubuntu 20.04, CUDA 12.4, PyTorch 2.5, and Python 3.10+.


```
git clone <>
cd MiniKV
conda create -n minikv python=3.9
conda activate minikv
pip install -r requirements.txt -Uv
```
2. Install quant package from the KIVI repo
```
cd quant
pip install -e .
```
3. Install our selective flash-attention kernel implementation from <pending> 
```
cd flash-atten
python setup.py install
```

## Quick Start
### Setup env
1. `cd experiments/LongBench/`
2. Include minikv source files in the PYTHONPATH.
```bash
export PYTHONPATH=../../../MiniKV/:$PYTHONPATH
```

### Running pred_minikv.py
1. To run **MiniKV**: H2O + quantization
   1. set `--use_snap False` to enable the H2O selection mechanism during pre-filling
   2. set `--heavy_ratio, --recent_ratio, --eviction_strategy` to control the eviction strategy
   3. set `--use_eviction_flash` to either enable the selective flash-attention kernel or use the quadratic attention map to get the cumulative attention score
   4. set `--quant_bits, group_size, residual_length` to control the quantization parameters
   5. An example
    ```bash
    python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy uniform/pyramid --use_eviction_flash False/True --quant_bits 2 --group_size 16 --residual_length 128
    ```

2. To run snapKV
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap True --prompt_sparsity_ratio 0.4 --quant_bits 16
```

3. Uncompressed model
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model True
```

4. To run snapKV + quantization (results not reported in the paper)
```bash
python pred_minikv.py --model <model_name_or_path> --e --full_model False --use_snap False --heavy_ratio 0.2 --recent_ratio 0.2 --eviction_strategy uniform/pyramid --use_eviction_flash False/True --quant_bits 16
```

### Create sbatch jobs
1. `job_helper.py` creates sbatch files for running multiple experiments.
2. Jobs are saved in `slurm_jobs/` directory.
3. To run eval, ```bash launch_jobs.sh```


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
