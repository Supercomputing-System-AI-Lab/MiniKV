import torch
import os, sys, json
import time
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from minikv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--prompt-length', type=int, required=True, help='Prompt length')
    parser.add_argument('--output-length', type=int, required=True, help='Output length')
    parser.add_argument('--ratio', type=float, required=True, help='Heavy hitter and recent ratio')
    parser.add_argument('--bits', type=int, required=True, help='Quantization')
    parser.add_argument('--prefix', type=str, required=True, help='Metric file prefix')
    return parser.parse_args(args)

def get_quantiles(data):
    """Compute quantiles for a list of numbers."""
    if not data:
        return {}
    sorted_data = sorted(data)
    n = len(sorted_data)
    return {
        "mean": sum(data) / n,
        "std": (sum((x - sum(data) / n) ** 2 for x in data) / n) ** 0.5,
        "min": sorted_data[0],
        "Q1": sorted_data[n // 4],
        "median": sorted_data[n // 2],
        "Q3": sorted_data[(3 * n) // 4],
        "max": sorted_data[-1],
    }

# ---- Main ---- #
args = parse_args()

# Configuration
K_BITS = args.bits
V_BITS = args.bits
RATIO = args.ratio
GROUP_SIZE = 16
RESIDUAL_LENGTH = 128
BATCH_SIZE = args.batch_size
PROMPT_LENGTH = args.prompt_length
OUTPUT_LENGTH = args.output_length

num_repeats = 5
num_warmups = 10
PATH_TO_YOUR_SAVE_DIR = os.getenv('HF_HUB_CACHE', './')
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR
latency, used_mem, reserved_mem = [], [], []

model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'

device = 'cuda'
if K_BITS < 16 and V_BITS < 16:
    print(f"[INFO] Loading MiniKV model.")
    if 'llama' in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            _attn_implementation='flash_attention_2'
        ).to(device)
    elif 'mistral' in model_name_or_path.lower():
        raise NotImplementedError("Mistral model not supported yet.")
    
    args.use_snap = False
    args.quant_bits = K_BITS
    args.group_size = GROUP_SIZE
    args.residual_length = RESIDUAL_LENGTH
    args.recent_ratio = RATIO
    args.heavy_ratio = RATIO
    args.use_eviction_flash = True 
    
    replace_llama(args)

    layers = len(model.model.layers)
    for i in range(layers):
        model.model.layers[i].self_attn.config.use_snap = args.use_snap
        model.model.layers[i].self_attn.config.heavy_ratio = RATIO
        model.model.layers[i].self_attn.config.recent_ratio = RATIO
        model.model.layers[i].self_attn.config.use_eviction_flash = args.use_eviction_flash
        model.model.layers[i].self_attn.config.quant_bits = V_BITS
        model.model.layers[i].self_attn.config.group_size = GROUP_SIZE
        model.model.layers[i].self_attn.config.residual_length = RESIDUAL_LENGTH
        model.model.layers[i].self_attn.config.eviction_strategy = 'uniform'

else:
    print(f"[INFO] Loading full model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        _attn_implementation='flash_attention_2',
    ).to(device)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

tokenizer.pad_token = tokenizer.eos_token

# Prepare input data
context = ['t,' * (PROMPT_LENGTH // 2) for _ in range(BATCH_SIZE)]
inputs = tokenizer(context, return_tensors="pt", padding=True).to(device)
input_ids = inputs['input_ids']

print(f"bs: {BATCH_SIZE}, seqlen: {input_ids.shape[1]}+{OUTPUT_LENGTH}\nmodel:{model_name_or_path}")

# Warmup
with torch.no_grad():
    print(f"Warming up {num_warmups} times")
    for _ in range(num_warmups):
        model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            temperature=1.0
        )

torch.cuda.reset_peak_memory_stats()

# Main testing
oom_error = False
try:
    with torch.no_grad():
        for _ in tqdm(range(num_repeats), desc="Repeats"):
            torch.cuda.synchronize()
            st = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=OUTPUT_LENGTH,
                min_new_tokens=OUTPUT_LENGTH,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=None
            )
            torch.cuda.synchronize()
            latency.append(time.perf_counter() - st)
            assert outputs.shape[1] == PROMPT_LENGTH + OUTPUT_LENGTH + 1
            used_mem.append(torch.cuda.max_memory_allocated())
            reserved_mem.append(torch.cuda.max_memory_reserved())

            # Clear cache
            for name, module in model.named_modules():
                if "llama" in model_name_or_path.lower():
                    pass  # Placeholder for module-specific cleaning
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
except torch.cuda.OutOfMemoryError:
    print(f"[ERROR] CUDA OOM at bs: {BATCH_SIZE}, seqlen: {PROMPT_LENGTH}+{OUTPUT_LENGTH}")
    oom_error = True
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Metrics
latency_stats = get_quantiles(latency) if not oom_error else {}
used_mem_stats = get_quantiles([mem / (1024 ** 3) for mem in used_mem]) if not oom_error else {}
reserved_mem_stats = sum(reserved_mem) / ((1024 ** 3) * num_repeats) if not oom_error else None

metrics = {
    "model": model_name_or_path,
    "batch_size": BATCH_SIZE,
    "prompt_length": PROMPT_LENGTH,
    "output_length": OUTPUT_LENGTH,
    "quantization": f"K={K_BITS}, V={V_BITS}",
    "ratio": RATIO,
    "OOM": "Yes" if oom_error else "No",
    "latency_mean": latency_stats.get("mean"),
    "latency_std": latency_stats.get("std"),
    "latency_min": latency_stats.get("min"),
    "latency_Q1": latency_stats.get("Q1"),
    "latency_median": latency_stats.get("median"),
    "latency_Q3": latency_stats.get("Q3"),
    "latency_max": latency_stats.get("max"),
    "peak_mem_usage_mean": used_mem_stats.get("mean"),
    "peak_mem_usage_std": used_mem_stats.get("std"),
    "peak_mem_usage_min": used_mem_stats.get("min"),
    "peak_mem_usage_Q1": used_mem_stats.get("Q1"),
    "peak_mem_usage_median": used_mem_stats.get("median"),
    "peak_mem_usage_Q3": used_mem_stats.get("Q3"),
    "peak_mem_usage_max": used_mem_stats.get("max"),
    "peak_mem_reserved_mean": reserved_mem_stats,
    "tokens/second": (PROMPT_LENGTH + OUTPUT_LENGTH) * BATCH_SIZE / latency_stats["mean"] if not oom_error else None
}

# Save results
current_dir = os.path.dirname(os.path.realpath(__file__))
output_csv = os.path.join(current_dir, f"results/{args.prefix}_{BATCH_SIZE}_{PROMPT_LENGTH}_{OUTPUT_LENGTH}_{RATIO}_{args.bits}.csv")
if not os.path.isfile(output_csv):
    pd.DataFrame([metrics]).to_csv(output_csv, header=True, index=False)
else:            
    pd.DataFrame([metrics]).to_csv(output_csv, mode='a', header=False, index=False)

if oom_error:
    sys.exit(1)
