from transformers import AutoConfig

for name in [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3.2-3B-Instruct"
]:
    cfg = AutoConfig.from_pretrained(name)
    h  = cfg.num_attention_heads
    h_kv = cfg.num_key_value_heads
    g = h // h_kv
    print(f"{name:<40}  H={h:2d}  H_kv={h_kv:2d}  groups={g}")
