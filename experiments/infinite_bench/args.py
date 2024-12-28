import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, help="Task name")
    parser.add_argument('--data_dir', type=str, default="./data", help="Data directory")
    parser.add_argument('--output_dir', type=str, default="./results", help="Output directory") 
    parser.add_argument('--rewrite', action='store_true', help="Rewrite existing results")
    parser.add_argument('--num_eval_examples', type=int, default=-1, help="Number of examples to evaluate")
    parser.add_argument('--verbose', action='store_true', help="Verbose mode")
    
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "llama2-13b-chat-4k", "llama3-8b-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", 
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m",
        "Yarn-llama-2-7b-128k"])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--full_model', type=lambda x: x.lower() == 'true', help="Use uncompressed model", default=False)
    parser.add_argument('--use_snap', type=lambda x: x.lower() == 'true', help="Use snapKV for eviction", default=False)
    
    # snapKV args
    parser.add_argument('--prompt_sparsity_ratios', type=float, help="The sparsity ratio of the prompt", default=0.5)
    parser.add_argument('--window_sizes', type=int, help="The window size of the prompt", default=32)
    parser.add_argument('--kernel_sizes', type=int, help="The kernel size of the prompt", default=7)
    parser.add_argument('--pooling', type=str, help="The pooling method of the prompt", default="maxpool")
    
    # h2o args
    parser.add_argument('--heavy_ratio', type=float, help="The ratio of heavy hitter", default=0.25)
    parser.add_argument('--recent_ratio', type=float, help="The ratio of recent window", default=0.25)
    parser.add_argument('--eviction_strategy', type=str, help="The eviction strategy", default="uniform", choices=["uniform", "pyramid"])
    parser.add_argument('--use_eviction_flash', type=lambda x: x.lower() == 'true', help="Use custom flash_attn kernel which returns the cumulative attention map", default=False)
    
    # quantization args
    parser.add_argument('--quant_bits', type=int, help="The number of bits for key/value", default=2)
    parser.add_argument('--group_size', type=int, help="The group size", default=16)
    parser.add_argument('--residual_length', type=int, help="The residual length", default=128)
    
    args = parser.parse_args(args)
    if args.pooling == 'None':
        args.pooling = None
    return args