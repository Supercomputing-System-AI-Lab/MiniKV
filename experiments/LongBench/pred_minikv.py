import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch
from minikv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral
def parse_args(args=None):
    parser = argparse.ArgumentParser()
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

def process_decimal_string(str_):
    if not isinstance(str_, str):
        str_ = str(str_)
    return str_.replace(".", "d")

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        print('chatglm3')
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        print('chatglm')
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        print('longchat')
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2"  in model_name or "llama-2" in model_name or "lwm" in model_name:
        # print('llama2', model_name)
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        print('xgen')
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        print('internlm')
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "mistral" in model_name or "mixtral" in model_name:
        # print('mistral')
        # from fastchat.model import get_conversation_template
        # conv = get_conversation_template("mistral")
        # conv.append_message(conv.roles[0], prompt)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        prompt = prompt
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, 
                        prompt_format, dataset, model_name, 
                        model2path, out_path, 
                        compress=False, 
                        use_snap=False,
                        prompt_sparsity_ratios=None,
                        window_sizes=None,
                        kernel_sizes=None,
                        pooling=None,
                        heavy_ratio=None,
                        recent_ratio=None,
                        eviction_strategy=None,
                        use_eviction_flash=None,
                        quant_bits=None,
                        group_size=None,
                        residual_length=None,
                        ):
    # device = torch.device(f'cuda:{rank}')
    # device = model.device
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device = "cuda", compress=compress)
    device = model.device
    printed = False
    preds = []
    for json_obj in tqdm(data):
        ############################################################################################################
        # load compress args
        if compress:
            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(prompt_sparsity_ratios, list):
                prompt_sparsity_ratios = [prompt_sparsity_ratios] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.use_snap = use_snap
                model.model.layers[i].self_attn.config.prompt_sparsity_ratio = prompt_sparsity_ratios[i]
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                model.model.layers[i].self_attn.config.heavy_ratio = heavy_ratio
                model.model.layers[i].self_attn.config.recent_ratio = recent_ratio
                model.model.layers[i].self_attn.config.eviction_strategy = eviction_strategy
                model.model.layers[i].self_attn.config.use_eviction_flash = use_eviction_flash
                model.model.layers[i].self_attn.config.quant_bits = quant_bits
                model.model.layers[i].self_attn.config.group_size = group_size
                model.model.layers[i].self_attn.config.residual_length = residual_length
        ############################################################################################################
        
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress=False):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name or "llama3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype = torch.float16, _attn_implementation = 'flash_attention_2').to(device)   # cant use torch_dtype=torch.bfloat16 as kivi's quantization kernels dont support it
    elif "longchat" in model_name or "vicuna" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True
                )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
        )
    elif "llama-2" in model_name or "lwm" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True,
                    trust_remote_code=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_cache=True,
                    use_flash_attention_2=True,
                    trust_remote_code=True
                )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
            trust_remote_code=True
        )
    elif "mistral" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
        )
    elif "mixtral" in model_name:
        if not compress:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                use_flash_attention_2=True
            )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            # padding_side="right",
            # use_fast=False,
        )
    else:
        raise ValueError(f"Model {model_name} not supported!")
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    print(f"\033[92m{args}\033[0m")
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["2wikimqa", "qasper", "multifieldqa_en", "hotpotqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred") and not args.e:
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if not args.full_model:
        compress = True
        compress_args = {
            "use_snap": args.use_snap,
            "prompt_sparsity_ratios": args.prompt_sparsity_ratios,
            "window_sizes": args.window_sizes,
            "kernel_sizes": args.kernel_sizes,
            "pooling": args.pooling,
            "heavy_ratio": args.heavy_ratio,
            "recent_ratio": args.recent_ratio,
            "eviction_strategy": args.eviction_strategy,
            "use_eviction_flash": args.use_eviction_flash,
            "quant_bits": args.quant_bits,
            "group_size": args.group_size,
            "residual_length": args.residual_length,
        }
        
        if args.use_snap:
            if args.quant_bits == 16:
                write_model_name = model_name + f"use_snap{args.use_snap}_p{process_decimal_string(args.prompt_sparsity_ratios)}_w{args.window_sizes}_k{args.kernel_sizes}_pool{args.pooling}"
            else:
                write_model_name = model_name + f"use_snap{args.use_snap}_p{process_decimal_string(args.prompt_sparsity_ratios)}_w{args.window_sizes}_k{args.kernel_sizes}_pool{args.pooling}_bits{args.quant_bits}_g{args.group_size}_r{args.residual_length}"
        else:
            if args.quant_bits == 16:
                write_model_name = model_name + f"use_snap{args.use_snap}_h{process_decimal_string(args.heavy_ratio)}_r{process_decimal_string(args.recent_ratio)}_use_eviction_flash{args.use_eviction_flash}"
            else:
                write_model_name = model_name + f"use_snap{args.use_snap}_h{process_decimal_string(args.heavy_ratio)}_r{process_decimal_string(args.recent_ratio)}_use_eviction_flash{args.use_eviction_flash}_bits{args.quant_bits}_g{args.group_size}_r{args.residual_length}"
        
        if args.eviction_strategy == "pyramid":
            write_model_name = "pyramid/" + write_model_name
        
        if 'llama' in model_name:
            replace_llama(args)
        elif 'mistral' in model_name:
            replace_mistral(args)
        elif 'mixtral' in model_name:
            replace_mixtral()
    else:
        compress = False
        compress_args = None
        write_model_name = model_name
    
    for dataset in tqdm(datasets):
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{write_model_name}"):
                os.makedirs(f"pred_e/{write_model_name}")
            out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{write_model_name}"):
                os.makedirs(f"pred/{write_model_name}")
            out_path = f"pred/{write_model_name}/{dataset}.jsonl"
        print(f"[INFO] Writing to {out_path = }")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if compress_args is not None:
            get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, compress, **compress_args)
        else:
            get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, compress)
