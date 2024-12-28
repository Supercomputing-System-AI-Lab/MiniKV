from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
import argparse
import random
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)

from minikv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral

from args import parse_args

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    max_gen: int, 
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tok, max_input_length)
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
        
    inputs = torch.tensor([input_tokens]).to(model.device)
    context_length = inputs.shape[-1]
    outputs = model.generate(
        inputs,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
    )[0]
    
    pred = tok.decode(outputs[context_length:], skip_special_tokens=True)

    print("Chunked generation:", pred)
    return pred

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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress=False):
    print("device = ", device)
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name or "llama3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype = torch.float16, _attn_implementation = 'flash_attention_2').to(device)   # cant use torch_dtype=torch.bfloat16 as kivi's quantization kernels dont support it
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

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    print(f"\033[92m{args}\033[0m")
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    check_benchmark_availability(args.data_dir)
    model_name = args.model
    max_length = model2maxlen[model_name]
    data_name = args.task

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

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
    
    results = {}

    model, tok = load_model_and_tokenizer(
        model2path[model_name], 
        model_name, 
        device = "cuda", 
        compress=compress
    )
    
    ############################################################################################################
    # load compress args
    if compress:
        layers = len(model.model.layers)
        # check if window_sizes is a list
        if not isinstance(args.window_sizes, list):
            window_sizes = [args.window_sizes] * layers
        if not isinstance(args.prompt_sparsity_ratios, list):
            prompt_sparsity_ratios = [args.prompt_sparsity_ratios] * layers
        if not isinstance(args.kernel_sizes, list):
            kernel_sizes = [args.kernel_sizes] * layers
        for i in range(layers):
            model.model.layers[i].self_attn.config.use_snap = args.use_snap
            model.model.layers[i].self_attn.config.prompt_sparsity_ratio = prompt_sparsity_ratios[i]
            model.model.layers[i].self_attn.config.window_size = window_sizes[i]
            model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
            model.model.layers[i].self_attn.config.pooling = args.pooling
            model.model.layers[i].self_attn.config.heavy_ratio = args.heavy_ratio
            model.model.layers[i].self_attn.config.recent_ratio = args.recent_ratio
            model.model.layers[i].self_attn.config.eviction_strategy = args.eviction_strategy
            model.model.layers[i].self_attn.config.use_eviction_flash = args.use_eviction_flash
            model.model.layers[i].self_attn.config.quant_bits = args.quant_bits
            model.model.layers[i].self_attn.config.group_size = args.group_size
            model.model.layers[i].self_attn.config.residual_length = args.residual_length
    ############################################################################################################
        

    for data_name in data_names:
        
        # Model
        max_gen = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_gen >= max_length:
            max_gen = 500

        # Data
        result_dir = Path(args.output_dir, f"{write_model_name}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max gen: {max_gen}")

        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, write_model_name, max_seq_length)
            with open(output_path) as f:
                preds = [json.loads(ii) for ii in f.readlines()]

        for i, eg in tqdm(enumerate(examples)):
            #if i < args.start_example_id or i < len(preds):
            #    continue
            input_text = create_prompt(eg, data_name, write_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            # print(input_text.index(ground_truth), len(input_text), input_text.index(ground_truth) / len(input_text))
            # print(f"====== Example {i} ======")
            pred = get_pred(
                model,
                tok,
                input_text,
                max_input_length=max_length - max_gen,
                max_gen=max_gen,
                verbose=args.verbose,
            )
            print("Ground Truth", get_answer(eg, data_name))
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{write_model_name}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
