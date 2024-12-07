
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from typing import List, Optional, Tuple, Union

import math
from quant.new_pack import triton_quantize_and_pack_along_last_dim, quant_and_pack_vcache, unpack_and_dequant_vcache
from quant.matmul import cuda_bmm_fA_qB_outer

def get_attn_weights(group_size, query_states, key_code, key_scale, key_mn, key_unq, bits, head_dim):
    '''
    Get unnormalized attention weights using either the triton kernel or simulated unpacking of the key tensor
    '''
    if bits == 2:   # use kivi's kernel for INT2 numbers
        if key_code is not None:
            att_qkquant = cuda_bmm_fA_qB_outer(group_size, query_states, key_code, key_scale, key_mn, bits)
        else:
            att_qkquant = None

        if key_unq is not None:
            att_qkfull = torch.matmul(query_states, key_unq.transpose(2,3))
        else :
            att_qkfull = None

        attn_list = []
        if att_qkquant is not None:
            attn_list.append(att_qkquant)
        if att_qkfull is not None:
            attn_list.append(att_qkfull)
        attn_weights = torch.cat(attn_list, dim=-1) / math.sqrt(head_dim)
        
    elif bits == 4 or bits == 8:   # simulate unpacking and do multiplication
        if key_code is not None:
            keys_ = unpack_and_dequant_vcache(key_code, key_scale, key_mn, group_size, bits)
        else :
            keys_ = None
        
        if key_unq is not None:
            if keys_ is not None:
                keys_ = torch.cat([keys_, key_unq.transpose(2,3)], dim=-1)
            else:
                keys_ = key_unq.transpose(2,3)
        attn_weights = torch.matmul(query_states, keys_) / math.sqrt(head_dim)
        
    else :
        raise NotImplementedError(f"Quantization bits {bits} not supported")
    
    return attn_weights

def get_attn_output(group_size, attn_weights, value_code, value_scale, value_mn, value_unq, bits):
    '''
    Get attention output using either the triton kernel or simulated unpacking of the value tensor
    '''
    if bits == 2:
        value_full_length = value_unq.shape[-2] if value_unq is not None else 0
        if value_code is None:
            attn_output = torch.matmul(attn_weights, value_unq)
        else:
            if value_full_length != 0:
                attn_output = cuda_bmm_fA_qB_outer(group_size, attn_weights[:, :, :, :-value_full_length], value_code, value_scale, value_mn, bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_unq)
            else :
                attn_output = cuda_bmm_fA_qB_outer(group_size, attn_weights, value_code, value_scale, value_mn, bits)
                
    elif bits == 4 or bits == 8:
        if value_code is not None:
            values_ = unpack_and_dequant_vcache(value_code, value_scale, value_mn, group_size, bits)
        else :
            values_ = None
        
        if value_unq is not None:
            if values_ is not None:
                values_ = torch.cat([values_, value_unq], dim=-2)
            else:
                values_ = value_unq
        
        attn_output = torch.matmul(attn_weights, values_)
        
    else:
        raise NotImplementedError(f"Quantization bits {bits} not supported")
    
    return attn_output

def pack_along_last_dim(input_tensor: torch.Tensor, group_size: int, quant_bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if quant_bits == 2:
        return triton_quantize_and_pack_along_last_dim(input_tensor, group_size, quant_bits)
    else:
        return quant_and_pack_vcache(input_tensor, group_size, quant_bits)

class QuantizedCache(DynamicCache):
    '''
    The kv cache here is (key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq)
    '''
    def __init__(self, quant_bits:int = 2, group_size:int = 16, residual_length:int = 128, num_layers = 32):
        super().__init__()
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.num_layers = num_layers
        assert self.residual_length % self.group_size == 0, f"residual_length should be divisible by group_size, {self.residual_length = } % {self.group_size = } != 0"
        
        self.kv_seq_len = 0
        self._seen_tokens = 0
        self.quantized_cache = [{} for i in range(self.num_layers)] # stores quantized/unq KV values
        
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[dict] = None
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        if key_states.shape[-2] != 1: # the pre-fill phase
            num_keys = key_states.shape[-2]
            if num_keys < self.group_size:
                key_code, key_scale, key_mn = None, None, None
                key_unq = key_states
            else :
                if num_keys % self.group_size != 0:
                    key_unq = key_states[:, :, -(num_keys % self.group_size):, :]
                    key_states_to_quant = key_states[:, :, :-(num_keys % self.group_size), :].transpose(2,3).contiguous()
                    key_code, key_scale, key_mn = pack_along_last_dim(key_states_to_quant, self.group_size, self.quant_bits)
                else :
                    key_code, key_scale, key_mn = pack_along_last_dim(key_states.transpose(2,3).contiguous(), self.group_size, self.quant_bits)
                    key_unq = None
                    
            num_values = value_states.shape[-2]
            if num_values < self.group_size:
                value_code, value_scale, value_mn = None, None, None
                value_unq = value_states
            else :
                if num_values % self.group_size != 0:
                    value_unq = value_states[:, :, -(num_values % self.group_size):, :]
                    value_states_to_quant = value_states[:, :, :-(num_values % self.group_size), :].contiguous()
                    value_code, value_scale, value_mn = pack_along_last_dim(value_states_to_quant, self.group_size, self.quant_bits)
                else :
                    value_code, value_scale, value_mn = pack_along_last_dim(value_states.contiguous(), self.group_size, self.quant_bits)
                    value_unq = None
                    
            # update cache
            self.quantized_cache[layer_idx] = {
                "key_code": key_code, "key_scale": key_scale, "key_mn": key_mn, "key_unq": key_unq,
                "value_code": value_code, "value_scale": value_scale, "value_mn": value_mn, "value_unq": value_unq
            }
        
        else :  # the generation phase
            key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq = self.quantized_cache[layer_idx].values()
            
            if key_unq is not None:
                key_unq = torch.cat([key_unq, key_states], dim = 2)
            else:
                key_unq = key_states
            
            if key_unq.shape[-2] >= self.residual_length:
                key_code_new, key_scale_new, key_mn_new = pack_along_last_dim(key_unq.transpose(2,3).contiguous(), self.group_size, self.quant_bits)
                key_unq = None
                if key_code is not None:
                    key_code = torch.cat([key_code, key_code_new], dim = 3)
                    key_scale = torch.cat([key_scale, key_scale_new], dim = 3)
                    key_mn = torch.cat([key_mn, key_mn_new], dim = 3)
                else:
                    key_code, key_scale, key_mn = key_code_new, key_scale_new, key_mn_new
                    
            if value_unq is not None:
                value_unq = torch.cat([value_unq, value_states], dim = 2)
            else:
                value_unq = value_states
                
            if value_unq.shape[-2] >= self.residual_length:
                value_code_new, value_scale_new, value_mn_new = pack_along_last_dim(value_unq.contiguous(), self.group_size, self.quant_bits)
                value_unq = None
                if value_code is not None:
                    value_code = torch.cat([value_code, value_code_new], dim = 2)
                    value_scale = torch.cat([value_scale, value_scale_new], dim = 2)
                    value_mn = torch.cat([value_mn, value_mn_new], dim = 2)
                else:
                    value_code, value_scale, value_mn = value_code_new, value_scale_new, value_mn_new

            # update cache
            self.quantized_cache[layer_idx] = {
                "key_code": key_code, "key_scale": key_scale, "key_mn": key_mn, "key_unq": key_unq,
                "value_code": value_code, "value_scale": value_scale, "value_mn": value_mn, "value_unq": value_unq
            }
            
        return key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return 0
    
    def reset(self):
        raise RuntimeError(f"Shouldn't be calling reset explicitly on {self.__class__.__name__}")
