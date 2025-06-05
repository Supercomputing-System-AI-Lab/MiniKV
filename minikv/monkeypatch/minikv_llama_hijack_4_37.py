import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

from minikv.monkeypatch.minikv_utils import init_minikv
from minikv.monkeypatch.cache_impl import QuantizedCache, get_attn_weights, get_attn_output

import math
from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

logger = logging.get_logger(__name__)

##############
def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def minikv_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    init_minikv(self, layer_id = self.layer_idx, num_layers = self.config.num_hidden_layers)
    
    if isinstance(past_key_value, DynamicCache) and not isinstance(past_key_value, QuantizedCache):
        raise RuntimeError(f"past_key_value should be of type QuantizedCache, got {type(past_key_value)}")
        
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            
            if not self.config.use_eviction_flash:
                attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)
                
                attention_mask = _make_causal_mask(
                    bsz=bsz,
                    tgt_len=q_len,
                    past_key_values_length=0,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                    
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )
                
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                
                cumulative_attn_map = attn_weights.sum(2)
            else :
                ######pytorch version#######
                attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)
                
                attention_mask = _make_causal_mask(
                    bsz=bsz,
                    tgt_len=q_len,
                    past_key_values_length=0,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                    
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                    )
                
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output_torch = attn_output.transpose(1, 2).contiguous()
                
                cumulative_attn_map_torch = attn_weights.sum(2)
                ################################
                
                query_states_kernel = query_states.permute(0, 1, 2, 3) 
                key_states_kernel = key_states.permute(0, 1, 2, 3)
                value_states_kernel = value_states.permute(0, 1, 2, 3)
             
                from selection_kernel import selection_attention
                attn_output, cumulative_attn_map, LSE = selection_attention(
                    query_states_kernel.contiguous(),
                    key_states_kernel.contiguous(),
                    value_states_kernel.contiguous(),
                    True,
                    1.0 / math.sqrt(self.head_dim),
                )
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
                
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, cumulative_attn_map)   # only eviction here
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs) # quantization happens here
            
        else:
            self.kv_seq_len += q_len
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
            attn_weights = get_attn_weights(self.config.group_size, query_states, key_code, key_scale, key_mn, key_unq, self.config.quant_bits, self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = get_attn_output(self.config.group_size, attn_weights, value_code, value_scale, value_mn, value_unq, self.config.quant_bits)
                
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.transpose(1, 2).contiguous()   ## to match flash attn output shape
            
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def minikv_prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None or not isinstance(past_key_values, QuantizedCache):
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
        past_key_values = QuantizedCache(quant_bits = self.config.quant_bits, group_size = self.config.group_size, residual_length = self.config.residual_length, num_layers = self.config.num_hidden_layers)
        
    elif past_key_values is not None:
        cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
        max_cache_length = None
            
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
