import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)
from minikv.monkeypatch.minikv_utils import init_minikv
from minikv.monkeypatch.cache_impl import QuantizedCache

import math
from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

logger = logging.get_logger(__name__)

from selection_kernel import selection_attention

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

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

def minikv_mistral_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    init_minikv(self, layer_id = self.layer_idx, num_layers = self.config.num_hidden_layers)
    
    if isinstance(past_key_value, DynamicCache) and not isinstance(past_key_value, QuantizedCache):
        raise RuntimeError(f"past_key_value should be of type QuantizedCache, got {type(past_key_value)}")
    
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
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

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    # rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, "sliding_window", None) is not None
        and kv_seq_len > self.config.sliding_window
    )
    if use_sliding_windows:
        raise RuntimeError("MKV's output accuracy is tested on inputs of length 4k, where mistral's sliding window logic does not become active. Support for sliding window will come in future releases.")
    use_sliding_windows = False

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
            " make sure to upgrade flash-attn library."
        )
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        if key_states.shape[-2] >= kv_seq_len:
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
                attn_output, cumulative_attn_map, LSE = selection_attention(
                    query_states.permute(0,2,1,3),
                    key_states.permute(0,2,1,3),
                    value_states.permute(0,2,1,3),
                    True,
                    1.0 / math.sqrt(self.head_dim),
                )
                cumulative_attn_map = cumulative_attn_map.permute(0,2,1)
            
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, cumulative_attn_map)   # only eviction here
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs) # quantization happens here
            
        else:
            self.kv_seq_len += q_len
            key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
            # call mkv kernels here to get attn map + attn output
            if key_code is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.config.group_size, query_states, key_code, key_scale, key_mn, self.config.quant_bits)
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
            attn_weights = torch.cat(attn_list, dim=-1) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_full_length = value_unq.shape[-2] if value_unq is not None else 0
            if value_code is None:
                attn_output = torch.matmul(attn_weights, value_unq)
            else:
                if value_full_length != 0:
                    attn_output = cuda_bmm_fA_qB_outer(self.config.group_size, attn_weights[:, :, :, :-value_full_length], value_code, value_scale, value_mn, self.config.quant_bits)
                    attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_unq)
                else :  # list[:-0] does not work
                    attn_output = cuda_bmm_fA_qB_outer(self.config.group_size, attn_weights, value_code, value_scale, value_mn, self.config.quant_bits)
                
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.transpose(1, 2).contiguous()   ## to match flash attn output shape

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def minikv_prepare_inputs_for_generation_mistral(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # Omit tokens covered by past_key_values
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
        past_key_values = QuantizedCache(quant_bits = self.config.quant_bits, group_size = self.config.group_size, residual_length = self.config.residual_length, num_layers = self.config.num_hidden_layers)
        
    if past_key_values is not None:
        cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
        max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
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

    # print('prepare position_ids', position_ids)
    # print('prepare input shape', input_ids.shape)
    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
