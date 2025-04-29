
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVSelectionMechanism():
    def __init__(self, window_size = 64, prompt_sparsity_ratio = 0.25, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.prompt_sparsity_ratio = prompt_sparsity_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling
        logger.info(f"[SnapKV Selection Mechanism] {window_size = }, {prompt_sparsity_ratio = }, {kernel_size = }, {pooling = }")

    def update_kv(self, key_states, query_states, value_states):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if int(self.prompt_sparsity_ratio * q_len) < self.window_size:
            retained_tokens = max(1, int(self.prompt_sparsity_ratio * q_len))
            key_states = key_states[:, :, -retained_tokens:, :]; value_states = value_states[:, :, -retained_tokens:, :]
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == None:
                attn_cache = attn_weights_sum
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(int(self.prompt_sparsity_ratio * q_len) - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class PyramidSnapKVSelectionMechanism(SnapKVSelectionMechanism):
    def __init__(self, window_size = 64, prompt_sparsity_ratio = 0.25, kernel_size = 5, pooling = 'avgpool', layer_id = None, num_layers = None):
        self.window_size = window_size
        self.prompt_sparsity_ratio = prompt_sparsity_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling
        
        self.layer_id = layer_id
        self.num_layers = num_layers
        
        m = num_layers
        delta = 7
        k_total = self.prompt_sparsity_ratio * m
        k = [0.] * m
        k[m-1] = k_total / m / delta
        k[0] = 2 * k_total / m - k[m-1]
        for l in range(1, m-1):
            k[l] = k[0] + (k[m-1]-k[0]) / (m-1) * l
        self.prompt_sparsity_ratio = k[layer_id]
        self.prompt_sparsity_ratio = min(1, self.prompt_sparsity_ratio) # snapKV with more than 0.5 prompt sparsity ratio distributes too much budget at the lower layers
        
        logger.info(f"[Pyramid SnapKV Selection Mechanism] {window_size = }, {self.prompt_sparsity_ratio = }, {kernel_size = }, {pooling = }, {layer_id = }, {num_layers = }")

class H2OSelectionMechanism():
    def __init__(self, heavy_ratio = 0.25, recent_ratio = 0.25):
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        logger.info(f"[H2O Selection Mechanism] heavy_ratio : {heavy_ratio}, recent_ratio : {recent_ratio}")

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def update_kv(self, key_states, query_states, value_states, cumulative_attn_map):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_hh = int(q_len * self.heavy_ratio)
        num_rw = int(q_len * self.recent_ratio)
        
        if num_hh + num_rw >= q_len:
            return key_states, value_states
        else:
            hh_score = cumulative_attn_map

            select_hh_scores = hh_score[:, :, :q_len - num_rw]
            _, keep_topk = torch.topk(select_hh_scores, num_hh, dim=-1)
            keep_topk = keep_topk.sort().values

            keep_recent = torch.arange(q_len - num_rw, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)
            keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

            mask = torch.zeros(hh_score.shape, dtype=torch.bool).to(keep_topk.device)
            mask = mask.scatter(-1, keep_idx, 1)
            
            key_states_compress = key_states[mask].view(bsz, num_heads, -1, head_dim)
            value_states_compress = value_states[mask].view(bsz, num_heads, -1, head_dim)
            
            return key_states_compress, value_states_compress

class PyramidH2OSelectionMechanism(H2OSelectionMechanism):
    def __init__(self, heavy_ratio = 0.25, recent_ratio = 0.25, layer_id = None, num_layers = None):
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.layer_id = layer_id
        self.num_layers = num_layers
        
        m = num_layers
        delta = 7
        k_total = self.heavy_ratio * m
        k = [0.] * m
        k[m-1] = k_total / m / delta
        k[0] = 2 * k_total / m - k[m-1]
        for l in range(1, m-1):
            k[l] = k[0] + (k[m-1]-k[0]) / (m-1) * l
        self.heavy_ratio = k[layer_id]
        
        logger.info(f"[Pyramid Selection Mechanism] heavy_ratio : {self.heavy_ratio}, recent_ratio : {self.recent_ratio}, {self.layer_id = }, {self.num_layers = }")
            
def init_minikv(self, layer_id = None, num_layers = None):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
            
        if not self.config.use_snap:
            assert not self.config.use_snap and hasattr(self.config, 'heavy_ratio')
            if self.config.eviction_strategy == "uniform":
                self.kv_cluster = H2OSelectionMechanism(
                    heavy_ratio = self.config.heavy_ratio,
                    recent_ratio = self.config.recent_ratio,
                    )
            elif self.config.eviction_strategy == "pyramid":
                self.kv_cluster = PyramidH2OSelectionMechanism(
                    heavy_ratio = self.config.heavy_ratio,
                    recent_ratio = self.config.recent_ratio,
                    layer_id = layer_id, 
                    num_layers = num_layers,
                )
            else:
                raise ValueError(f"Eviction strategy '{self.config.eviction_strategy}' not supported")
            
        else:
            # assert self.config.eviction_strategy == "uniform", f"SnapKV integration currently supports only uniform eviction strategy and 16-bit quantization, not {self.config.eviction_strategy} and {self.config.quant_bits}"
            assert hasattr(self.config, 'prompt_sparsity_ratio')
            if self.config.eviction_strategy == "uniform":
                self.kv_cluster = SnapKVSelectionMechanism(
                    window_size = self.config.window_size, 
                    prompt_sparsity_ratio = self.config.prompt_sparsity_ratio, 
                    kernel_size = self.config.kernel_size,
                    pooling = self.config.pooling
                    )
            elif self.config.eviction_strategy == "pyramid":
                self.kv_cluster = PyramidSnapKVSelectionMechanism(
                    window_size = self.config.window_size, 
                    prompt_sparsity_ratio = self.config.prompt_sparsity_ratio, 
                    kernel_size = self.config.kernel_size,
                    pooling = self.config.pooling,
                    layer_id = layer_id, 
                    num_layers = num_layers,
                    )
                
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
