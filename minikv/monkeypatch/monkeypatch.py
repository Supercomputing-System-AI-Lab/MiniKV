from importlib.metadata import version
import warnings
import transformers
from minikv.monkeypatch.snap_minikv_llama_hijack_4_37 import \
        sparsity_llama_flash_attn2_forward, sparsity_prepare_inputs_for_generation_llama, \
        snap_minikv_llama_flash_attn2_forward, snap_minikv_prepare_inputs_for_generation_llama

from minikv.monkeypatch.minikv_llama_hijack_4_37 import \
        minikv_llama_flash_attn2_forward, minikv_prepare_inputs_for_generation_llama

from minikv.monkeypatch.snap_minikv_mistral_hijack_4_37 import \
    sparsity_mistral_flash_attn2_forward, sparsity_prepare_inputs_for_generation_mistral, \
    snap_minikv_mistral_flash_attn2_forward, snap_minikv_prepare_inputs_for_generation_mistral
    
from minikv.monkeypatch.minikv_mistral_hijack_4_37 import minikv_mistral_flash_attn2_forward, minikv_prepare_inputs_for_generation_mistral

from minikv.monkeypatch.mixtral_hijack_4_37 import mixtral_flash_attn2_forward as mixtral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mixtral as prepare_inputs_for_generation_mixtral_4_37

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama(args = None):
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with minikv. SnapKV is tested with Transformers version {version_list}.")
      
    if not args.use_snap:
        if args.quant_bits != 16:
            logger.info(f"Loading MiniKV fwd pass")
            transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = minikv_prepare_inputs_for_generation_llama
            transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = minikv_llama_flash_attn2_forward
        else:
            raise NotImplementedError(f"This configuration uses H2O during pre-fill and saves all the generated tokens, which is not the original H2O algo (and probably not what you want). Not supported: {args = }")
    
    else:
        if args.quant_bits != 16:
            logger.info(f"Loading Snap+MiniKV fwd pass")
            transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = snap_minikv_prepare_inputs_for_generation_llama
            transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = snap_minikv_llama_flash_attn2_forward
        else :
            logger.info(f"Loading SnapKV fwd pass")
            transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = sparsity_prepare_inputs_for_generation_llama
            transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = sparsity_llama_flash_attn2_forward

def replace_mistral(args = None):
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with minikv. SnapKV is tested with Transformers version {version_list}.")
    
    if not args.use_snap:
        if args.quant_bits == 2:
            transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = minikv_prepare_inputs_for_generation_mistral
            transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = minikv_mistral_flash_attn2_forward
        else:
            raise NotImplementedError(f"This configuration is not supported: {args = }")
        
    else:
        if args.quant_bits == 2:
            transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = snap_minikv_prepare_inputs_for_generation_mistral
            transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = snap_minikv_mistral_flash_attn2_forward
        else :
            logger.info(f"Loading SnapKV fwd pass")
            transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = sparsity_prepare_inputs_for_generation_mistral
            transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = sparsity_mistral_flash_attn2_forward
            
def replace_mixtral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with minikv. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mixtral_4_37
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = mixtral_flash_attn2_forward_4_37
