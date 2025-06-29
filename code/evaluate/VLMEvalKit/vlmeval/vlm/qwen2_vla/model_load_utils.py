import warnings
import torch
import transformers
import json
import logging
from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import os


# def find_all_linear_names(model, rank0_print, lora_module=None):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#
#     multimodal_keywords = ['multi_modal_projector', 'policy_head', 'lm_head',
#                            'xattn', 'input_action_proj', 'reasoning_action_proj', 'reasoning_film', 'merger', 'expert']
#     if 'vit' not in lora_module:
#         multimodal_keywords.append("vision_tower")
#     if 'llm' not in lora_module:
#         multimodal_keywords.append("language_model")
#     rank0_print("##" * 20)
#
#     for name, module in model.named_modules():
#         if any(mm_keyword in name for mm_keyword in multimodal_keywords):
#             continue
#
#         if isinstance(module, cls):
#             lora_module_names.add(name)
#
#     if 'lm_head' in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove('lm_head')
#
#     return list(lora_module_names)
#
#
# def load_model(config=None, qwen2_vla_config=None, rank0_print=print, tokenizer=None):
#     model_args = config['model_args']
#     training_args = config['training_args']
#     data_args = config['data_args']
#     if training_args.load_pretrain:
#         kwargs = {"device_map": "cuda", "torch_dtype": torch.bfloat16}
#         rank0_print("@@@@@@@Loading pretrain weights...@@@@@@@@@@")
#         assert config[
#                    'model_args'].model_pretrain is not "", "load pretrain weights need set the model_pretrain in DataArguments!!!!"
#         model_path = config['model_args'].model_pretrain
#         model_base = config['model_args'].model_name_or_path
#         print('Loading pretrained <<QWEN2_VLA>> from base models...')
#         model = AutoModelForCausalLM.from_pretrained(
#             model_base,
#             config=qwen2_vla_config,
#             cache_dir=config['training_args'].cache_dir,
#             trust_remote_code=True,
#             _fast_init=False,
#             attn_implementation="flash_attention_2",
#         )
#         print('Loading pretrained additional <<QWEN2_VLA>> weights...')
#         if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#             non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
#         else:
#             raise f"there is no non_lora_trainables.bin in {model_path}"
#         non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
#                                non_lora_trainables.items()}
#         if any(k.startswith('model.policy_head.') for k in non_lora_trainables):
#             non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#
#         keys_to_del = []
#         if config['action_head_args'].action_dim == 14:
#             print("Deleting some modules to adapt for bimanual setting....")
#             for name in ['policy_head.combine.weight', 'policy_head.down_modules.0.0.blocks.0.block.0.weight',
#                          'policy_head.down_modules.0.0.residual_conv.weight', 'policy_head.final_conv.1.weight',
#                          'policy_head.final_conv.1.bias']:
#                 keys_to_del.append(name)
#         for k, v in non_lora_trainables.items():
#             if 'lora' in k:
#                 keys_to_del.append(k)
#         for key in keys_to_del:
#             del non_lora_trainables[key]
#
#         model.load_state_dict(non_lora_trainables, strict=False)
#
#         from peft import PeftModel
#         print('Loading LoRA weights...')
#         model = PeftModel.from_pretrained(model, model_path)
#         print('Merging LoRA weights...')
#         model = model.merge_and_unload()
#         print('Model is loaded...')
#         model.to(torch.bfloat16)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             config['model_args'].model_name_or_path,
#             config=qwen2_vla_config,
#             cache_dir=config['training_args'].cache_dir,
#             trust_remote_code=True,
#             _fast_init=False,
#             # attn_implementation="flash_attention_2",
#         )
#
#     #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> initialize moe weights <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     if config['model_args'].using_moe:
#         mlp_weights_path = os.path.join(config['model_args'].model_name_or_path, "mlp.bin")
#         assert os.path.isfile(mlp_weights_path), "No MLP params found at {}".format(
#             config['model_args'].model_name_or_path)
#         params = torch.load(mlp_weights_path)
#         loaded = {}
#         for k, v in params.items():
#             loaded[k.replace('mlp', 'mlp_lan_expert')] = v
#             # loaded[k.replace('mlp', 'mlp_vision_expert')] = v
#             # loaded[k.replace('mlp', 'mlp_vision_expert.experts.0')] = v
#             # loaded[k.replace('mlp', 'mlp_vision_expert.experts.1')] = v
#             loaded[k.replace('mlp', 'mlp_vision_expert.0')] = v
#             loaded[k.replace('mlp', 'mlp_vision_expert.1')] = v
#         del params
#         model.load_state_dict(loaded, strict=False)
#         rank0_print("MOE adapter initialized.")
#
#     if config['model_args'].using_vision_moe:
#         vision_mlp_weights_path = os.path.join(config['model_args'].model_name_or_path, "vision_mlp.bin")
#         assert os.path.isfile(vision_mlp_weights_path), "No Vision MLP params found at {}".format(
#             config['model_args'].model_name_or_path)
#         params = torch.load(vision_mlp_weights_path)
#         loaded = {}
#         for k, v in params.items():
#             loaded[k.replace('mlp', 'smoe.experts.0')] = v
#             loaded[k.replace('mlp', 'smoe.experts.1')] = v
#         del params
#         model.load_state_dict(loaded, strict=False)
#         rank0_print("Vision MOE adapter initialized.")
#
#     ############################################ setting pretrained dit ####################################################
#     if config['model_args'].pretrain_dit_path is not None and config['action_head_args'].policy_class == 'dit_diffusion_policy':
#         assert config['model_args'].pretrain_dit_path is not None, "please specify a pretrained dit path when setting load_pretrain_dit==True"
#         rank0_print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Loading pretrained dit weights...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#         pretrain_dit_weights = torch.load(config['model_args'].pretrain_dit_path, map_location='cpu')['nets']['ema']
#         keys_to_del_dit = []
#         pretrain_dit_weights = {k[7:] if k.startswith('policy.') else k: v for k, v in pretrain_dit_weights.items()}
#         for k in pretrain_dit_weights.keys():
#             if 'noise_pred' not in k:
#                 keys_to_del_dit.append(k)
#             if 'cond_obs_emb' in k:
#                 keys_to_del_dit.append(k)
#
#         for k in keys_to_del_dit:
#             del pretrain_dit_weights[k]
#         pretrain_dit_weights = {k[15:] if k.startswith('noise_pred_net.') else k: v for k, v in pretrain_dit_weights.items()}
#
#         model.policy_head.load_state_dict(pretrain_dit_weights, strict=False)
#     #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> setting training weights <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     model.config.use_cache = False
#     model_args.freeze_backbone = training_args.freeze_backbone
#     if model_args.freeze_backbone:
#         model.requires_grad_(False)
#     else:
#         model.requires_grad_(True)
#
#     #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> activate visual param <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     model.visual.requires_grad_(True)  # set to true first
#     model.config.freeze_vision_tower = model_args.freeze_vision_tower = training_args.freeze_vision_tower
#     if model_args.freeze_vision_tower:
#         for n, p in model.visual.named_parameters():
#             if not 'lora' in n.lower():
#                 p.requires_grad = False
#     else:
#         for p in model.visual.parameters():
#             p.requires_grad = True
#
#     if training_args.gradient_checkpointing:
#         if hasattr(model, "enable_input_require_grads"):
#             model.enable_input_require_grads()
#         else:
#             def make_inputs_require_grad(module, input, output):
#                 output.requires_grad_(True)
#
#             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
#
#     if training_args.lora_enable:
#         from peft import LoraConfig, get_peft_model
#         lora_config = LoraConfig(
#             r=training_args.lora_r,
#             lora_alpha=training_args.lora_alpha,
#             target_modules=find_all_linear_names(model, rank0_print, training_args.lora_module),
#             lora_dropout=training_args.lora_dropout,
#             bias=training_args.lora_bias,
#             task_type=training_args.lora_task_type,
#         )
#         if training_args.bits == 16:
#             if training_args.bf16:
#                 model.to(torch.bfloat16)
#             if training_args.fp16:
#                 model.to(torch.float16)
#         rank0_print("##" * 20)
#
#         rank0_print("Adding LoRA adapters...")
#         model = get_peft_model(model, lora_config)  # !!!only set lora weights to requires_grad True!!!
#         rank0_print(model)
#         model.print_trainable_parameters()
#     elif training_args.load_pretrain:
#         rank0_print("Already loaded pretrained weights which is based on lora, skipping LoRA initialize...")
#
#     if config['model_args'].using_moe:
#         for name, module in model.named_modules():
#             if any(mm_keyword in name for mm_keyword in ['expert']):
#                 module.requires_grad_(True)
#
#     if config['model_args'].with_llm_head:
#         model.lm_head.requires_grad_(True)
#
#     #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> activate action head <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#     model.policy_head.requires_grad_(True)
#     if config['model_args'].using_film:
#         model.input_action_proj.requires_grad_(True)
#         model.reasoning_action_proj.requires_grad_(True)
#         model.reasoning_film.requires_grad_(True)
#
#     vision_tower = model.visual
#     vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
#     model.model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
#     model.to(torch.bfloat16)
#     for k, v in model.named_parameters():
#         if v.requires_grad:
#             rank0_print(k, v.requires_grad, v.dtype)
#
#     model.config.non_lora_lr = training_args.non_lora_lr
#
#     print("!" * 100)
#     lora_para = sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and 'lora' in n))
#     all_para = sum(p.numel() for n, p in model.named_parameters())
#     train_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
#     print(
#         f"Lora parameters/trainalbe parameters/all parameters:{lora_para / 1000000}M/{train_para / 1000000}M/{(all_para - lora_para) / 1000000}M")
#     return model, data_args
#
#
# def maybe_zero_3(param, ignore_status=False, name=None):
#     from deepspeed import zero
#     from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
#     if hasattr(param, "ds_id"):
#         if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
#             if not ignore_status:
#                 logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param
#
#
# # Borrowed from peft.utils.get_peft_model_state_dict
# def get_peft_state_maybe_zero_3(named_params, bias):
#     if bias == "none":
#         to_return = {k: t for k, t in named_params if "lora_" in k}
#     elif bias == "all":
#         to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
#     elif bias == "lora_only":
#         to_return = {}
#         maybe_lora_bias = {}
#         lora_bias_names = set()
#         for k, t in named_params:
#             if "lora_" in k:
#                 to_return[k] = t
#                 bias_name = k.split("lora_")[0] + "bias"
#                 lora_bias_names.add(bias_name)
#             elif "bias" in k:
#                 maybe_lora_bias[k] = t
#         for k, t in maybe_lora_bias:
#             if bias_name in lora_bias_names:
#                 to_return[bias_name] = t
#     else:
#         raise NotImplementedError
#     to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
#     return to_return
#
#
# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return
#
#
# def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
#     to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return
#
#
# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
#                                    output_dir: str):
#     """Collects the state dict and dump to disk."""
#
#     if trainer.deepspeed:
#         torch.cuda.synchronize()
#         trainer.save_model(output_dir)
#         return
#
#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {
#             key: value.cpu()
#             for key, value in state_dict.items()
#         }
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
#
#
# def load_merge_lora_weights(model_path=None, model_base=None, kwargs=None):
#     path = model_path.split('/')[0:-1]
#     root_path = '/'.join(path)
#
#     lora_cfg_pretrained = AutoConfig.from_pretrained(root_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)  # default use_fast=False
#     print('Loading QWen2-VLA from base model...')
#     model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
#                                                  config=lora_cfg_pretrained, **kwargs)
#
#     print('Loading additional QWen2-VLA weights expecially non-lora part(diffusion head)...')
#     if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#         non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
#                                          )
#     else:
#         raise "do not contain non_lora_trainables.bin, please check the non lora param part saved correctly!!!!"
#     non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
#                            non_lora_trainables.items()}
#     if any(k.startswith('model.policy_head.') for k in non_lora_trainables):
#         non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
#                                non_lora_trainables.items()}
#
#     keys_to_del = []
#     for k, v in non_lora_trainables.items():
#         if 'lora' in k:
#             keys_to_del.append(k)
#     for key in keys_to_del:
#         del non_lora_trainables[key]
#
#     model.load_state_dict(non_lora_trainables, strict=False)
#
#     from peft import PeftModel
#     print('Loading LoRA weights...')
#     model = PeftModel.from_pretrained(model, model_path)
#     print('Merging LoRA weights...')
#     model = model.merge_and_unload()
#     print('Model is loaded...')
#     return model, tokenizer


def load_model_for_eval(model_path, model_base, device_map="cuda",
                        policy_config=None):
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.bfloat16

    if policy_config['save_model']:
        kwargs['torch_dtype'] = torch.bfloat16

    if model_base is not None and '72B' in model_base:
        kwargs = {
            "device_map": "cpu",
            "max_memory": {0: "45GiB", 1: "45GiB", "cpu": "80GiB"},
            "offload_folder": "/home/eai/wjj/qwen2_vla/offload",
            "offload_state_dict": True,
        }
        with open(os.path.join(model_base, 'device_map.json'), 'r') as f:
            device_map = json.load(f)
        kwargs['device_map'] = device_map

    if 'qwen2' in model_path.lower() or 'qwen' in model_path.lower():
        assert model_path is not None
        print("load QWen2-VLA!!!")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_safetensors=True,
            **kwargs).to("cuda")
    else:
        raise "do not support this model!"

    multi_modal_processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    model.to(device="cuda")
    return tokenizer, model, multi_modal_processor, context_len
