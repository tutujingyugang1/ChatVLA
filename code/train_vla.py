import gc
import pickle

import os

from gitdb.util import exists

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
from dataclasses import dataclass, field, fields, asdict
import transformers
from data_utils.utils import load_data  # data functions
from data_utils.utils import compute_dict_mean, set_seed  # helper functions
from typing import Dict, Optional, Sequence, List
from policy_heads import *
from qwen2_vla.models.configuration_qwen2_vla import Qwen2VLAConfig
from qwen2_vla.models.modeling_qwen2_vla import Qwen2VLForConditionalGenerationForVLA
from aloha_scripts.constants import TASK_CONFIGS
from data_utils.utils import Qwen2VLAProcess
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen2_vla import QWen2VLATrainer
from data_utils.data_collator import *
import IPython

e = IPython.embed
from data_utils.data_collator import DataCollatorForSupervisedDataset
from qwen2_vla import model_load_utils as ml_utils
import torch

local_rank = None


#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@dataclass
class ActionHeadArguments:
    policy_head_type: str = field(default="scale_dp_policy") # or unet_diffusion_policy
    policy_head_size: str = field(default="ScaleDP_L") # ScaleDP_XL, ScaleDP_L, ScaleDP_B, ScaleDP_S
    state_dim: int = 7 # state dimension
    action_dim: int = 10 # action dimension

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_pretrain: Optional[str] = field(default="")  # pretrained model weights path

    with_llm_head: bool = field(default=False)

    using_moe: bool = field(default=False)



@dataclass
class DataArguments:
    episode_first: bool = True  # batchsampler will samples episode index first and then samples timestepsF
    use_reasoning: bool = False
    task_name: str = field(default="stack_cube_2024_6_2")
    skip_mirrored_data: bool = field(default=False)
    chunk_size: int = field(default=16)
    delta_control: bool = field(default=False)
    vl_ratio: float = field(default=-1) # -1 represents use ALL VL DATA


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)

    min_lr: Optional[float] = None
    min_lr_rate: Optional[float] = None

    remove_unused_columns: bool = field(default=False)
    with_flash_attention: bool = field(default=True)
    is_local_debug:bool = field(default=False)

    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)

    head_lr: Optional[float] = None
    resume_from_checkpoint: bool = field(default=False)
    llm_loss_weight: float = field(default=1.0)

    seed: int = field(default=0)

    # moe
    init_moe: bool = field(default=False)
    freeze_vl_expert: bool = field(default=False)

    # logger
    logging_dir: str = field(default='./logs')  # TensorBoard日志的保存目录
    logging_strategy: str = field(default='steps')  # 设置为`steps`表示每几步记录一次日志
    logging_steps: int = field(default=10)


    save_steps: int = field(default=10)
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=5000)

    # validate
    do_eval: bool = field(default=False)
    evaluation_strategy: str = field(default="no")
    eval_steps: int = field(default=200)
    per_device_eval_batch_size: int = field(default=32)

    load_pretrain: bool = False
    pretrain_image_size: int = 480  # default 270 x 480 and pretrain may be 180 x 320

    dataloader_pin_memory: bool = False
    # lora
    lora_enable: bool = False
    lora_module: str = "vit llm"
    lora_task_type: str = 'CAUSAL_LM'
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    non_lora_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def parse_param():
    global local_rank
    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments))
    model_args, data_args, training_args, action_head_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **asdict(action_head_args))
    if action_head_args.policy_head_type == 'scale_dp_policy': # scaledp, using dit block
        config.policy_head_size = action_head_args.policy_head_size
        config.policy_head_config = AutoConfig.for_model(model_type=action_head_args.policy_head_type,
                                                       model_size=action_head_args.policy_head_size,
                                                       cond_dim=config.hidden_size, action_dim=action_head_args.action_dim,
                                                        prediction_horizon=data_args.chunk_size,
                                                       state_dim=action_head_args.state_dim)
    elif action_head_args.policy_head_type == 'unet_diffusion_policy':
        config.policy_head_config = AutoConfig.for_model(model_type=action_head_args.policy_head_type,
                                                       global_cond_dim=config.hidden_size, action_dim=action_head_args.action_dim,
                                                         prediction_horizon=data_args.chunk_size,
                                                       state_dim=action_head_args.state_dim)
    else:
        raise NotImplementedError(f"Unsupported policy head type {action_head_args.policy_head_type}.")
    setattr(config.policy_head_config, "input_dim", asdict(action_head_args)['action_dim'])
    setattr(config.policy_head_config, "state_dim", asdict(action_head_args)['state_dim'])


    for k in ['with_llm_head', 'using_moe']:
        setattr(config, k, asdict(model_args)[k])
    config.llm_loss_weight = training_args.llm_loss_weight
    config.with_flash_attention = training_args.with_flash_attention
    if training_args.lr_scheduler_type == 'cosine_with_min_lr':
        training_args.lr_scheduler_kwargs['min_lr'] = training_args.min_lr
        training_args.lr_scheduler_kwargs['min_lr_rate'] = training_args.min_lr_rate


    return model_args, data_args, training_args, action_head_args, config


def train_bc(train_dataset=None, val_dataset=None, model=None, config=None, sampler_params=None, tokenizer=None,
             processor=None):
    set_seed(config['training_args'].seed)
    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if config['training_args'].bf16 else torch.float32))
    data_collator = DataCollatorForSupervisedDataset(multimodal_processor=processor, computed_type=compute_dtype,
                                                     tokenizer=tokenizer)

    model.config.use_cache = True
    model.config.save_pretrained(config['training_args'].output_dir)

    data_module = dict(train_dataset=train_dataset,
                       data_collator=data_collator,
                       eval_dataset=val_dataset
                       )
    trainer = QWen2VLATrainer(model=model,
                              tokenizer=tokenizer,
                              args=config['training_args'],
                              sampler_params=sampler_params,
                              **data_module)

    trainer.train(resume_from_checkpoint=config['training_args'].resume_from_checkpoint)

    trainer.save_state()

    model.config.use_cache = True

    if config['training_args'].lora_enable:
        state_dict = ml_utils.get_peft_state_maybe_zero_3(
            model.named_parameters(), config['training_args'].lora_bias
        )
        non_lora_state_dict = ml_utils.get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if config['training_args'].local_rank == 0 or config['training_args'].local_rank == -1:
            model.config.save_pretrained(config['training_args'].output_dir)
            model.save_pretrained(config['training_args'].output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict,
                       os.path.join(config['training_args'].output_dir, 'non_lora_trainables.bin'))
    else:
        ml_utils.safe_save_model_for_hf_trainer(trainer=trainer,
                                                output_dir=config['training_args'].output_dir)


def main(all_config=None, model_config=None):
    set_seed(1)
    # get task parameters
    task_config = TASK_CONFIGS[all_config['data_args'].task_name]
    dataset_dir = task_config['dataset_dir']
    vl_file = task_config.get('vl_file', None)
    vl_image_dir = task_config.get('vl_image_dir', None)

    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    name_filter = task_config.get('name_filter', lambda n: True)

    all_config['camera_names'] = camera_names
    all_config['episode_len'] = episode_len

    # load model and processor
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        all_config['model_args'].model_name_or_path,
    )

    multimodal_processor = AutoProcessor.from_pretrained(all_config['model_args'].model_name_or_path)
    model, data_args = ml_utils.load_model(config=all_config, qwen2_vla_config=model_config, rank0_print=rank0_print,
                                           tokenizer=tokenizer)
    vla_process = Qwen2VLAProcess(tokenizer=tokenizer, multimodal_processor=multimodal_processor,
                                  camera_names=camera_names)

    # load dataset (containing both robotic data with h5 format and vl data with json format)
    train_dataset, val_dataset, stats = load_data(dataset_dir, name_filter, camera_names,
                                                                  all_config['data_args'].chunk_size,
                                                                  vl_file=vl_file,
                                                                  vl_image_dir=vl_image_dir,
                                                                  skip_mirrored_data=all_config['data_args'].skip_mirrored_data,
                                                                  config=all_config,
                                                                  stats_dir_l=stats_dir,
                                                                  policy_head_type=all_config['action_head_args'].policy_head_type,
                                                                  llava_pythia_process=vla_process,
                                                                  vl_ratio=all_config['data_args'].vl_ratio,
                                                                  is_local_debug=all_config['training_args'].is_local_debug
                                                                  )

    os.makedirs(all_config['training_args'].output_dir, exist_ok=True)
    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # training start
    train_bc(train_dataset=train_dataset, model=model, val_dataset=val_dataset, config=all_config, tokenizer=tokenizer, processor=multimodal_processor)


if __name__ == '__main__':

    model_args, data_args, training_args, action_head_args, model_config = parse_param()
    config = {
        'model_args': model_args,
        'data_args': data_args,
        'training_args': training_args,
        'action_head_args': action_head_args,
    }


    config_dict = {k: asdict(v) if not isinstance(v, dict) else v for k, v in config.items()}

    ckpt = os.path.join(config['training_args'].output_dir, f"checkpoint-{config['training_args'].save_steps}")
    if os.path.exists(ckpt):
        config['training_args'].resume_from_checkpoint = True
        print("Resuming Training............")
    main(all_config=config, model_config=model_config)
    pass