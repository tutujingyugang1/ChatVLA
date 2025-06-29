from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from fontTools.misc.cython import returns

from ..base import BaseModel
from .prompt import Qwen2VLAPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from .modeling_qwen2_vla import Qwen2VLForConditionalGenerationForVLA
from .configuration_qwen2_vla import Qwen2VLAConfig, Qwen2VLAVisionConfig
from .model_load_utils import load_model_for_eval

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLA(Qwen2VLAPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        save_model: bool = False,
        eval_in_vqa: bool = False,
        use_cache: bool = True,
        model_base: str |None = None,
        return_action: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eval_in_vqa=eval_in_vqa,
            use_cache=use_cache,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2

        from transformers import Qwen2VLProcessor
        rank, world_size = get_rank_and_world_size()

        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, )
        self.return_action = return_action

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0


        action_head = 'None'
        model_size = '2B'
        policy_config = {
            "model_path": model_path,
            "model_base": model_base,
            "pretrain_path": None,
            "enable_lora": True,
            "temp_agg": False,
            "action_head": action_head,
            'model_size': model_size,
            'save_model': save_model
        }

        # If only one process and GPU memory is less than 40GB
        if auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(
                model_path, torch_dtype='bfloat16', device_map='auto',
                # attn_implementation='flash_attention_2'
            )
        elif '72b' not in self.model_path.lower():
            self.policy = qwen2_vla_policy(policy_config)
            self.model = self.policy.policy
            self.processor = self.policy.multimodal_processor
            self.model.eval()

        else:
            self.model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(),
                # attn_implementation='flash_attention_2'
            )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        if self.return_action:
            all_actions, outputs = self.model.evaluate(
                **inputs,
                **self.generate_kwargs,
                is_eval=True,)
            print('*'*50)
            print(all_actions)
            return tuple(all_actions, outputs)
        else:
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
                is_eval=True,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
            if self.verbose:
                print(f'\033[32m{response}\033[0m')
            return response

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from PIL import Image
from qwen_vl_utils import fetch_image
import numpy as np
class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]
        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(
            model_path=model_path,
            model_base=model_base, policy_config=policy_config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})
        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
