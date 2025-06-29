import copy
from dataclasses import dataclass, field, fields, asdict, replace
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import sys
import torch

import transformers
import gc

from PIL import Image
import numpy as np
import os
from qwen_vl_utils import process_vision_info
from qwen_vl_utils import fetch_image, fetch_video


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    multimodal_processor: transformers.AutoProcessor=None
    computed_type: torch.dtype=None
    tokenizer: transformers.AutoTokenizer=None

    # @profile
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'].squeeze(0) for instance in instances]
        attention_mask = [instance['attention_mask'].squeeze(0) for instance in instances]
        labels = [instance['labels'].squeeze(0) for instance in instances]

        pixel_values = [instances['pixel_values'] for instances in instances if instances['pixel_values'] is not None]
        image_grid_thw = [instances['image_grid_thw'] for instances in instances if instances['image_grid_thw'] is not None]
        vl_data_mask = torch.concat([instances["vl_data_mask"] for instances in instances], dim=0).to(torch.bool)
        text_only_mask = torch.concat([instances["text_only_mask"] for instances in instances], dim=0).to(torch.bool)

        labels = torch.nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-100)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)

        input_ids = input_ids[:, :2048]
        labels = labels[:, :2048]

        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values, dim=0)
            image_grid_thw = torch.concat(image_grid_thw, dim=0)


        attention_mask = input_ids.ne(self.tokenizer.pad_token_id),

        if not isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.tensor(np.array([instance['action'] for instance in instances]))
            states = torch.tensor(np.array([instance['state'] for instance in instances]))
        else:
            actions = torch.stack([instance['action'] for instance in instances])
            states = torch.stack([instance['state'] for instance in instances])

        is_pad_all = torch.stack([instance['is_pad'] for instance in instances])

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask[0],
            labels=labels,
            image_grid_thw=image_grid_thw,
            actions=actions,
            states=states,
            pixel_values=pixel_values,
            is_pad=is_pad_all,
            vl_data_mask=vl_data_mask,
            text_only_mask=text_only_mask,
        )


        del input_ids
        del attention_mask
        del labels
        del pixel_values
        del actions
        del states
        del image_grid_thw
        del is_pad_all
        del vl_data_mask
        del text_only_mask
        gc.collect()
        torch.cuda.empty_cache()
        return batch


