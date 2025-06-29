import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Type, Optional

class StaticMoE(nn.Module):
    def __init__(self, config, expert_module_class: Type[nn.Module]):
        super(StaticMoE, self).__init__()
        self.experts = nn.ModuleList([expert_module_class(config) for _ in range(2)])
    def forward(self, x, vl_data_mask, eval_in_vqa=False):
        if self.training:
            vl_data_mask = vl_data_mask.type_as(x)
            mask_dim = x.shape[-2:]
            vl_data_mask = vl_data_mask.repeat(mask_dim[1], mask_dim[0], 1).transpose(0, 2)
            output = self.experts[0](x) * vl_data_mask \
                                + self.experts[1](x) * (1. - vl_data_mask)
        else:
            dim = x.shape[-1]
            if eval_in_vqa:
                output = self.experts[0](x.reshape(-1, dim)).unsqueeze(0)
            else:
                output = self.experts[1](x.reshape(-1, dim)).unsqueeze(0)
        return output
