import qwen2_vla
import policy_heads
import torch
import re
import os

from qwen2_vla import Qwen2VLForConditionalGenerationForVLA

p = '/path/to/origin/qwen2_vl' # official qwen2_vl weights
pattern_weights = r"model\.layers\.(\d+)\.mlp\.(.*)"
save_filename = os.path.join(p, "mlp.bin")
model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(p, torch_dtype=torch.bfloat16)

save_dict = {}
for name, param in model.named_parameters():
    match = re.match(pattern_weights, name)
    if match:
        save_dict[name] = param
torch.save(save_dict, save_filename)