#!/bin/bash

source_dir=/path/to/save/stage_1_lora
#source_dir=/media/jz08/HDD/zhouzy/model_param/local_debug/chatvla_lora
echo 'tranfer checkpoints to non_lora_trainables.bin'
for dir in "$source_dir"/* ; do
    if [[ "$(basename "$dir")" == *"checkpoint-5000"* ]]; then
      if ! find "$dir" -mindepth 1 -type f -name "non_lora_trainables.bin" | grep -q .; then
        python ./evaluate/zero_to_fp32.py ${source_dir}/$(basename "$dir") ${source_dir}/$(basename "$dir")/non_lora_trainables.bin
      fi
    fi
done

