#!/bin/bash
TASK=example_tasks_stage_2
LLM=qwen2_vl
LLM_MODEL_SIZE=2B
ACTION_HEAD="scale_dp_policy" # unet_diffusion_policy or scale_dp_policy
MNOP=/path/to/save/stage_1/checkpoint-5000 # your stage-1 weights
OUTPUT=/path/to/save/ChatVLA_qwen2_vl_stage_2


mkdir -p $OUTPUT/src
cp -r ./scripts $OUTPUT/
cp -r ./data_utils $OUTPUT/src/
cp -r ./qwen2_vla $OUTPUT/src/
cp -r ./policy_heads $OUTPUT/src/

deepspeed --master_port 29607 --num_gpus=8 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --action_dim 10 \
  --state_dim 7 \
  --chunk_size 16 \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_L" \
  --resume_from_checkpoint False \
  --with_llm_head True \
  --pretrain_image_size 320 \
  --task_name ${TASK} \
  --model_name_or_path ${MNOP} \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --bf16 True \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 2000 \
  --max_steps 10000 \
  --save_total_limit 100 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr"\
  --min_lr 0\
  --logging_steps 4 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --output_dir $OUTPUT \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log \
  --lora_enable False \
  --vl_ratio 0.33 \
  --using_moe True \
  --init_moe False | tee $OUTPUT/log.log


for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${mnop}/preprocessor_config.json $dir
        cp ${mnop}/chat_template.json $dir
    fi
done
mv ./60030.log $OUTPUT
echo $OUTPUT
