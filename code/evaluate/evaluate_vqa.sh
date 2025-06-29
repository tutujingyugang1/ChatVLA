#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LMUData=/path/to/LMUData torchrun --master_port 29666 --nproc-per-node=8 run.py --config config_vla.json
CUDA_VISIBLE_DEVICES=0,1 LMUData=/path/to/LMUData torchrun --master_port 29666 --nproc-per-node=2 ./evaluate/VLMEvalKit/run.py --config ./evaluate/VLMEvalKit/config_vla.json
