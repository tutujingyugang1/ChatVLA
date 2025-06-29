import os
from qwen2_vla.model_load_utils import load_model_for_eval

import torch
from torchvision import transforms
import cv2

import pickle
import numpy as np
import time

from aloha_scripts.constants import FPS

from data_utils.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, \
    postprocess_base_action  # helper functions
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from einops import rearrange
import torch_utils as TorchUtils
# import matplotlib.pyplot as plt
import sys
from policy_heads import *
# from cv2 import aruco
from qwen2_vla.utils.image_processing_qwen2_vla import *
from qwen2_vla.utils.processing_qwen2_vla import *
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

import copy


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def process_obs(obs, states, stats):
    """
    obs: three cameras' images
    states: Tensor, robot states
    stats: mean, std of robot states and actions
    This function is used to get observations(images and robot states) in your robot environment.
    """
    cur_left_wrist = obs['left_wrist']
    cur_right_wrist = obs['right_wrist']
    cur_top = obs['top']
    assert np.max(cur_left_wrist) > 1, "All images must be 0-255."
    traj_rgb_np = np.array([cur_top, cur_left_wrist, cur_right_wrist]) # sequential must align with constants.py
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)

    return traj_rgb_np, cur_state # images, states

def time_ms():
    return time.time_ns() // 1_000_000


def convert_actions(pred_action):
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    return pred_action


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

    def datastruct_droid2qwen2vla(self, raw_lang):
        messages = [
            {
                "role": "user",
                "content": [
                ],
            },
        ]
        for i in range(3):
            messages[0]['content'].append(
                {
                    "type": "image",
                    "image": None,
                }
            )
        messages[0]['content'].append({"type": "text", "text": f""})
        messages[0]['content'][-1]['text'] = raw_lang
        return messages

    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # left, right ,wrist
        image_list = []
        for i, each in enumerate(image_data):
            ele = {}
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            ele['image'] = each
            if i == 2 :
                ele['resized_height'] = 56
                ele['resized_width'] = 56
            else:
                ele['resized_height'] = 240
                ele['resized_width'] = 320
            each = fetch_image(ele)
            image_list.append(torch.from_numpy(np.array(each)))

        image_data = image_list
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_inputs = None
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict


def eval_bc(policy, env, policy_config, raw_lang=None, eval_in_vqa=False, query_frequency=16):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    rand_crop_resize = False
    model_config = policy.config.policy_head_config


    policy.policy.eval()

    ## 4. load data stats(min,max,mean....) and define post_process####################################
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'scale_dp_policy' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    #############################################################################################################


    from collections import deque
    action_queue = deque(maxlen=query_frequency)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):

        env.reset(randomize=False)
        print(f"env has reset!")

        with torch.inference_mode():
            for t in range(max_timesteps):

                obs, states = env.get_obs()

                ### 5. Realize the function of get_obs###################
                traj_rgb_np, robot_state = process_obs(obs, states, stats)
                #########################################################
                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    ### 6. Augment the images if needed ##########################################################
                    curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        print('original size', original_size)
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)
                    ####################################################################################

                    ##### 7. Process inputs and predict actions ####################################################

                    batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                    all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True,
                                                                      tokenizer=policy.tokenizer, eval_in_vqa=eval_in_vqa)
                    action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])

                raw_action = action_queue.popleft()

                print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                ### 8. post process actions##########################################################
                action = post_process(raw_action)
                #####################################################################################
                print(f"after post_process action size: {action.shape}")
                action = convert_actions(action.squeeze())
                print(f'step {t}, pred action: {outputs}{action}')
                ##### Execute ######################################################################
                action_info = env.step(action.tolist())
                ####################################################################################


    return

class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self):
        pass

    def step(self, action):
        print("Execute action successfully!!!")

    def reset(self):
        print("Reset to home position.")

    def get_obs(self):
        img = np.random.rand(480, 640, 3) * 255
        obs = {
            'left_wrist': img,
            'right_wrist': img,
            'top': img,
        }
        states = np.zeros(14)
        return obs, states


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'
    query_frequency = 16
    policy_config = {
        ############### 1. Specify path to trained ChatVLA(Required)#############################
        "model_path": "/path/to/save/ChatVLA_qwen2_vl_stage_1/checkpoint-60000",
        #############################################################################
        "model_base": "/path/to/origin/qwen2_vl",
        "pretrain_path": None,
        "enable_lora": False,
        "action_head": action_head,
        'save_model': False,
    }
    global im_size
    im_size = 320
    eval_in_vqa=False

    raw_lang = 'Remove the towel from the shelf.'

    #### 2. Initialize robot env(Required)##########
    # Replace to Real robot env
    deploy_env = FakeRobotEnv()
    deploy_env.reset()



    #### 3. Load ChatVLA####################
    policy = qwen2_vla_policy(policy_config)
    eval_bc(policy, deploy_env, policy_config, raw_lang=raw_lang, eval_in_vqa=eval_in_vqa, query_frequency=query_frequency)
