import json
import random
from lib2to3.fixer_util import is_list
from typing import Dict, List

import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2

from PIL import Image
from time import time

from numpy.core.numeric import indices

from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import IPython
from transformers.models.llava_onevision.convert_llava_onevision_weights_to_hf import chat_template

e = IPython.embed
from qwen_vl_utils import *

colors = {
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'reset': '\033[0m'
}

def flatten_list(l):
    return [item for sublist in l for item in sublist]


import gc

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class,
                 llava_pythia_process=None, imsize=480, data_args=None, vl_file=None, vl_image_dir=None,
                 vl_ratio=0, is_local_debug=False, robot=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len) if episode_len is not None and len(episode_len)>0 else 0
        self.policy_class = policy_class
        self.llava_pythia_process = llava_pythia_process
        self.imsize = imsize
        self.data_args = data_args
        self.robot = robot
        self.is_local_debug = is_local_debug
        if 'diffusion' in self.policy_class.lower() or 'scale_dp' in self.policy_class.lower():
            self.augment_images = True
        else:
            self.augment_images = False

        # for vl data (w/o action and state. vl data only)
        if vl_file is not None:
            random.seed(42)
            vl_data_list_lens = []
            if isinstance(vl_file, list):
                vl_data_list = []
                for f in vl_file:
                    tmp_vl_list = json.load(open(f, 'r'))
                    vl_data_list_lens.append(len(tmp_vl_list))
                    vl_data_list.extend(tmp_vl_list)
                    del tmp_vl_list
            else:
                vl_data_list_lens.append(0)
                vl_data_list = json.load(open(vl_file, 'r'))
                vl_data_list_lens.append(len(vl_data_list))

            x = self.cumulative_len[-1] if len(self.cumulative_len) > 0 else 0
            print(
                f"######################## Ori {colors['red']}{'ROBOTICS DATA Length'}{colors['reset']} is {x}###################################")
            print(
                f"########################Process VL DATA for support {colors['red']}{'ROBOTICS DATA : VL DATA'}{colors['reset']} is {vl_ratio}###################################")

            if vl_ratio != -1 and x is not None and x>0:
                target_len = int(x * vl_ratio)
                vl_data_list_temp = vl_data_list
                len_vl = len(vl_data_list_temp)
                if target_len < len_vl:
                    indices = random.sample(range(vl_data_list_lens[0],len_vl), target_len-vl_data_list_lens[0])
                    indices.extend([i for i in range(vl_data_list_lens[0])])
                    vl_data_list = [vl_data_list_temp[i] for i in indices]
                else:
                    vl_data_list = [vl_data_list_temp[i % len_vl]for i in range(target_len)]
                del vl_data_list_temp
            self.episode_len.append(len(vl_data_list))
            self.cumulative_len = np.append(self.cumulative_len, x + len(vl_data_list))
            self.cumulative_len = self.cumulative_len.astype(int)
            x = max(self.episode_ids) if len(self.episode_ids)>0 else 0
            self.episode_ids = np.append(self.episode_ids, x+1)
            self.episode_ids = self.episode_ids.astype(int)
            print(
                f"########################Current {colors['red']}{'VL DATA Length'}{colors['reset']} is {len(vl_data_list)}###################################")
        else:
            vl_data_list = None
        self.vl_data_list = vl_data_list
        self.vl_image_dir = vl_image_dir



        self.transformations = None
        if self.imsize == 320:
            print(
                f"########################Current {colors['red']}{'Image Size is[240, 320]'}{colors['reset']}; maybe due to the pretrain data image size###################################")
            self.imsize = (320, 240)
        else:
            self.imsize = (320, 240)
        print(f"policy class: {colors['red']}{self.policy_class}{colors['reset']}; augument: {self.augment_images}")
        _ = self.__getitem__(0)  # initialize self.is_sim and self.transformations
        if len(self.camera_names) == 3:
            print("%" * 40)
            print(f"{colors['red']}{'There are three views: left, right, top'}{colors['reset']}")
        self.is_sim = False

    def __len__(self):
        return sum(self.episode_len)

    @property
    def modality_lengths(self):
        # TODO
        # explain: 3 view (two external 240 * 320, one wrist, 56 * 56), equals to 200 image token, we suppose max 100 tokens for text
        length_list = [300] * self.cumulative_len[-2]
        # compute vl sample length
        for sample in self.vl_data_list:
            cur_len = sum(len(conv['value'].split()) for conv in sample["conversations"])
            cur_len += 100  # plus the number of image tokens (suppose to 100)
            length_list.append(-cur_len)  # vl data length set to -cur_len
        return length_list

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        vl_data_only = not self.dataset_path_list or (self.vl_data_list is not None and self.episode_ids.shape[0] != 1 and episode_id == self.episode_ids[-1])
        text_data_only = False

        if vl_data_only:  # vl data only
            try:
                source = self.vl_data_list[start_ts]
            except IndexError:
                print(episode_id,start_ts)
                exit(-1)
            raw_lang = ""
            try:
                img_list = source["image"]
            except:
                img_list = None
                text_data_only = True
            reasoning = ""
            image_data = []
            if img_list is None or img_list== "":
                text_data_only = True
            else:
                if not isinstance(img_list, list):
                    img_list = [img_list]
                for img_name in img_list:
                    if self.is_local_debug:
                        img = np.zeros((320,240,3))
                    else:
                        img = np.array(Image.open(os.path.join(self.vl_image_dir, img_name)).convert("RGB"))
                        img = cv2.resize(img, (320, 240))
                    image_data.append(img)
                image_data = np.array(image_data)
                image_data = torch.from_numpy(image_data)
                assert len(image_data.shape) == 4, f"image_data shape is {image_data.shape}, the length of {source['image']} does not equal to 4"
                image_data = torch.einsum('k h w c -> k c h w', image_data)
            if self.robot == 'aloha':
                qpos_data = torch.ones((14,)).float()
                action_data = torch.ones((self.chunk_size, 14)).float()
            else:
                qpos_data = torch.ones((7, )).float()
                action_data = torch.ones((self.chunk_size, 10)).float()
            is_pad = torch.ones((self.chunk_size, )).bool()

        else:
            dataset_path = self.dataset_path_list[episode_id]
            with h5py.File(dataset_path, 'r') as root:
                try:  # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                raw_lang = root['language_raw'][0].decode('utf-8')
                reasoning = ""
                if self.data_args.use_reasoning:
                    if 'substep_reasonings' in root.keys():
                        reasoning = root['substep_reasonings'][start_ts].decode('utf-8')
                    else:
                        try:
                            reasoning = root['reasoning'][0].decode('utf-8')
                        except Exception as e:
                            reasoning = ""
                            print(e)
                            print(dataset_path)
                # Construct source from raw_lang and reasoning
                source = {
                    "conversations": [
                        {"from": "human", "value": raw_lang}
                    ]
                }
                if self.data_args.use_reasoning:
                    source["conversations"].append({"from": "gpt", "value": reasoning + "Next Action:"})
                action = root['/action'][()]
                original_action_shape = action.shape
                episode_len = original_action_shape[0]

                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                    if compressed:
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)
                    if self.imsize[0] != image_dict[cam_name].shape[1]:
                        image_dict[cam_name] = cv2.resize(image_dict[cam_name], self.imsize)

                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            if self.data_args.delta_control:
                padded_action[:action_len - 1] = action[1:] - action[:-1]
            else:
                padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            if self.robot == 'franka':
                assert image_data.ndim == 4, f"image_data's shape is {image_data.shape}, maybe the reason of adding historical images"
                image_data = torch.stack(
                    [torch.from_numpy(cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)) for img in image_data], dim=0)
            # channel last
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            if 'diffusion' in self.policy_class or "dit" in self.policy_class:
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (
                            self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
                ]
            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)


        sample = {
            'image': image_data,
            'state': qpos_data,
            'action': action_data,
            'is_pad': is_pad,
            "conversation": source,
        }
        if index == 0:
            print(raw_lang)
        del image_data
        del qpos_data
        del action_data
        del is_pad
        del raw_lang
        del reasoning
        gc.collect()
        torch.cuda.empty_cache()

        return self.llava_pythia_process.forward_process(sample, use_reasoning=self.data_args.use_reasoning,
                                                         vl_data_only= vl_data_only,text_data_only=text_data_only)


class Qwen2VLAProcess:
    def __init__(
            self,
            tokenizer=None,
            max_seq_len=512,
            multimodal_processor=None,
            camera_names=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.camera_names = camera_names
        self.multimodal_processor = multimodal_processor  # Qwen2VLProcessor
        self.end_r = "\n<|im_start|>assistant\n"
        self.start_r = "\n<|im_start|>user\n"

    def preprocess(
            self,
            text: str,
            image_inputs,
            video_inputs=None,
    ):
        data = {}
        rounds = text.split(self.end_r)
        input_ids = []
        labels = []
        for i, rnd in enumerate(rounds):
            if i != len(rounds) - 1:
                rnd += self.end_r
            # else:
            #     a = self.multimodal_processor.tokenizer.encode(rnd)
            #     if isinstance(a, list):
            #         a = torch.Tensor(a)
            #     input_ids.extend(a)
            #     labels.extend(a)
            #     continue
            if i == 0:  # may only have one iter (for w/o reasoning robot data)
                inputs = self.multimodal_processor(text=[rnd], images=image_inputs, videos=video_inputs, padding=True,
                                   return_tensors="pt")
                try:
                    data["pixel_values"] = inputs["pixel_values"]
                    data["image_grid_thw"] = inputs["image_grid_thw"]
                ### for text-only data
                except:
                    data["pixel_values"] = None
                    data["image_grid_thw"] = None
                input_ids.extend(inputs["input_ids"].squeeze())
                labels.extend(torch.ones_like(inputs['input_ids'].squeeze()) * -100)
            else:
                input_ids.extend(self.multimodal_processor.tokenizer.encode(rnd))
                tmp = rnd.split(self.start_r)
                a = tmp[0]
                a = self.multimodal_processor.tokenizer.encode(a)
                if isinstance(a, list):
                    a = torch.Tensor(a)
                labels.extend(a)
                if len(tmp) > 1:
                    q = self.start_r + tmp[1]
                    q = self.multimodal_processor.tokenizer.encode(q)
                    if isinstance(q, list):
                        q = torch.Tensor(q)
                    labels.extend(torch.ones_like(q) * -100)
        data["input_ids"] = torch.Tensor(input_ids).unsqueeze(0).to(torch.int64)
        data["labels"] = torch.Tensor(labels).unsqueeze(0).to(torch.int64)
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        return data

    def forward_process(self, sample, use_reasoning=True, vl_data_only=False, text_data_only=False):
        messages = self.datastruct_droid2llava(sample, text_data_only)
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )[:-23]
        if text_data_only:
            image_data = None
        else:
            image_data = torch.chunk(sample['image'], sample['image'].shape[0], 0)
            images_list = []
            for i, each in enumerate(image_data):
                ele = {'image': each}
                each = Image.fromarray(each.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
                ele['image'] = each
                ####
                if i<len(self.camera_names) and 'wrist' in self.camera_names[i] and len(self.camera_names) > 1:
                    ele['resized_height'] = 56
                    ele['resized_width'] = 56
                else:
                    ele['resized_height'] = 240
                    ele['resized_width'] = 320
                each = fetch_image(ele)

                images_list.append(torch.from_numpy(np.array(each)))
            image_data = images_list
        video_inputs = None
        outputs = self.preprocess(text, image_data, video_inputs)

        data_dict = dict(
            messages=messages,
            images=None
        )
        for k, v in outputs.items():
            data_dict[k] = v

        data_dict['state'] = sample['state']
        data_dict['action'] = sample['action']
        data_dict['is_pad'] = sample['is_pad']
        try:
            data_dict['pixel_token_len'] = len(outputs['pixel_values'])
        except:
            data_dict['pixel_token_len'] = 0

        # vl data mask, 1 means this item dose not contain robotics state and action
        data_dict['vl_data_mask'] = torch.Tensor([1]) if vl_data_only else torch.Tensor([0])
        data_dict['text_only_mask'] = torch.Tensor([1]) if text_data_only else torch.Tensor([0])

        return data_dict

    def datastruct_droid2llava(self, sample, text_data_only=False):
        messages = []
        if text_data_only:
            for i in range(len(sample["conversation"]["conversations"])):
                rnd = sample["conversation"]["conversations"][i]
                if i % 2 == 0:
                    messages.append({
                        "role": "user",
                        "content": rnd['value']
                    })
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": rnd['value'],
                        }
                    )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [],
                }
            ]
            for i in range(sample['image'].shape[0]):
                messages[0]["content"].append({"type": "image"})
            for i in range(len(sample["conversation"]["conversations"])):
                rnd = sample["conversation"]["conversations"][i]
                if i == 0:
                    q = rnd['value'].replace('<image>', '').replace('\n', '')
                    messages[0]["content"].append({"type": "text", "text": f"{q}"})
                elif i % 2 == 0:
                    messages.append({
                        "role": "user",
                        "content": rnd['value']
                    })
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": rnd['value'],
                        }
                    )
        return messages


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


def load_data(dataset_dir_l, name_filter, camera_names,  chunk_size, config,
              skip_mirrored_data=False, stats_dir_l=None, policy_head_type=None,
              llava_pythia_process=None, vl_file=None, vl_image_dir=None, template_path=None, vl_ratio=0, is_local_debug=False):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    for d, dpl in zip(dataset_dir_l, dataset_path_list_list):
        if len(dpl) == 0:
            print("#2" * 20)
            print(f"{d} does not contain any hdf5 files")


    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)
    num_episodes_cumsum = np.insert(num_episodes_cumsum, 0, 0)
    if len(dataset_path_list) > 0:
        # num_episodes_0 = len(dataset_path_list_list[0])
        # train_episode_ids_0 = np.random.permutation(num_episodes_0)
        # train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for
        #                                                idx, num_episodes in enumerate(num_episodes_l[1:])]
        train_episode_ids_l = [np.arange(num_episodes) + num_episodes_cumsum[idx] for
                                                       idx, num_episodes in enumerate(num_episodes_l[0:])]
        train_episode_ids = np.concatenate(train_episode_ids_l)
        print(
            f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n\n')

        _, all_episode_len = get_norm_stats(dataset_path_list)
        print("All images:", sum(all_episode_len))
        train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]

        train_episode_len = flatten_list(train_episode_len_l)
        if stats_dir_l is None:
            stats_dir_l = dataset_dir_l
        elif type(stats_dir_l) == str:
            stats_dir_l = [stats_dir_l]
        norm_stats, _ = get_norm_stats(
            flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
        print(f'Norm stats from: {stats_dir_l}')
        json_file = os.path.join(stats_dir_l[0], 'norm_stats.json')

        print(f'train_episode_len_l: {train_episode_len_l}')
    else:
        norm_stats = None
        train_episode_len = []
        train_episode_ids = []

    robot = 'aloha' if config['action_head_args'].action_dim == 14 or (
            'aloha' in config['training_args'].output_dir) else 'franka'
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len,
                                    chunk_size, policy_head_type, llava_pythia_process=llava_pythia_process, vl_file=vl_file,
                                    vl_image_dir=vl_image_dir,
                                    imsize=config['training_args'].pretrain_image_size, data_args=config['data_args'],
                                    vl_ratio=vl_ratio, is_local_debug=is_local_debug,
                                    robot=robot
                                    )

    return train_dataset, None, norm_stats


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0  # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action


def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5) / 5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)


def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action


def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
