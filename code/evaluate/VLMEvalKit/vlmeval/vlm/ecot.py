import torch
import re
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp


from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy


class ECOT(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='Embodied-CoT/ecot-openvla-7b-bridge',
                 **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except Exception as e:
            logging.critical("Please install Transformers version 4.45.1 by running: pip install transformers==4.45.1")
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        # self.model = (
        #     Kosmos2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        #     .to(torch.device('cuda'))
        # )
        # self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(torch.device('cuda'))

        default_kwargs = dict(
            # max_new_tokens=512,
            # use_cache=True
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        # TASK_TOKEN = '<grounding> '
        # QEUSTION_TOKEN = 'Question: '
        # ANSWER_TOKEN = 'Answer: '
        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        # def get_openvla_prompt(instruction: str) -> str:
        #     return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
        #
        # INSTRUCTION = "place the watermelon on the towel"
        # prompt = get_openvla_prompt(INSTRUCTION)



        images = []
        prompt = ''

        prompt += SYSTEM_PROMPT
        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                # prompt += QEUSTION_TOKEN
                prompt += s['value']
                # prompt += ANSWER_TOKEN

        images = [Image.open(s) for s in images]
        inputs = self.processor(text=prompt, images=images[0], return_tensors='pt').to(torch.device('cuda'), dtype=torch.bfloat16)

        # generated_ids = self.model.generate(
        #     pixel_values=inputs['pixel_values'],
        #     input_ids=inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        #     image_embeds=None,
        #     image_embeds_position_mask=inputs['image_embeds_position_mask'],
        #     **self.kwargs
        # )
        action, generated_ids = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False,
                                                   max_new_tokens=1024)
        generated_text = self.processor.batch_decode(generated_ids)[0]

        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # processed_text = self.processor.post_process_generation(generated_text, cleanup_and_extract=True)[0]
        # cleaned_answer = re.sub(r'(Question:.*?Answer:|Question:.*)', '', processed_text).strip()
        return generated_text

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False


    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message
