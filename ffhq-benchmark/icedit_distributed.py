import sys
import os

from diffusers import FluxFillPipeline
import torch
from PIL import Image
import numpy as np
import argparse
import random

from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import json

def get_instruction(neg_prompt,kwd):
    if ("dark hair" in neg_prompt or "light hair" in neg_prompt) and "hair" in kwd:
        if "dark hair" in neg_prompt:
            instruction = "the person now has light hair"
        elif "light hair" or "grey hair" in neg_prompt:
            instruction = "the person now has dark hair"
    elif "hat" in kwd:
        if "hat" in neg_prompt:
            instruction = "the person does not wear a hat anymore"
        else:
            instruction = "the person now wears a hat"
    elif "glasses" in kwd:
        if "glasses" in neg_prompt:
            instruction = "the person does not wear glasses anymore"
        else:
            instruction = "the person now wears glasses"
    elif "hair" == kwd:
        if "hair" in neg_prompt:
            instruction = "the person is now bald"
        else:
            instruction = "the person now has hair"
    elif "beard" in kwd:
        if "beard" in neg_prompt:
            instruction = "the person does not have a beard anymore"
        else:
            instruction = "the person now has a beard"
    elif "earrings" in kwd:
        if "earrings" in neg_prompt:
            instruction = "the person is not wearing earrings anymore"
        else:
            instruction = "the person is now wearing earrings"
    elif "smiling" in kwd:
        if "smiling" in neg_prompt:
            instruction = "the person is not smiling anymore"
        else:
            instruction = "the person is now smiling"
    return instruction

class FFHQ(Dataset):
    def __init__(
            self,
            data_path
    ):
        self.ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and len(os.listdir(os.path.join(data_path, d))) < 7]
        self.size = len(self.ids)
        print(f"PROCESSING {self.size} IDS")
                
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.ids[index]

def custom_collate(batch):
    return batch

class DistributedEditor:
    def __init__(
        self,
        pipe,
        data_path="FFHQ-ICEdit",
    ):
        self.pipe = pipe
        self.data_path = data_path

    def edit(self,data):
        for batch in data:
            filename = batch[0]+".png"
            id_folder = os.path.join(self.data_path,batch[0])
            img_path = os.path.join(self.data_path,batch[0],filename)
            img = Image.open(img_path).resize((1024,1024)).convert("RGB")
            json_file = filename.replace(".png",".json")
            json_path = os.path.join(self.data_path,batch[0],json_file)
            with open(json_path,"r") as f:
                prompts = json.load(f) 
            neg_prompts = [
                prompts["neg_p1"],
                prompts["neg_p2"],
                prompts["neg_p3"],
                prompts["neg_p4"],
                prompts["neg_p5"],
            ]
            kwds = [
                prompts["kw_1"],
                prompts["kw_2"],
                prompts["kw_3"],
                prompts["kw_4"],
                prompts["kw_5"]
            ]
            for i in range(5):
                neg_prompt = neg_prompts[i]
                kwd = kwds[i]
                instruction = get_instruction(neg_prompt,kwd)   
                prompt = f'A diptych with two side-by-side photos of the same person. On the right, the face is exactly the same as on the left but {instruction}'
                if img.size[0] != 512:
                    print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
                    new_width = 512
                    scale = new_width / img.size[0]
                    new_height = int(img.size[1] * scale)
                    new_height = (new_height // 8) * 8  
                    img = img.resize((new_width, new_height))
                    print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

                width, height = img.size
                combined_image = Image.new("RGB", (width * 2, height))
                combined_image.paste(img, (0, 0))
                combined_image.paste(img, (width, 0))
                mask_array = np.zeros((height, width * 2), dtype=np.uint8)
                mask_array[:, width:] = 255 
                mask = Image.fromarray(mask_array)

                imgs = self.pipe(
                    prompt=instruction,
                    image=combined_image,
                    mask_image=mask,
                    height=height,
                    width=width * 2,
                    guidance_scale=200,
                    num_inference_steps=28,
                ).images[0]

                img = imgs.crop((width,0,width*2,height))
                img.save(os.path.join(id_folder,f"{i}_{filename}"))

def main(args):
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    flux_path = "models--black-forest-labs--FLUX.1-Fill-dev/snapshots/358293da0354175698b67ec8299acf928313a78a"
    lora_path = "lora_icedit.safetensors"
    pipe = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(lora_path)
    pipe = pipe.to("cuda")

    distributed_editor = DistributedEditor(
        pipe=pipe,
        data_path = args.data_path,
    )
    dataset = FFHQ(data_path = args.data_path)
    data = DataLoader(dataset,batch_size=1,sampler=DistributedSampler(dataset),collate_fn=custom_collate)
    distributed_editor.edit(data)
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="FFHQ-ICEdit")
    args = parser.parse_args()
    main(args)
