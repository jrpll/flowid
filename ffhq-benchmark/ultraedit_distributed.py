import os
import json
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import argparse
import torch
from ultraedit_diffusers import StableDiffusion3InstructPix2PixPipeline
from ultraedit_diffusers.utils import load_image
import requests
from PIL import Image
import PIL.ImageOps

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
        data_path="FFHQ-UltraEdit",
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
            src_prompt = prompts["src"]
            edit_prompts = [
                prompts["edit_prompt_1"],
                prompts["edit_prompt_2"],
                prompts["edit_prompt_3"],
                prompts["edit_prompt_4"],
                prompts["edit_prompt_5"]
            ]
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
            prompts = [src_prompt] + edit_prompts
            for i in range(5):
                if "bald" in prompts[i] and kwds[i]=="hair":
                    instruction = "Add hair to this man"
                elif "bald" in prompts[i+1] and kwds[i]=="hair":
                    instruction = "Make this man fully bald"
                elif "hat" in prompts[i] and kwds[i]=="hat":
                    instruction = "Remove the hat"
                elif "hat" in prompts[i+1] and kwds[i]=="hat":
                    instruction = "Add a hat to this person"
                elif "earrings" in prompts[i] and kwds[i]=="earrings":
                    instruction = "Remove the earrings"
                elif "earrings" in prompts[i+1] and kwds[i]=="earrings":
                    instruction = "Add earrings to this person"
                elif "beard" in prompts[i] and kwds[i]=="beard":
                    instruction = "Remove the beard"
                elif "beard" in prompts[i+1] and kwds[i]=="beard":
                    instruction = "Add a beard to this person"
                elif "smiling" in prompts[i] and kwds[i]=="smiling":
                    instruction = "Remove the smile"
                elif "smiling" in prompts[i+1] and kwds[i]=="smiling":
                    instruction = "Add a smile to this person"
                elif "glasses" in prompts[i] and kwds[i]=="glasses":
                    instruction = "Remove the glasses"
                elif "glasses" in prompts[i+1] and kwds[i]=="glasses":
                    instruction = "Add glasses"
                elif "dark hair" in prompts[i] and kwds[i]=="hair":
                    instruction = "Change the hair color to light"
                else:
                    instruction = "Change the hair color to dark"
                with torch.autocast("cuda",torch.float16):
                    prompt=instruction
                    img = img.resize((512, 512))
                    mask_img = PIL.Image.new("RGB", img.size, (255, 255, 255))
                    img = self.pipe(
                        prompt,
                        image=img,
                        mask_img=mask_img,
                        negative_prompt=neg_prompts[i],
                        num_inference_steps=28,
                        image_guidance_scale=1.5,
                        guidance_scale=7.5,
                    ).images[0]
                img.save(os.path.join(id_folder,f"{i}_{filename}"))

def main(args):
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)
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
    parser.add_argument("--data_path",type=str, default="FFHQ-UltraEdit")
    args = parser.parse_args()
    main(args)
    

