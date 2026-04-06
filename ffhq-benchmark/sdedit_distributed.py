from diffusers import FluxImg2ImgPipeline
import torch
import os
import json
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import argparse

def get_strength(neg_prompt,kwd):
    if ("dark hair" in neg_prompt or "light hair" in neg_prompt) and "hair" in kwd:
        strength = 0.93
    elif "hat" in kwd:
        strength = 0.9    
    elif "glasses" in kwd:
        if "glasses" in neg_prompt:
            strength = 0.95
        else:
            strength = 0.7
    elif "hair"==kwd:
        strength = 0.9  
    elif "beard" in kwd:
        if "beard" in neg_prompt:
           strength = 0.95
        else:
           strength = 0.6
    elif "earrings" in kwd:
        if "earrings" in neg_prompt:
            strength = 0.9
        else:
            strength = 0.7
    elif "smiling" in kwd:
        if "smiling" in neg_prompt:
           strength = 0.95
        else:
           strength = 0.9
    return strength

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
        data_path
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
            for i in range(5):
                tgt_prompt = edit_prompts[i]
                neg_prompt = neg_prompts[i]
                kwd = kwds[i]
                img = self.pipe(
                    image=img,
                    prompt=tgt_prompt,
                    negative_prompt=neg_prompt,
                    strength=get_strength(neg_prompt,kwd)
                ).images[0]
                img.save(os.path.join(id_folder,f"{i}_{filename}"))

def main(args):
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    pipe = FluxImg2ImgPipeline.from_pretrained("models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44", torch_dtype=torch.bfloat16).to("cuda")
    distributed_editor = DistributedEditor(
        pipe=pipe,
        data_path = args.data_path
    )
    dataset = FFHQ(data_path = args.data_path)
    data = DataLoader(dataset,batch_size=1,sampler=DistributedSampler(dataset),collate_fn=custom_collate)
    distributed_editor.edit(data)
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="FFHQ-SDEdit")
    args = parser.parse_args()
    main(args)