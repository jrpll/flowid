import torch
from PIL import Image
import os
import json
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import argparse
import os
import re
import time
from io import BytesIO
import uuid
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image

import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from transformers import pipeline
import sys
sys.path.insert(0, 'RF-Solver-Edit')
from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, load_ae, load_clip, load_flow_model, load_t5)
from huggingface_hub import login

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

def get_inject_step(neg_prompt,kwd):
    if ("dark hair" in neg_prompt or "light hair" in neg_prompt) and "hair" in kwd:
        inject_step = 0
    elif "hat" in kwd:
        inject_step = 0
    elif "glasses" in kwd:
        inject_step = 2
    elif "hair"==kwd:
        inject_step = 0
    elif "beard" in kwd:
        inject_step = 2
    elif "earrings" in kwd:
        inject_step = 2
    elif "smiling" in kwd:
        inject_step = 0
    return inject_step

class FFHQ(Dataset):
    def __init__(
            self,
            data_path
    ):
        self.ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and len(os.listdir(os.path.join(data_path, d))) < 7]
        self.size = len(self.ids)
                
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.ids[index]

def custom_collate(batch):
    return batch

class DistributedEditor:
    def __init__(
        self,
        ae,
        t5,
        clip,
        name,
        output_dir,
        add_sampling_metadata,
        model,
        device,
        data_path="FFHQ-RFSolver",
    ):
        self.ae = ae
        self.t5 = t5
        self.clip = clip
        self.name = name
        self.output_dir = output_dir
        self.add_sampling_metadata = add_sampling_metadata
        self.model = model
        self.data_path = data_path
        self.device = device

    @torch.inference_mode()
    def edit_rf_solver(self, init_image, source_prompt, target_prompt, num_steps, inject_step, guidance):
    
        device = self.device
        torch.cuda.empty_cache()
        seed = None
            
        shape = init_image.shape
    
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
    
        init_image = init_image[:new_h, :new_w, :]
    
        width, height = init_image.shape[0], init_image.shape[1]
    
    
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(device)
        with torch.no_grad():
            init_image = self.ae.encode(init_image.to(device)).to(torch.bfloat16)
    
        print(init_image.shape)
    
        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance=guidance,
                seed=seed,
            )
        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
            
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()
    
        opts.seed = None
    
        #############inverse#######################
        info = {}
        info['feature'] = {}
        info['inject_step'] = inject_step
        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
    
        # inversion initial noise
        with torch.no_grad():
            z, info = denoise(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            
        inp_target["img"] = z
    
        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))
    
        # denoise initial noise
        x, _ = denoise(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)
    
        # decode latents to pixel space
        x = unpack(x.float(), opts.width, opts.height)
    
        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0
                
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            x = self.ae.decode(x)
    
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    
        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
    
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        if self.add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = source_prompt
        # img.save(fn, exif=exif_data, quality=95, subsampling=0)
    
        print("End Edit")
        return img

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
                img = np.array(img)
                src_prompt = prompts[i]
                tgt_prompt = prompts[i+1]
                neg_prompt = neg_prompts[i]
                kwd = kwds[i]
                inject_step = get_inject_step(neg_prompt,kwd)
                img = self.edit_rf_solver(
                    init_image = img,
                    source_prompt = src_prompt,
                    target_prompt = tgt_prompt,
                    num_steps = 25,
                    inject_step = inject_step,
                    guidance = 2
                )
                img.save(os.path.join(id_folder,f"{i}_{filename}"))

def main(args):
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    name = 'flux-dev'
    ae = load_ae(name, device)
    t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device=device)
    offload = False
    name = "flux-dev"
    is_schnell = False
    feature_path = 'feature'
    output_dir = 'result'
    add_sampling_metadata = False

    distributed_editor = DistributedEditor(
        ae = ae,
        t5 = t5,
        clip = clip,
        name = name,
        output_dir = output_dir,
        add_sampling_metadata = add_sampling_metadata,
        model = model,
        device = device,
        data_path = args.data_path,
    )
    dataset = FFHQ(data_path = args.data_path)
    data = DataLoader(dataset,batch_size=1,sampler=DistributedSampler(dataset),collate_fn=custom_collate)
    distributed_editor.edit(data)
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default="FFHQ-RFSolver")
    args = parser.parse_args()
    main(args)
    

