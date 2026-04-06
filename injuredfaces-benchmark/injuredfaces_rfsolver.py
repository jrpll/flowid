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
import numpy as np
from transformers import pipeline

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, load_ae, load_clip, load_flow_model, load_t5)
from huggingface_hub import login
from tqdm import tqdm

import torch.distributed as dist

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()


def split_data_for_rank(data_list, rank, world_size):
    """Split data evenly across processes."""
    # Each process gets a subset of the data
    return data_list[rank::world_size]


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


@torch.inference_mode()
def edit(init_image, source_prompt, target_prompt, num_steps, inject_step, guidance,
         device, ae, t5, clip, model, name, output_dir, add_sampling_metadata):

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
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)

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
        inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
        inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

    # inversion initial noise
    with torch.no_grad():
        z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        
    inp_target["img"] = z

    timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # denoise initial noise
    x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

    # decode latents to pixel space
    x = unpack(x.float(), opts.width, opts.height)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0
            
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

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
    exif_data[ExifTags.Base.Model] = name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = source_prompt

    print("End Edit")
    return img


def main():
    # Setup distributed environment
    rank, world_size, local_rank, device = setup_distributed()
    
    print(f"[Rank {rank}/{world_size}] Starting on device {device}")
    
    # Configuration
    name = 'flux-dev'
    output_dir = 'result'
    add_sampling_metadata = True
    
    # Load models on the correct device
    print(f"[Rank {rank}] Loading models...")
    ae = load_ae(name, device)
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(name, device=device)
    print(f"[Rank {rank}] Models loaded.")
    
    # Prepare the full list of (image_path, prompt_path) pairs
    data_path = "InjuredFaces-RFSolver-Inject10"
    
    all_tasks = []
    identities = [id for id in os.listdir(data_path) if ".DS" not in id]
    for id in identities:
        id_path = os.path.join(data_path, id)
        images = [img for img in os.listdir(id_path) if "injured" in img]
        for img_name in images:
            img_path = os.path.join(id_path, img_name)
            prompt_path = img_path.replace("injured", "prompt").replace(".jpg", ".txt")
            output_img_path = os.path.join(id_path, f"edited_{img_name}")
            all_tasks.append((img_path, prompt_path, output_img_path))
    
    # Sort to ensure all ranks have the same order
    all_tasks.sort()
    
    # Split tasks across ranks
    my_tasks = split_data_for_rank(all_tasks, rank, world_size)
    
    print(f"[Rank {rank}] Processing {len(my_tasks)}/{len(all_tasks)} tasks")
    
    # Process this rank's subset
    for img_path, prompt_path, output_img_path in tqdm(my_tasks, desc=f"Rank {rank}", disable=(rank != 0)):
        # Skip if already processed
        if os.path.exists(output_img_path):
            print(f"[Rank {rank}] Skipping {output_img_path} (already exists)")
            continue
            
        try:
            image = np.array(Image.open(img_path))
            with open(prompt_path, "r") as p:
                src_prompt = p.read()

            edited = edit(
                init_image=image,
                source_prompt=src_prompt,
                target_prompt="a person's face",
                num_steps=25,
                inject_step=10,
                guidance=2,
                device=device,
                ae=ae,
                t5=t5,
                clip=clip,
                model=model,
                name=name,
                output_dir=output_dir,
                add_sampling_metadata=add_sampling_metadata
            )
            edited.save(output_img_path)
            print(f"[Rank {rank}] Saved {output_img_path}")
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing {img_path}: {e}")
            continue
    
    # Synchronize all processes before cleanup
    dist.barrier()
    print(f"[Rank {rank}] Done!")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()