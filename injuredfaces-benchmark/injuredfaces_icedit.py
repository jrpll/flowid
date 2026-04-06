import torch
from diffusers.utils import load_image
import os
import json
from PIL import Image
from tqdm import tqdm
from diffusers import FluxFillPipeline
import numpy as np

flux_path = "models--black-forest-labs--FLUX.1-Fill-dev/snapshots/358293da0354175698b67ec8299acf928313a78a"
lora_path = "lora_icedit.safetensors"
seed = 42

pipe = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_path)
pipe = pipe.to("cuda")

data_path = "InjuredFaces-ICEdit"
identities = [id for id in os.listdir(data_path) if ".DS" not in id]

for id in tqdm(identities):
    id_path = os.path.join(data_path,id)
    images = [img for img in os.listdir(id_path) if "injured" in img]
    for img_name in images:
        img_path = os.path.join(id_path,img_name)
        img = Image.open(img_path)
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

        prompt_path = img_path.replace("injured","prompt").replace(".jpg",".txt")
        with open(prompt_path,"r") as p:
            prompt = p.read()

        sub_prompts = []
        if "blood" in prompt:
            sub_prompts.append("now there is no blood")

        if "mouth open" in prompt:
            sub_prompts.append("now the mouth is closed")

        if "eye closed" in prompt or "eyes closed" in prompt:
            sub_prompts.append("now the eyes are opened")

        if "bump" in prompt:
            sub_prompts.append("now there is no bump")

        if "bruise" in prompt:
            sub_prompts.append("now there are no bruises")

        if "cut" in prompt:
            sub_prompts.append("now there is no cut")

        if "wound" in prompt:
            sub_prompts.append("now there is no wound")
        
        if "injury" in prompt:
            sub_prompts.append("now there is no injury")

        if "swollen" in prompt:
            sub_prompts.append("now there is no swelling")

        instruction = ", ".join(sub_prompts)
        instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'
        edited = pipe(
            prompt=instruction,
            image=combined_image,
            mask_image=mask,
            height=height,
            width=width * 2,
            guidance_scale=200,
            num_inference_steps=28,
        ).images[0]
        edited = edited.crop((width,0,width*2,height))
        output_img_path = os.path.join(id_path,f"edited_{img_name}")
        edited.save(output_img_path)