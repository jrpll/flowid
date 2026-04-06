import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os
import json
from PIL import Image
from tqdm import tqdm

pipe = FluxKontextPipeline.from_pretrained("models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/af58063aa431f4d2bbc11ae46f57451d4416a170", torch_dtype=torch.bfloat16).to("cuda")

data_path = "InjuredFaces-Kontext"
identities = [id for id in os.listdir(data_path) if ".DS" not in id]

for id in tqdm(identities):
    id_path = os.path.join(data_path,id)
    images = [img for img in os.listdir(id_path) if "injured" in img]
    for img_name in images:
        img_path = os.path.join(id_path,img_name)
        prompt_path = img_path.replace("injured","prompt").replace(".jpg",".txt")
        image = Image.open(img_path)

        with open(prompt_path,"r") as p:
            prompt = p.read()

        
        sub_prompts = []
        if "blood" in prompt:
            sub_prompts.append("remove the blood")

        if "mouth open" in prompt:
            sub_prompts.append("close the mouth")

        if "eye closed" in prompt or "eyes closed" in prompt:
            sub_prompts.append("open the eyes")

        if "bump" in prompt:
            sub_prompts.append("remove the bump")

        if "bruise" in prompt:
            sub_prompts.append("remove the bruises")

        if "cut" in prompt:
            sub_prompts.append("remove the cut")

        if "wound" in prompt:
            sub_prompts.append("remove the wound")
        
        if "injury" in prompt:
            sub_prompts.append("remove the injury")

        if "swollen" in prompt:
            sub_prompts.append("remove the swelling")

        instruction = ", ".join(sub_prompts)

        edited = pipe(
            image=image,
            prompt = instruction,
            guidance_scale=2.5
        ).images[0]
        output_img_path = os.path.join(id_path,f"edited_{img_name}")
        edited.save(output_img_path)