import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
import os
import json
from PIL import Image
from tqdm import tqdm

pipe = FluxImg2ImgPipeline.from_pretrained("/tmpdir/m23042rpll/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44", torch_dtype=torch.bfloat16)
pipe.to("cuda")

data_path = "InjuredFaces-SDEdit"
identities = [id for id in os.listdir(data_path) if ".DS" not in id]

for id in tqdm(identities):
    id_path = os.path.join(data_path,id)
    images = [img for img in os.listdir(id_path) if "injured" in img]
    for img_name in images:
        img_path = os.path.join(id_path,img_name)
        image = Image.open(img_path)

        prompt_path = img_path.replace("injured","prompt").replace(".jpg",".txt")
        with open(prompt_path,"r") as p:
            prompt = p.read()

        edited = pipe(
            image=image,
            prompt="a person's face",
            negative_prompt=prompt.replace("a person's face, ",""),
            strength=0.65,
            num_inference_steps=28
        ).images[0]
        output_img_path = os.path.join(id_path,f"edited_{img_name}")
        edited.save(output_img_path)