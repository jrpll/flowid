from ultraedit_diffusers import StableDiffusion3InstructPix2PixPipeline
from ultraedit_diffusers.utils import load_image
import os
from PIL import Image
import torch

pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

data_path = "InjuredFaces-UltraEdit"
identities = [id for id in os.listdir(data_path) if ".DS" not in id]
for id in identities:
    id_path = os.path.join(data_path,id)
    injured_files = [f for f in os.listdir(id_path) if "injured" in f]
    for f in injured_files:
        image_path = os.path.join(id_path,f)
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        mask_img = Image.new("RGB", image.size, (255, 255, 255))
        edited = pipe(
            "Remove blood, swelling and wounds off this person's face",
            image=image,
            mask_img=mask_img,
            num_inference_steps=28,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
        ).images[0]  
        save_path = os.path.join(id_path,f"edited_{f}")
        edited.save(save_path)      
