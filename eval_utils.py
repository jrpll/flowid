"""
from v2 -> v3: add CLIP Score and CLIP I.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
import torch
import clip
import json
import os
import io

from tqdm import tqdm
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from pyiqa.utils.img_util import is_image_file

from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Clip-I: call the clip model and self implement it

########################### Basic Func ################################

def imread(img_source, rgb=False, target_size=None):
    """Read image
    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
        target_size: resize image to target size if not None
    """
    if type(img_source) == bytes:
        img = Image.open(io.BytesIO(img_source))
    elif type(img_source) == str:
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif type(img_source) == Image.Image:
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    return img

########################### Evaluation ################################

def eval_distance(image_pairs, metric='l1'):
    """
    Using pytorch to evaluate l1 or l2 distance
    """
    if metric == 'l1':
        criterion = nn.L1Loss()
    elif metric == 'l2':
        criterion = nn.MSELoss()
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        gen_img = img_pair[0].convert('RGB')
        gt_img = img_pair[1].convert('RGB')
        # resize to gt size
        gen_img = gen_img.resize(gt_img.size)
        # convert to tensor
        gen_img = transforms.ToTensor()(gen_img)
        gt_img = transforms.ToTensor()(gt_img)
        # calculate distance
        per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
        eval_score += per_score

    return eval_score / len(image_pairs)

def eval_clip_i(device, image_pairs, model, transform, metric='clip_i'):
    """
    Calculate CLIP-I score, the cosine similarity between the generated image and the ground truth image
    """
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if metric == 'clip_i':
                image_features = model.encode_image(image_input).detach().cpu().float()
            elif metric == 'dino':
                image_features = model(image_input).detach().cpu().float()
        return image_features
    # model, transform = clip.load("ViT-B/32", args.device)
    eval_score = 0
    for img_pair in tqdm(image_pairs):
        generated_features = encode(img_pair[0].convert('RGB'), model, transform)
        gt_features = encode(img_pair[1].convert('RGB'), model, transform)
        similarity = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                 gt_features.view(gt_features.shape[1]))
        if similarity > 1 or similarity < -1:
            raise ValueError(" strange similarity value")
        eval_score = eval_score + similarity
        
    return eval_score / len(image_pairs)

def eval_clip_score(device, image_pairs, clip_metric, caption_dict):
    """
    Calculate CLIP score, the cosine similarity between the image and caption
    return gen_clip_score, gt_clip_score
    """
    trans = transforms.Compose([
        transforms.Resize(256),  # scale to 256x256
        transforms.CenterCrop(224),  # crop to 224x224
        transforms.ToTensor(),  # convert to pytorch tensor
    ])

    def clip_score(image_path, caption):
        image = Image.open(image_path).convert('RGB')
        image_tensor = trans(image).to(device)
        return clip_metric(image_tensor, caption).detach().cpu().float()
    
    gen_clip_score = 0
    gt_clip_score = 0
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]
        gt_img_name = gt_img_path.split('/')[-1]
        gt_caption = caption_dict[gt_img_name]
        gen_clip_score += clip_score(gen_img_path, gt_caption)
        gt_clip_score += clip_score(gt_img_path, gt_caption)

    
    return gen_clip_score / len(image_pairs), gt_clip_score / len(image_pairs)

def eval_clip_t(device, image, caption, model, transform):
    """
    Calculate CLIP-T score, the cosine similarity between the image and the text CLIP embedding
    """
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features

    generated_features = encode(image, model, transform)
    # get text CLIP embedding
    text_features = clip.tokenize(caption).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_features).detach().cpu().float()

    gen_clip_t = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                text_features.view(text_features.shape[1]))
        
    return gen_clip_t

def eval_clip_dir(device, images, captions, model, transform):
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features

    img_feat_1 = encode(images[0], model, transform)
    img_feat_2 = encode(images[1], model, transform)

    text_feat_1 = clip.tokenize(captions[0]).to(device)
    text_feat_2 = clip.tokenize(captions[1]).to(device)

    img_feat_1 = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
    img_feat_2 = img_feat_2 / img_feat_2.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        text_feat_1 = model.encode_text(text_feat_1).detach().cpu().float()
        text_feat_2 = model.encode_text(text_feat_2).detach().cpu().float()

    text_feat_1 = text_feat_1 / text_feat_1.norm(dim=-1, keepdim=True)
    text_feat_2 = text_feat_2 / text_feat_2.norm(dim=-1, keepdim=True)

    change_img = img_feat_2 - img_feat_1
    change_txt = text_feat_2 - text_feat_1

    agreement = 1 - spatial.distance.cosine(change_img.view(change_img.shape[1]),change_txt.view(change_txt.shape[1]))

    return agreement

########################### Data Loading ################################

def get_final_turn_img(dir, img_type):
    """
    GET THE FINAL TURN IMAGE PATH. for generated image, get the last edit image in the iteratively edit setting. for gt image, get the last gt image.
    dir: the directory of the image
    img_type: 'generated' or 'gt'
    "naming rule in generated image: 
        '_1.png' for first edit
        'inde_' + str(i) + '.png' for i-th edit in independent editing setting
        'iter_' + str(i) + '.png' for  i-th edit in iteratively editing setting
    "naming rule in gt image:
        'output' + str(i) + '.png' for i-th edit
    return: the final turn image path
    """
    img_name_list = os.listdir(dir)
    # make sure end with .png or .jpg
    img_name_list = [s for s in img_name_list if '.png' in s or '.jpg' in s]
    # remove mask
    img_name_list = [s for s in img_name_list if 'mask' not in s]

    if img_type == 'generated':
        if len(img_name_list) == 1:
            if '_1' in img_name_list[0]:  # should be the first edit
                return os.path.join(dir, img_name_list[0])
            else:
                raise ValueError(f"Only one image but wrongly named, image name: {os.path.join(dir,img_name_list[0])}\n img_name_list: {img_name_list}")
        else:
            gen_list = [s for s in img_name_list if 'iter_' in s]
            gen_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

            if len(gen_list) == 0:
                raise ValueError(f"Empty gen list, image name: {dir}\n img_name_list: {img_name_list}")
            return os.path.join(dir, gen_list[-1])

    elif img_type == 'gt':
        gt_list = [s for s in img_name_list if 'output' in s]
        if len(gt_list) == 0:
            raise ValueError(f"Empty gt list, image name: {dir}\n img_name_list: {img_name_list}")
        gt_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        return os.path.join(dir, gt_list[-1])
    else:
        raise ValueError(f"Image type is not expected, only support 'generated' and 'gt', but got {img_type}")


def get_all_turn_imgs(dir, img_type):
    """
    GET ALL TURN IMAGE PATH. for generated image, get the image in the independently edit setting. for gt image, get all outputs.
    dir: the directory of the image
    img_type: 'generated' or 'gt'
    "naming rule in generated image: 
        '_1.png' for first edit
        'inde_' + str(i) + '.png' for i-th edit in independent editing setting
    "naming rule in gt image:
        'output' + str(i) + '.png' for i-th edit
    return: the final turn image path
    """
    img_name_list = os.listdir(dir)
    # make sure end with .png or .jpg
    img_name_list = [s for s in img_name_list if '.png' in s or '.jpg' in s]
    # remove mask
    img_name_list = [s for s in img_name_list if 'mask' not in s]

    if len(img_name_list) == 0:
        raise ValueError(f"Empty img list, image name: {dir}\n img_name_list: {img_name_list}")
    # all turn included the first edit
    if img_type == 'generated':
        if len(img_name_list) == 1:
            if '_1' in img_name_list[0]:
                return [os.path.join(dir, img_name_list[0])]
            else:
                raise ValueError(f"Only one image but wrongly named, image name: {os.path.join(dir,img_name_list[0])}\n img_name_list: {img_name_list}")
        else:
            gen_list = [os.path.join(dir, s) for s in img_name_list if '_1' in s]
            gen_list.extend([os.path.join(dir, s) for s in img_name_list if 'inde_' in s])
            gen_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

            if len(gen_list) == 0:
                raise ValueError(f"Empty gen list, image name: {dir}\n img_name_list: {img_name_list}")
            return gen_list
    elif img_type == 'gt':
        gt_list = [os.path.join(dir, s) for s in img_name_list if 'output' in s]
        if len(gt_list) == 0:
            raise ValueError(f"Empty gt list, image name: {dir}\n img_name_list: {img_name_list}")
        gt_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        return gt_list
    else:
        raise ValueError(f"Image type is not expected, only support 'generated' and 'gt', but got {img_type}")
    

def vlm_single_image(model,processor,image,question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ],
        }
    ]

    # Process the input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[:, -1, :]

    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    logit_yes = next_token_logits[0, yes_id].item()
    logit_no = next_token_logits[0, no_id].item()

    prob = torch.sigmoid(torch.tensor(logit_yes - logit_no)).item()
    #print(f"P(Yes) = {prob:.4f}")

    return prob

def vlm_two_images(model,processor,ref_image,edited_image,question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_image},
                {"type": "image", "image": edited_image},
                {"type": "text", "text": question}
            ],
        }
    ]

    # Process the input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[:, -1, :]

    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    logit_yes = next_token_logits[0, yes_id].item()
    logit_no = next_token_logits[0, no_id].item()

    prob = torch.sigmoid(torch.tensor(logit_yes - logit_no)).item()
    #print(f"P(Yes) = {prob:.4f}")

    return prob