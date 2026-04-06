from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import os
from torch.nn.functional import cosine_similarity
import torch
import cv2
from eval_utils import eval_clip_t, vlm_two_images, vlm_single_image
import clip
from pytorch_fid import fid_score
import argparse
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import shutil
import gc
import json

import sys
sys.path.insert(0, "cmmd-pytorch")

import distance
import embedding
import io_util

from tqdm import tqdm

from pathlib import Path

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def compute_emb(face_app, img_path, use_detection=True):
    input_image = Image.open(img_path)
    cv2_image = np.array(input_image.convert("RGB"))
    cv2_image = cv2_image[:, :, ::-1]
    
    if use_detection:
        try:
            faces = face_app.get(cv2_image)
            bboxes = np.array([face.bbox for face in faces])
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            largest_idx = np.argmax(areas)
            emb = torch.tensor(faces[largest_idx].normed_embedding).unsqueeze(0)
        except:
            img = cv2.resize(cv2_image, (112, 112))
            rec_model = face_app.models['recognition']
            emb = rec_model.get_feat(img)
            emb = emb / np.linalg.norm(emb)
            emb = torch.tensor(emb)
    else:
        img = cv2.resize(cv2_image, (112, 112))
        rec_model = face_app.models['recognition']
        emb = rec_model.get_feat(img)
        emb = emb / np.linalg.norm(emb)
        emb = torch.tensor(emb)
        
    return emb

def get_question_with_prompts(prompts):
    kw_1 = prompts["kw_1"]
    tgt = prompts["edit_prompt_1"]
    add = None
    if "earrings" in kw_1:
        question = "have earrings"
        if "earrings" in tgt:
            add = True
        else:
            add = False
    elif "glasses" in kw_1:
        question = "have glasses"
        if "glasses" in tgt:
            add = True
        else:
            add = False
    elif "hat" in kw_1:
        question = "have a hat"
        if "hat" in tgt:
            add = True
        else:
            add = False
    elif "beard" in kw_1:
        question = "have a beard"
        if "beard" in tgt:
            add = True
        else:
            add = False
    elif "smiling" in kw_1:
        question = "have a smile"
        if "smiling" in tgt:
            add = True
        else:
            add = False
    elif "hair" in kw_1:
        if "dark hair" in tgt:
            question = "have dark hair"
            add = True
        elif "light hair" in tgt:
            question = "have light hair"
            add = True
        elif "with hair" in tgt:
            question = "have hair"
            add = True
        elif "bald" in tgt:
            question = "have hair"
            add = False

    return f"Have a look at this portrait of a person. Does the person {question} ? Only answer with Yes or No", add

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to edited images")
    parser.add_argument("--fid_tmp_dir", type=str, default="fid_tmp_dir", help="Path to temporary dir for FID")
    parser.add_argument("--imgs_fid_path", type=str, default=None, help="Path to ref images for FID")
    parser.add_argument("--fid_ref_path", type=str, default=None, required=True, help="Path to precomputed reference embeddings for FID")
    parser.add_argument("--cmmd_ref_path", type=str, default=None, help="Path to precomputed reference embeddings for CMMD")
    args = parser.parse_args()

    data_dir = args.data_dir
    fid_tmp_dir = args.fid_tmp_dir
    fid_ref_path = args.fid_ref_path

    identities = [i for i in os.listdir(data_dir) if ".DS" not in i]

    #############
    # VLM JUDGE #
    #############

    files_to_process = []
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    vlm_score = num_processed = 0
    for id in tqdm(identities):
        json_path = os.path.join(data_dir, id, f"{id}.json")
        with open(json_path, "r") as f:
            prompts = json.load(f)
        img_path = os.path.join(data_dir, id, f"0_{id}.png")
        edited = Image.open(img_path)
        question, add = get_question_with_prompts(prompts)
        tmp_vlm_score = vlm_single_image(model,processor,edited,question)
        if add == True:
            vlm_score += tmp_vlm_score
            actual_score = tmp_vlm_score
        else:
            vlm_score += 1 - tmp_vlm_score
            actual_score = 1 - tmp_vlm_score
        if actual_score > 0.5:
            files_to_process.append(img_path)

        num_processed += 1
    
    vlm_score /= num_processed
    print("VLM",vlm_score)

    #################
    # PROCESSING ID #
    #################

    snapshot_download(
        "fal/AuraFace-v1",
        local_dir="models/auraface",
    )
    face_app = FaceAnalysis(
        name="auraface",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        root=".",
    )
    face_app.prepare(ctx_id=0, det_thresh=0.1)

    IR = num_processed = 0
    for img_path in tqdm(files_to_process):
        id = os.path.basename(os.path.dirname(img_path))
        src_img_path = os.path.join(os.path.dirname(img_path), f"{id}.png")
        #src_img_path = os.path.join(data_dir,id,f"{id}.png")
        edited_img_path = os.path.join(data_dir,id,f"0_{id}.png")

        src_emb = compute_emb(face_app, src_img_path, True)
        edited_emb = compute_emb(face_app, edited_img_path, True)
        sim = cosine_similarity(src_emb, edited_emb)
        IR += sim
        num_processed += 1

    IR /= num_processed
    print("IR", IR)

    del face_app
    gc.collect()
    torch.cuda.empty_cache()

    ###################
    # PROCESSING CLIP #
    ###################
    clip_model, clip_transform = clip.load("ViT-B/32", "cuda")
    CLIP = num_processed = 0
    for img_path in tqdm(files_to_process):
        id = os.path.basename(os.path.dirname(img_path))
        json_path = os.path.join(data_dir, id, f"{id}.json")
        with open(json_path, "r") as f:
            prompts = json.load(f)
        prompt = prompts["edit_prompt_1"]
        edited = Image.open(img_path)
        clip_value = eval_clip_t("cuda", edited, prompt, clip_model, clip_transform)
        CLIP += clip_value
        num_processed += 1

    CLIP /= num_processed
    print("CLIP", CLIP)

    del clip_model
    gc.collect()
    torch.cuda.empty_cache()

    ##################
    # PROCESSING FID #
    ##################

    os.makedirs(fid_tmp_dir, exist_ok=True)

    #for id in tqdm(identities):
    for img_path in tqdm(files_to_process):
        id = os.path.basename(os.path.dirname(img_path))
        dst = os.path.join(fid_tmp_dir, f"{id}_tmp.jpg")
        
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1024, 1024), Image.LANCZOS)
        img.save(dst)

    fid_value = fid_score.calculate_fid_given_paths(
        paths=[fid_tmp_dir, fid_ref_path],
        batch_size=50,
        device="cuda",
        dims=2048
    )
    print(f"FID: {fid_value}")

    ###################
    # PROCESSING CMMD #
    ###################

    embedding_model = embedding.ClipEmbeddingModel()
    gen_embs = io_util.compute_embeddings_for_dir(args.fid_tmp_dir, embedding_model, batch_size=32, max_count=-1).astype("float32") 
    ref_embs = io_util.compute_embeddings_for_dir(args.imgs_fid_path, embedding_model, batch_size=32, max_count=-1).astype("float32") if args.cmmd_ref_path is None else np.load(args.cmmd_ref_path)
    np.save("ref_cmmd.npy",ref_embs)
    cmmd_value = distance.mmd(ref_embs, gen_embs).numpy()
    print(f"CMMD: {cmmd_value:.3f}")

    shutil.rmtree(fid_tmp_dir)

    #################
    # PROCESS LPIPS #
    #################

    avg_lpips = num_processed = 0
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
    #for id in tqdm(identities):
    for img_path in tqdm(files_to_process):
        id = os.path.basename(os.path.dirname(img_path))
        src_img_path = os.path.join(data_dir,id,f"{id}.png")
        edited_img_path = os.path.join(data_dir,id,f"0_{id}.png")
        ref = torch.tensor(np.array(Image.open(src_img_path))).permute(2,0,1).unsqueeze(0).cuda()
        ref = 2*(ref/255.)-1

        edited = torch.tensor(np.array(Image.open(img_path))).permute(2,0,1).unsqueeze(0).cuda()
        edited = 2*(edited/255.)-1

        avg_lpips += lpips(ref,edited)
        num_processed += 1

    avg_lpips /= num_processed
    print(f"LPIPS: {avg_lpips}")

    del lpips
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
