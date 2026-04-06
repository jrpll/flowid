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

import sys
sys.path.insert(0, "cmmd-pytorch")

import distance
import embedding
import io_util

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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    question = "Have a look at this portrait of a person. Does it show facial injuries ? Only answer with Yes or No"
    vlm_score = num_processed = 0
    files_to_process = []
    for id in identities:
        id_path = os.path.join(data_dir, id)
        edited_photos = [e for e in os.listdir(id_path) if "edited" in e]
        for edited_photo in edited_photos:

            edited_path = os.path.join(id_path, edited_photo)
            edited_image = Image.open(edited_path)#.resize(original_size)

            tmp_score = vlm_single_image(model,processor,edited_image,question)
            vlm_score += tmp_score
            num_processed += 1

            if tmp_score < 0.5:
                files_to_process.append(edited_path)
    
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
    for id in identities:
        id_path = os.path.join(data_dir, id)
        edited_photos = [e for e in os.listdir(id_path) if "edited" in e]
        ref_emb_path = os.path.join(id_path, "emb.pt")
        ref_emb = torch.load(ref_emb_path)
        for edited in edited_photos:
            edited_path = os.path.join(id_path, edited)
            if edited_path in files_to_process:
                edited_emb = compute_emb(face_app, edited_path, True)
                sim = cosine_similarity(ref_emb, edited_emb)
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
    for id in identities:
        id_path = os.path.join(data_dir, id)
        edited_photos = [e for e in os.listdir(id_path) if "edited" in e]
        for edited_photo in edited_photos:
            edited_path = os.path.join(id_path, edited_photo)
            edited = Image.open(edited_path)
            if edited_path in files_to_process:
                text_path = os.path.join(id_path, edited_photo.replace("edited","prompt").replace(".jpg",".txt").replace("injured_",""))
                with open(text_path, "r") as t:
                    prompt = t.read()

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

    for id in identities:
        id_path = os.path.join(data_dir, id)
        edited_photos = [e for e in os.listdir(id_path) if "edited" in e]
        for edited in edited_photos:
            src = os.path.join(id_path, edited)
            if src in files_to_process:
                dst = os.path.join(fid_tmp_dir, f"{id}_{edited}")
                
                img = Image.open(src).convert("RGB")
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
    for id in identities:
        id_path = os.path.join(data_dir, id)
        edited_photos = [e for e in os.listdir(id_path) if "edited" in e]
        for edited_photo in edited_photos:
            ref_path = os.path.join(id_path, edited_photo.replace("edited_",""))
            original_size = Image.open(ref_path).size
            ref = torch.tensor(np.array(Image.open(ref_path))).permute(2,0,1).unsqueeze(0).cuda()
            ref = 2*(ref/255.)-1

            edited_path = os.path.join(id_path, edited_photo)
            edited = torch.tensor(np.array(Image.open(edited_path).resize(original_size))).permute(2,0,1).unsqueeze(0).cuda()
            edited = 2*(edited/255.)-1

            if edited_path in files_to_process:
                avg_lpips += lpips(ref,edited)
                num_processed += 1

    avg_lpips /= num_processed
    print(f"LPIPS: {avg_lpips}")

    del lpips
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
