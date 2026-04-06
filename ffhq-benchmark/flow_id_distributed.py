from flow_id import FlowID
from diffusers import StableDiffusion3Pipeline
import torch
from PIL import Image
import os
import json
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import argparse

def get_edit_params(neg_prompt,kwd):
    if ("dark hair" in neg_prompt or "light hair" in neg_prompt) and "hair" in kwd:
        num_skipped_steps = 3
        kernel_size = 15
        remove = True
    elif "hat" in kwd:
        if "hat" in neg_prompt:
            num_skipped_steps = 0
            kernel_size = 21
            remove = True
        else:
            num_skipped_steps = 2
            kernel_size = 15
            remove = False
    elif "glasses" in kwd:
        if "glasses" in neg_prompt:
            num_skipped_steps = 10
            kernel_size = 15
            remove = True
        else:
            num_skipped_steps = 8
            kernel_size = 25
            remove = False
    elif "hair"==kwd:
        if "hair" in neg_prompt:
            num_skipped_steps = 4
            kernel_size = 19
            remove = True
        else:
            num_skipped_steps = 2
            kernel_size = 19
            remove = False
    elif "beard" in kwd:
        if "beard" in neg_prompt:
            num_skipped_steps = 3
            kernel_size = 29
            remove = True
        else:
            num_skipped_steps = 2
            kernel_size = 25
            remove = False
    elif "earrings" in kwd:
        if "earrings" in neg_prompt:
            num_skipped_steps = 7
            kernel_size = 15
            remove = True
        else:
            num_skipped_steps = 7
            kernel_size = 11
            remove = False
    elif "smiling" in kwd:
        if "smiling" in neg_prompt:
            num_skipped_steps = 10
            kernel_size = 11
            remove = True
        else:
            num_skipped_steps = 8
            kernel_size = 11
            remove = False
    return num_skipped_steps, kernel_size, remove

class FFHQ(Dataset):
    def __init__(
            self,
            data_path
    ):
        self.ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and len(os.listdir(os.path.join(data_path, d))) < 3]
        self.size = len(self.ids)
        print(f"PROCESSING {self.size} IDENTITIES")
                
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
        simulated_batch_size=8,
        batch_size=4,
        num_inversion_steps=50,
        fine_tune=True,
        num_skipped_steps=3,
        kernel_size=15,
        data_path="FFHQ-FlowID",
        pasting_cutoff=0
    ):
        self.simulated_batch_size = simulated_batch_size
        self.batch_size = batch_size
        self.num_inversion_steps = num_inversion_steps
        self.fine_tune = fine_tune
        self.num_skipped_steps = num_skipped_steps
        self.kernel_size = kernel_size
        self.editor = FlowID()
        self.data_path = data_path
        self.pasting_cutoff = pasting_cutoff

    def edit(self,data):
        for batch in data:
            filename = batch[0]+".png"
            id_folder = os.path.join(self.data_path,batch[0])
            img_path = os.path.join(self.data_path,batch[0],filename)
            src_img = Image.open(img_path).resize((1024,1024)).convert("RGB")
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
            self.editor.ft_invert(
                img = src_img,
                prompt = src_prompt,
                simulated_batch_size = self.simulated_batch_size,
                batch_size = self.batch_size,
                num_inversion_steps = self.num_inversion_steps,
                fine_tune = self.fine_tune
            )
            for i in range(1):
                tgt_prompt = edit_prompts[i]
                neg_prompt = neg_prompts[i]
                kwd = kwds[i]
                num_skipped_steps, kernel_size, remove = get_edit_params(neg_prompt,kwd)
                edited = self.editor.edit(
                    prompt=tgt_prompt,
                    neg_prompt=neg_prompt,
                    concept=kwd,
                    num_skipped_steps=num_skipped_steps,
                    remove=remove,
                    kernel_size=kernel_size,
                    pasting_cutoff=self.pasting_cutoff
                )
                self.editor.validate_edit()
                edited.save(os.path.join(id_folder,f"{i}_{filename}"))

def main(args):
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    #pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16).to("cuda")
    distributed_editor = DistributedEditor(
        #pipe=pipe,
        simulated_batch_size=args.simulated_batch_size,
        batch_size=args.batch_size,
        fine_tune=args.fine_tune,
        data_path = args.data_path,
        pasting_cutoff = args.pasting_cutoff
    )
    dataset = FFHQ(data_path = args.data_path)
    data = DataLoader(dataset,batch_size=1,sampler=DistributedSampler(dataset),collate_fn=custom_collate)
    distributed_editor.edit(data)
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated_batch_size",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--pasting_cutoff",type=int,default=0)
    parser.add_argument("--fine_tune",action="store_true")
    parser.add_argument("--data_path",type=str, default="FFHQ-FlowID")
    args = parser.parse_args()
    main(args)
    

