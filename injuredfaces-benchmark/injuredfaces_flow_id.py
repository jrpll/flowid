from flow_id import FlowID
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, Dataset
from PIL import Image
import argparse
import os
import torch

class InjuredFaces(Dataset):
    def __init__(self,dataset_path):
        self.data_path = dataset_path
        
        identities = [id for id in os.listdir(dataset_path) if ".DS" not in id]
        self.injured_filepaths = []
        for identity in identities:
            id_path = os.path.join(dataset_path,identity)
            for file in os.listdir(id_path):
                if "injured" in file:
                    injured_filepath = os.path.join(id_path,file)
                    self.injured_filepaths.append(injured_filepath)
        
        self.size = len(self.injured_filepaths)
        print(f"PROCESSING {self.size} IMGS")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image_path = self.injured_filepaths[index]
        prompt_path = image_path.replace("injured","prompt").replace(".jpg",".txt")

        image = Image.open(image_path).resize((1024,1024))
        with open(prompt_path,"r") as p:
            prompt = p.read()

        return {
            "image":image,
            "prompt":prompt,
            "path":image_path
        }

def custom_collate(batch):
    return {
        "image": [item["image"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "path": [item["path"] for item in batch]
    }

class DistributedFlowID:
    def __init__(
        self,
        num_train_steps,
        num_skipped_steps,
        ft_batch_size,
        dataset_path
    ):
        self.flow_id = FlowID()
        self.num_train_steps = num_train_steps
        self.num_skipped_steps = num_skipped_steps
        self.ft_batch_size = ft_batch_size
        self.data_path = dataset_path

    def edit(self,data):
        for batch in data:
            image = batch["image"][0]
            prompt = batch["prompt"][0]
            image_path = batch["path"][0]

            rank = int(os.environ.get("LOCAL_RANK", 0))
            print(f"[GPU {rank}] Processing: {image_path}")

            num_train_steps = self.num_train_steps
            num_skipped_steps = self.num_skipped_steps

            negative_prompt = prompt.replace("a person's face, ","")

            self.flow_id.ft_invert(
                img = image,
                prompt = prompt,
                num_train_steps = num_train_steps,
                simulated_batch_size = 10,
                batch_size = self.ft_batch_size
            )

            edited = self.flow_id.edit(
                prompt = "a person's face",
                neg_prompt = negative_prompt,
                num_skipped_steps = num_skipped_steps    
            )[0]

            save_path = image_path.replace("injured","edited")
            edited.save(save_path)
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_train_steps", type=int, required=True)
    parser.add_argument("--num_skipped_steps", type=int, required=True)
    parser.add_argument("--ft_batch_size", type=int, required=True)
    args = parser.parse_args()

    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    distributed_flow_id = DistributedFlowID(
        num_train_steps = args.num_train_steps,
        num_skipped_steps = args.num_skipped_steps,
        ft_batch_size = args.ft_batch_size,
        dataset_path = args.dataset_path
    )

    dataset = InjuredFaces(args.dataset_path)
    dataloader = DataLoader(dataset,sampler=DistributedSampler(dataset), collate_fn=custom_collate)

    distributed_flow_id.edit(dataloader)

    destroy_process_group()

if __name__ == "__main__":
    main()