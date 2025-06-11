import sys, os, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from src.encoder.encoders import ResNet_Encoder, ViT_Encoder, DINO_Encoder


class Imagenet_Dataloader():
    def __init__(self, encoder_type="resnet"):
        if encoder_type == "resnet":
            self.encoder = ResNet_Encoder()
            self.save_path = "./data/embeddings/imagenet/resnet"
        elif encoder_type == "vit":
            self.encoder = ViT_Encoder()
            self.save_path = "./data/embeddings/imagenet/vit"
        elif encoder_type == "dino":
            self.encoder = DINO_Encoder()
            self.save_path = "./data/embeddings/imagenet/dino"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'resnet', 'vit', or 'dino'.")
        
        os.makedirs(self.save_path, exist_ok=True)


    def _process_split(self, split_path, split_name):
        transform = transforms.Compose([  
            transforms.Resize((224, 224)),
            transforms.ToTensor(),             
        ])
        dataset = datasets.ImageFolder(root=split_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8)

        for idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            embeddings = self.encoder.encode_batch(images, do_rescale=False)

            batch_data = []
            for emb, label in zip(embeddings, labels):
                batch_data.append({
                    "label": int(label.item()),
                    "embedding": emb.tolist()
                })

            save_file = os.path.join(self.save_path, f"{split_name}", f"batch_{idx}.json")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(batch_data, f, indent=2)

    def process_and_save_imagenet(self, data_dir):
        # self._process_split(os.path.join(data_dir, "train"), "train")
        self._process_split(os.path.join(data_dir, "val"), "val")
        

if __name__ == "__main__":
    data_path = "./data/imagenet-1k"
    for encoder in ["resnet", "vit", "dino"]:
        dataloader = Imagenet_Dataloader(encoder_type=encoder)
        dataloader.process_and_save_imagenet(data_path)

    # dataloader = Imagenet_Dataloader(encoder_type="resnet")
    # dataloader.process_and_save_imagenet(data_path)

    # python3 src/loader/imagenet.py