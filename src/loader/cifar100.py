import sys, os, pickle, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.encoder.encoders import ResNet_Encoder, ViT_Encoder, DINO_Encoder

class CIFAR100_Dataloader():
    def __init__(self, encoder_type="resnet"):
        if encoder_type == "resnet":
            self.encoder = ResNet_Encoder()
            self.save_path = "./data/embeddings/cifar100/resnet"
        elif encoder_type == "vit":
            self.encoder = ViT_Encoder()
            self.save_path = "./data/embeddings/cifar100/vit"
        elif encoder_type == "dino":
            self.encoder = DINO_Encoder()
            self.save_path = "./data/embeddings/cifar100/dino"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'resnet', 'vit', or 'dino'.")
        
        os.makedirs(self.save_path, exist_ok=True)

    def load_cifar100_batch_as_pil(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            raw_images = data[b'data']  # shape: [N, 3072]
            labels = data[b'fine_labels']
            filenames = data.get(b'filenames', [f"img_{i}.png".encode() for i in range(len(labels))])  # fallback if no filenames

            images = []
            for i in range(len(raw_images)):
                img_flat = raw_images[i]
                r = img_flat[0:1024].reshape(32, 32)
                g = img_flat[1024:2048].reshape(32, 32)
                b = img_flat[2048:].reshape(32, 32)
                img = np.stack([r, g, b], axis=2).astype(np.uint8)
                pil_img = Image.fromarray(img)
                images.append(pil_img)

            filenames = [fn.decode('utf-8') for fn in filenames]
            return images, labels, filenames
        
    def process_and_save_cifar100(self, data_dir):
        batch_size = 1000
        all_data = []

        # —————————————————————— Train Data ——————————————————————
        file_path = os.path.join(data_dir, "train")
        images, labels, filenames = self.load_cifar100_batch_as_pil(file_path)

        for idx in tqdm(range(0, len(images), batch_size), desc=f"Train Batch", ascii=True):
            img_batch = images[idx:idx + batch_size]
            label_batch = labels[idx:idx + batch_size]
            fname_batch = filenames[idx:idx + batch_size]

            embeddings = self.encoder.encode_batch(img_batch)

            for fname, label, emb in zip(fname_batch, label_batch, embeddings):
                all_data.append({
                    "filename": fname,
                    "label": label,
                    "embedding": emb.tolist()
                })

        peek_save_path = os.path.join(self.save_path, "cifar100_embeddings_peek.json")
        with open(peek_save_path, 'w') as f:
            for item in all_data[:5]:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')

        save_path = os.path.join(self.save_path, "cifar100_embeddings.json")
        with open(save_path, 'w') as f:
            for item in all_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')

        # —————————————————————— Test Data ——————————————————————
        file_path = os.path.join(data_dir, "test")
        images, labels, filenames = self.load_cifar100_batch_as_pil(file_path)

        for idx in tqdm(range(0, len(images), batch_size), desc=f"Test Batch", ascii=True):
            img_batch = images[idx:idx + batch_size]
            label_batch = labels[idx:idx + batch_size]
            fname_batch = filenames[idx:idx + batch_size]

            embeddings = self.encoder.encode_batch(img_batch)

            for fname, label, emb in zip(fname_batch, label_batch, embeddings):
                all_data.append({
                    "filename": fname,
                    "label": label,
                    "embedding": emb.tolist()
                })

        save_path = os.path.join(self.save_path, "cifar100_test_embeddings.json")
        with open(save_path, 'w') as f:
            for item in all_data:
                json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        

if __name__ == "__main__":
    data_path = "./data/cifar-100-python"

    for encoder in ["resnet", "vit", "dino"]:
        dataloader = CIFAR100_Dataloader(encoder_type=encoder)
        dataloader.process_and_save_cifar100(data_path)