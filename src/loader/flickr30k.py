import sys, os
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import json
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from tqdm import tqdm


class Flickr30kClusteringProcessor:
    def __init__(self, csv_path, mm_encoder, image_dir=None, text_cluster_k=10, image_cluster_k=10, batch_size=1000, save_dir="output"):
        self.csv_path = csv_path
        self.mm_encoder = mm_encoder  # 实例化的 CLIP_Encoder
        self.image_dir = image_dir or ""
        self.text_cluster_k = text_cluster_k
        self.image_cluster_k = image_cluster_k
        self.batch_size = batch_size
        self.save_dir = save_dir

        self.data = pd.read_csv(csv_path)
        self.image_embeddings = {}              # filename -> image embedding
        self.text_embeddings = defaultdict(list)  # filename -> list of 5 text embeddings

        # os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
        # os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)

    def encode_data_batch(self):
        all_image_paths = []
        all_texts = []
        filename_list = []  # index align to data rows

        print("Preparing data for batch encoding...")

        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Collecting inputs"):
            filename = row['filename']
            sentences = eval(row['raw'])  # List of 5 captions
            image_path = os.path.join(self.image_dir, filename)

            all_image_paths.append(image_path)
            all_texts.extend(sentences)
            filename_list.append(filename)

        print("Encoding images...")
        image_embs = self.mm_encoder.encode_images_batch(all_image_paths)
        print("Encoding texts...")
        text_embs = self.mm_encoder.encode_texts_batch(all_texts)

        print("Distributing embeddings by filename...")
        for idx, fname in enumerate(tqdm(filename_list, desc="Mapping embeddings")):
            self.image_embeddings[fname] = image_embs[idx].numpy()
            start = idx * 5
            self.text_embeddings[fname] = [text_embs[start + i].numpy() for i in range(5)]

    def cluster_texts(self):
        all_text_embs = np.vstack([e for sublist in self.text_embeddings.values() for e in sublist])
        self.text_kmeans = KMeans(n_clusters=self.text_cluster_k, random_state=42).fit(all_text_embs)

        self.text_labels = {}  # filename -> list of 5 labels
        self.image_labels_from_text = {}

        i = 0
        for fname, embs in self.text_embeddings.items():
            labels = self.text_kmeans.labels_[i:i+5]
            self.text_labels[fname] = labels.tolist()
            mode_label = Counter(labels).most_common(1)[0][0]
            self.image_labels_from_text[fname] = int(mode_label)
            i += 5

    def cluster_images(self):
        all_img_embs = np.vstack(list(self.image_embeddings.values()))
        self.image_kmeans = KMeans(n_clusters=self.image_cluster_k, random_state=42).fit(all_img_embs)

        self.image_labels = {}  # filename -> label
        self.text_labels_from_image = {}  # filename -> label

        for i, fname in enumerate(self.image_embeddings.keys()):
            label = self.image_kmeans.labels_[i]
            self.image_labels[fname] = int(label)
            self.text_labels_from_image[fname] = int(label)

    def generate_dataset_and_save(self):
        image_data = []
        text_data = []

        for fname in self.image_embeddings:
            image_data.append({
                "embedding": self.image_embeddings[fname].tolist(),
                "label": int(self.image_labels_from_text[fname]),
                "type": "image",
                "id": fname
            })

            text_embs = np.array(self.text_embeddings[fname])
            text_avg_emb = text_embs.mean(axis=0)
            text_data.append({
                "embedding": text_avg_emb.tolist(),
                "label": int(self.text_labels_from_image[fname]),
                "type": "text",
                "id": fname
            })

        # Shuffle and split separately
        random.shuffle(image_data)
        random.shuffle(text_data)

        split_image = int(len(image_data) * 0.8)
        split_text = int(len(text_data) * 0.8)

        image_train, image_test = image_data[:split_image], image_data[split_image:]
        text_train, text_test = text_data[:split_text], text_data[split_text:]

        def save_batches(data, folder):
            os.makedirs(folder, exist_ok=True)
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                with open(os.path.join(folder, f"batch_{i // self.batch_size}.json"), "w") as f:
                    json.dump(batch, f)

        # Save to desired folder structure
        save_batches(image_train, os.path.join(self.save_dir, "image", "train"))
        save_batches(image_test, os.path.join(self.save_dir, "image", "test"))
        save_batches(text_train, os.path.join(self.save_dir, "text", "train"))
        save_batches(text_test, os.path.join(self.save_dir, "text", "test"))

        # Save statistics
        with open(os.path.join(self.save_dir, "label_counters.json"), "w") as f:
            json.dump({
                "image_labels_from_text": dict(Counter(self.image_labels_from_text.values())),
                "text_labels_from_image": dict(Counter(self.text_labels_from_image.values())),
                "image_train_size": len(image_train),
                "image_test_size": len(image_test),
                "text_train_size": len(text_train),
                "text_test_size": len(text_test)
            }, f, indent=2)

    def run_all(self):
        self.encode_data_batch()
        self.cluster_texts()
        self.cluster_images()
        self.generate_dataset_and_save()


if __name__ == "__main__":
    from src.encoder.mm_encoders import CLIP_Encoder, SigLIP_Encoder, CoCa_Encoder

    # clip_encoder = CLIP_Encoder()
    # processor = Flickr30kClusteringProcessor(
    #     csv_path="data/Flickr30k/flickr_annotations_30k.csv",
    #     mm_encoder=clip_encoder,
    #     image_dir="data/Flickr30k/flickr30k-images",
    #     text_cluster_k=10,
    #     image_cluster_k=10,
    #     batch_size=1000,
    #     save_dir="data/embeddings/flickr30k/clip"
    # )
    # processor.run_all()

    # siglip_encoder = SigLIP_Encoder()
    # processor = Flickr30kClusteringProcessor(
    #     csv_path="data/Flickr30k/flickr_annotations_30k.csv",
    #     mm_encoder=siglip_encoder,
    #     image_dir="data/Flickr30k/flickr30k-images",
    #     text_cluster_k=10,
    #     image_cluster_k=10,
    #     batch_size=1000,
    #     save_dir="data/embeddings/flickr30k/siglip"
    # )
    # processor.run_all()

    coca_encoder = CoCa_Encoder()
    processor = Flickr30kClusteringProcessor(
        csv_path="data/Flickr30k/flickr_annotations_30k.csv",
        mm_encoder=coca_encoder,
        image_dir="data/Flickr30k/flickr30k-images",
        text_cluster_k=10,
        image_cluster_k=10,
        batch_size=1000,
        save_dir="data/embeddings/flickr30k/coca"
    )
    processor.run_all()