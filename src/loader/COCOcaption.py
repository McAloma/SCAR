import sys, os
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import json
import random
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.cluster import KMeans


class COCOCaptionClusteringProcessor:
    def __init__(self, annotation_paths, image_dirs, mm_encoder, text_cluster_k=50, image_cluster_k=50, batch_size=1000, save_dir="output"):
        """
        annotation_paths: dict like {"train": "...captions_train2014.json", "val": "...captions_val2014.json"}
        image_dirs: dict like {"train": ".../train2014", "val": ".../val2014"}
        """
        self.annotation_paths = annotation_paths
        self.image_dirs = image_dirs
        self.mm_encoder = mm_encoder
        self.text_cluster_k = text_cluster_k
        self.image_cluster_k = image_cluster_k
        self.batch_size = batch_size
        self.save_dir = save_dir

        self.data = {"train": [], "val": []}
        self.image_embeddings = {}
        self.text_embeddings = defaultdict(list)

        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.save_dir, "image", split), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "text", split), exist_ok=True)

    def load_data(self):
        for split in ["train", "val"]:
            with open(self.annotation_paths[split], 'r') as f:
                annotations = json.load(f)

            image_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}
            image_captions = defaultdict(list)

            for ann in tqdm(annotations["annotations"], desc=f"[{split}] Parsing captions"):
                image_id = ann["image_id"]
                image_captions[image_id].append(ann["caption"])

            for image_id, captions in image_captions.items():
                filename = image_id_to_filename[image_id]
                full_path = os.path.join(self.image_dirs[split], filename)
                if os.path.exists(full_path):
                    self.data[split].append({
                        "filename": filename,
                        "filepath": full_path,
                        "captions": captions[:5]  # 取前5个 caption
                    })

    def encode_data_batch(self):
        for split in ["train", "val"]:
            all_image_paths = []
            all_captions = []
            filenames = []

            for item in tqdm(self.data[split], desc=f"[{split}] Collecting paths & captions"):
                all_image_paths.append(item["filepath"])
                all_captions.extend(item["captions"])
                filenames.append(item["filename"])

            print(f"[{split}] Encoding {len(all_image_paths)} images...")
            image_embs = self.mm_encoder.encode_images_batch(all_image_paths)

            print(f"[{split}] Encoding {len(all_captions)} captions...")
            text_embs = self.mm_encoder.encode_texts_batch(all_captions)

            for idx, item in enumerate(self.data[split]):
                fname = item["filename"]
                self.image_embeddings[fname] = image_embs[idx].numpy()
                self.text_embeddings[fname] = [text_embs[idx * 5 + i].numpy() for i in range(5)]

    def cluster_texts(self):
        all_text_embs = np.vstack([e for sublist in self.text_embeddings.values() for e in sublist])
        self.text_kmeans = KMeans(n_clusters=self.text_cluster_k, random_state=42).fit(all_text_embs)

        self.text_labels = {}
        self.image_labels_from_text = {}

        i = 0
        for fname, embs in self.text_embeddings.items():
            labels = self.text_kmeans.labels_[i:i+5]
            self.text_labels[fname] = labels.tolist()
            self.image_labels_from_text[fname] = int(Counter(labels).most_common(1)[0][0])
            i += 5

    def cluster_images(self):
        all_img_embs = np.vstack(list(self.image_embeddings.values()))
        self.image_kmeans = KMeans(n_clusters=self.image_cluster_k, random_state=42).fit(all_img_embs)

        self.image_labels = {}
        self.text_labels_from_image = {}

        for i, fname in enumerate(self.image_embeddings.keys()):
            label = self.image_kmeans.labels_[i]
            self.image_labels[fname] = int(label)
            self.text_labels_from_image[fname] = int(label)

    def generate_dataset_and_save(self):
        split_image_labels = {"train": [], "val": []}
        split_text_labels = {"train": [], "val": []}

        for split in ["train", "val"]:
            image_data = []
            text_data = []

            for item in self.data[split]:
                fname = item["filename"]
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

                split_image_labels[split].append(int(self.image_labels_from_text[fname]))
                split_text_labels[split].append(int(self.text_labels_from_image[fname]))

            def save_batches(data_list, folder):
                os.makedirs(folder, exist_ok=True)
                # for i in range(0, len(data_list), self.batch_size):
                for i in tqdm(range(0, len(data_list), self.batch_size), desc=f"Saving batches to {folder}"):
                    batch = data_list[i:i + self.batch_size]
                    with open(os.path.join(folder, f"batch_{i // self.batch_size}.json"), "w") as f:
                        json.dump(batch, f)

            save_batches(image_data, os.path.join(self.save_dir, "image", split))
            save_batches(text_data, os.path.join(self.save_dir, "text", split))

        label_stats = {
            "image_labels_from_text_train": dict(Counter(split_image_labels["train"])),
            "image_labels_from_text_val": dict(Counter(split_image_labels["val"])),
            "text_labels_from_image_train": dict(Counter(split_text_labels["train"])),
            "text_labels_from_image_val": dict(Counter(split_text_labels["val"])),
            "train_image_size": len(split_image_labels["train"]),
            "val_image_size": len(split_image_labels["val"]),
            "train_text_size": len(split_text_labels["train"]),
            "val_text_size": len(split_text_labels["val"]),
        }

        with open(os.path.join(self.save_dir, "label_counters.json"), "w") as f:
            json.dump(label_stats, f, indent=2)

    def run_all(self):
        self.load_data()
        self.encode_data_batch()
        self.cluster_texts()
        self.cluster_images()
        self.generate_dataset_and_save()




if __name__ == "__main__":
    from src.encoder.mm_encoders import CLIP_Encoder, SigLIP_Encoder, CoCa_Encoder

    # clip_encoder = CLIP_Encoder()
    # processor = COCOCaptionClusteringProcessor(
    #     annotation_paths={
    #         "train": "data/COCO-Caption/annotations/captions_train2014.json",
    #         "val": "data/COCO-Caption/annotations/captions_val2014.json"
    #     },
    #     image_dirs={
    #         "train": "data/COCO-Caption/train2014",
    #         "val": "data/COCO-Caption/val2014"
    #     },
    #     mm_encoder=clip_encoder,
    #     text_cluster_k=50,
    #     image_cluster_k=50,
    #     batch_size=1000,
    #     save_dir="data/embeddings/coco_caption/clip"
    # )
    # processor.run_all()

    clip_encoder = SigLIP_Encoder()
    processor = COCOCaptionClusteringProcessor(
        annotation_paths={
            "train": "data/COCO-Caption/annotations/captions_train2014.json",
            "val": "data/COCO-Caption/annotations/captions_val2014.json"
        },
        image_dirs={
            "train": "data/COCO-Caption/train2014",
            "val": "data/COCO-Caption/val2014"
        },
        mm_encoder=clip_encoder,
        text_cluster_k=50,
        image_cluster_k=50,
        batch_size=1000,
        save_dir="data/embeddings/coco_caption/siglip"
    )
    processor.run_all()

    clip_encoder = CoCa_Encoder()
    processor = COCOCaptionClusteringProcessor(
        annotation_paths={
            "train": "data/COCO-Caption/annotations/captions_train2014.json",
            "val": "data/COCO-Caption/annotations/captions_val2014.json"
        },
        image_dirs={
            "train": "data/COCO-Caption/train2014",
            "val": "data/COCO-Caption/val2014"
        },
        mm_encoder=clip_encoder,
        text_cluster_k=50,
        image_cluster_k=50,
        batch_size=1000,
        save_dir="data/embeddings/coco_caption/coca"
    )
    processor.run_all()