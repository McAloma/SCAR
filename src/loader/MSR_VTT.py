import sys, os, json, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from tqdm import tqdm




class MSRVTTProcessor:
    def __init__(
        self,
        video_dir,              
        split_jsons,            
        mm_encoder,            
        save_dir="msrvtt_output",
        batch_size=64,
        text_cluster_k=10,
        video_cluster_k=10
    ):
        self.video_dir = video_dir
        self.split_jsons = split_jsons
        self.mm_encoder = mm_encoder
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.text_cluster_k = text_cluster_k
        self.video_cluster_k = video_cluster_k

        self.splits = ["train", "val"]
        self.data = {}

        for split in self.splits:
            with open(self.split_jsons[split], "r") as f:
                self.data[split] = json.load(f)

        self.video_embeddings = {s: {} for s in self.splits}
        self.text_embeddings = {s: defaultdict(list) for s in self.splits}

    def encode_split(self, split):
        records = self.data[split]
        video_paths = []
        all_texts = []
        video_ids = []

        print(f"[{split}] Preparing data for batch encoding...")
        for record in tqdm(records, desc=f"Collecting {split}"):
            vid = record["video_id"]
            captions = record["caption"]
            video_path = os.path.join(self.video_dir, vid + ".mp4")

            video_paths.append(video_path)
            all_texts.extend(captions)
            video_ids.append(vid)

        print(f"[{split}] Encoding videos...")
        video_embs = self.mm_encoder.encode_videos_batch(video_paths, batch_size=self.batch_size)

        print(f"[{split}] Encoding texts...")
        text_embs = self.mm_encoder.encode_texts_batch(all_texts, batch_size=self.batch_size)

        print(f"[{split}] Mapping embeddings by video_id...")
        offset = 0
        for idx, vid in enumerate(tqdm(video_ids, desc=f"Mapping {split}")):
            self.video_embeddings[split][vid] = video_embs[idx].cpu().numpy()
            captions = self.data[split][idx]["caption"]
            n_cap = len(captions)
            self.text_embeddings[split][vid] = [text_embs[offset + i].cpu().numpy() for i in range(n_cap)]
            offset += n_cap

    # def cluster_and_save(self, split):
    #     print(f"[{split}] Clustering text embeddings...")
    #     all_text_embs = np.vstack([e for sublist in self.text_embeddings[split].values() for e in sublist])
    #     text_kmeans = KMeans(n_clusters=self.text_cluster_k, random_state=42).fit(all_text_embs)

    #     text_labels = {}
    #     video_labels_from_text = {}

    #     i = 0
    #     for vid, embs in self.text_embeddings[split].items():
    #         labels = text_kmeans.labels_[i : i + len(embs)]
    #         text_labels[vid] = labels.tolist()
    #         mode_label = Counter(labels).most_common(1)[0][0]
    #         video_labels_from_text[vid] = int(mode_label)
    #         i += len(embs)

    #     print(f"[{split}] Clustering video embeddings...")
    #     all_video_embs = np.vstack(list(self.video_embeddings[split].values()))
    #     video_kmeans = KMeans(n_clusters=self.video_cluster_k, random_state=42).fit(all_video_embs)

    #     video_labels = {}
    #     text_labels_from_video = {}

    #     for i, vid in enumerate(self.video_embeddings[split].keys()):
    #         label = video_kmeans.labels_[i]
    #         video_labels[vid] = int(label)
    #         text_labels_from_video[vid] = int(label)

    #     print(f"[{split}] Saving json batches...")
    #     video_data = []
    #     text_data = []

    #     for vid in self.video_embeddings[split]:
    #         video_data.append({
    #             "embedding": self.video_embeddings[split][vid].tolist(),
    #             "label": int(video_labels_from_text[vid]),
    #             "type": "video",
    #             "id": vid
    #         })

    #         text_embs = np.array(self.text_embeddings[split][vid])
    #         text_avg = text_embs.mean(axis=0)
    #         text_data.append({
    #             "embedding": text_avg.tolist(),
    #             "label": int(text_labels_from_video[vid]),
    #             "type": "text",
    #             "id": vid
    #         })

    #     random.shuffle(video_data)
    #     random.shuffle(text_data)

    #     def save_batches(data, folder):
    #         os.makedirs(folder, exist_ok=True)
    #         for i in range(0, len(data), self.batch_size):
    #             batch = data[i:i + self.batch_size]
    #             with open(os.path.join(folder, f"batch_{i // self.batch_size}.json"), "w") as f:
    #                 json.dump(batch, f)

    #     save_batches(video_data, os.path.join(self.save_dir, split, "video"))
    #     save_batches(text_data, os.path.join(self.save_dir, split, "text"))

    #     with open(os.path.join(self.save_dir, split, "label_counters.json"), "w") as f:
    #         json.dump({
    #             "video_labels_from_text": dict(Counter(video_labels_from_text.values())),
    #             "text_labels_from_video": dict(Counter(text_labels_from_video.values())),
    #             "video_size": len(video_data),
    #             "text_size": len(text_data),
    #         }, f, indent=2)


    def cluster_and_save(self, split):
        print(f"[{split}] Averaging text embeddings...")
        # 对每个视频的多个文本嵌入取平均
        avg_text_embeddings = {
            vid: np.mean(np.array(embs), axis=0)
            for vid, embs in self.text_embeddings[split].items()
        }

        print(f"[{split}] Clustering average text embeddings...")
        all_avg_text_embs = np.vstack(list(avg_text_embeddings.values()))
        text_kmeans = KMeans(n_clusters=self.text_cluster_k, random_state=42).fit(all_avg_text_embs)
        video_labels_from_text = {
            vid: int(label)
            for vid, label in zip(avg_text_embeddings.keys(), text_kmeans.labels_)
        }

        print(f"[{split}] Clustering video embeddings...")
        all_video_embs = np.vstack(list(self.video_embeddings[split].values()))
        video_kmeans = KMeans(n_clusters=self.video_cluster_k, random_state=42).fit(all_video_embs)
        text_labels_from_video = {
            vid: int(label)
            for vid, label in zip(self.video_embeddings[split].keys(), video_kmeans.labels_)
        }

        print(f"[{split}] Preparing data for saving...")
        video_data = []
        text_data = []

        for vid in self.video_embeddings[split]:
            video_data.append({
                "embedding": self.video_embeddings[split][vid].tolist(),
                "label": video_labels_from_text[vid],  # 视频标签来自文本聚类
                "type": "video",
                "id": vid
            })

            text_data.append({
                "embedding": avg_text_embeddings[vid].tolist(),  # 文本用平均嵌入
                "label": text_labels_from_video[vid],  # 文本标签来自视频聚类
                "type": "text",
                "id": vid
            })

        random.shuffle(video_data)
        random.shuffle(text_data)

        def save_batches(data, folder):
            os.makedirs(folder, exist_ok=True)
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                with open(os.path.join(folder, f"batch_{i // self.batch_size}.json"), "w") as f:
                    json.dump(batch, f)

        save_batches(video_data, os.path.join(self.save_dir, split, "video"))
        save_batches(text_data, os.path.join(self.save_dir, split, "text"))

        with open(os.path.join(self.save_dir, split, "label_counters.json"), "w") as f:
            json.dump({
                "video_labels_from_text": dict(Counter(video_labels_from_text.values())),
                "text_labels_from_video": dict(Counter(text_labels_from_video.values())),
                "video_size": len(video_data),
                "text_size": len(text_data),
            }, f, indent=2)

    def run_all(self):
        for split in self.splits:
            self.encode_split(split)
            self.cluster_and_save(split)


if __name__ == "__main__":
    from src.encoder.vt_encoder import XCLIP_Encoder 

    processor = MSRVTTProcessor(
        video_dir="data/MSR_VTT/MSRVTT_Videos/video",
        split_jsons={
            "train": "data/MSR_VTT/msrvtt_train_9k.json",
            "val": "data/MSR_VTT/msrvtt_test_1k.json"
        },
        mm_encoder=XCLIP_Encoder(),
        save_dir="data/embeddings/msrvtt",
        batch_size=1024,
        text_cluster_k=5,
        video_cluster_k=5
    )
    processor.run_all()


    from src.encoder.vt_encoder import VideoCLIP_Encoder 

    processor = MSRVTTProcessor(
        video_dir="data/MSR_VTT/MSRVTT_Videos/video",
        split_jsons={
            "train": "data/MSR_VTT/msrvtt_train_9k.json",
            "val": "data/MSR_VTT/msrvtt_test_1k.json"
        },
        mm_encoder=VideoCLIP_Encoder(),
        save_dir="data/embeddings/msrvtt/videoclip",
        batch_size=1024,
        text_cluster_k=5,
        video_cluster_k=5
    )
    processor.run_all()