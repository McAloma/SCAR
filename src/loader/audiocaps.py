import sys, os, json, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from datasets import load_dataset


class AudioCapsProcessor:
    def __init__(
        self,
        at_encoder,
        save_dir="data/embeddings/audiocaps",
        batch_size=256,
        text_cluster_k=5,
        audio_cluster_k=5
    ):
        self.at_encoder = at_encoder
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.text_cluster_k = text_cluster_k
        self.audio_cluster_k = audio_cluster_k

        # Load and split dataset
        full_dataset = load_dataset("OpenSound/AudioCaps", cache_dir="data/Audiocaps")
        self.dataset = {
            "train": full_dataset["train"],
            "val": full_dataset["validation"] if "validation" in full_dataset else full_dataset["test"],
        }

        self.audio_embeddings = {s: {} for s in self.dataset}
        self.text_embeddings = {s: defaultdict(list) for s in self.dataset}

    def encode_split(self, split):
        subset = self.dataset[split]
        audio_arrays = []
        sampling_rates = []
        all_captions = []
        audio_ids = []

        print(f"[{split}] Preparing audio and text...")
        for sample in tqdm(subset, desc=f"Collecting {split}"):
            audio_arrays.append(sample["audio"]["array"])
            sampling_rates.append(sample["audio"]["sampling_rate"])
            all_captions.append(sample["caption"])
            audio_ids.append(str(sample["audiocap_id"]))

        print(f"[{split}] Encoding audios...")
        audio_embs = self.at_encoder.encode_audios_from_dataset(audio_arrays, sampling_rates, batch_size=self.batch_size)

        print(f"[{split}] Encoding texts...")
        text_embs = self.at_encoder.encode_texts_batch(all_captions, batch_size=self.batch_size)

        print(f"[{split}] Mapping embeddings by audio_id...")
        for idx, aid in enumerate(tqdm(audio_ids, desc=f"Mapping {split}")):
            self.audio_embeddings[split][aid] = audio_embs[idx].cpu().numpy()
            self.text_embeddings[split][aid] = [text_embs[idx].cpu().numpy()]

    def cluster_and_save(self, split):
        print(f"[{split}] Clustering text embeddings...")
        all_text_embs = np.vstack([e[0] for e in self.text_embeddings[split].values()])
        text_kmeans = KMeans(n_clusters=self.text_cluster_k, random_state=42).fit(all_text_embs)

        text_labels = {}
        audio_labels_from_text = {}
        for i, aid in enumerate(self.text_embeddings[split]):
            label = text_kmeans.labels_[i]
            text_labels[aid] = [int(label)]
            audio_labels_from_text[aid] = int(label)

        print(f"[{split}] Clustering audio embeddings...")
        all_audio_embs = np.vstack(list(self.audio_embeddings[split].values()))
        audio_kmeans = KMeans(n_clusters=self.audio_cluster_k, random_state=42).fit(all_audio_embs)

        audio_labels = {}
        text_labels_from_audio = {}
        for i, aid in enumerate(self.audio_embeddings[split]):
            label = audio_kmeans.labels_[i]
            audio_labels[aid] = int(label)
            text_labels_from_audio[aid] = int(label)

        print(f"[{split}] Saving json batches...")
        audio_data = []
        text_data = []
        for aid in self.audio_embeddings[split]:
            audio_data.append({
                "embedding": self.audio_embeddings[split][aid].tolist(),
                "label": audio_labels_from_text[aid],
                "type": "audio",
                "id": aid
            })

            text_emb = self.text_embeddings[split][aid][0]
            text_data.append({
                "embedding": text_emb.tolist(),
                "label": text_labels_from_audio[aid],
                "type": "text",
                "id": aid
            })

        random.shuffle(audio_data)
        random.shuffle(text_data)

        def save_batches(data, folder):
            os.makedirs(folder, exist_ok=True)
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                with open(os.path.join(folder, f"batch_{i // self.batch_size}.json"), "w") as f:
                    json.dump(batch, f)

        save_batches(audio_data, os.path.join(self.save_dir, split, "audio"))
        save_batches(text_data, os.path.join(self.save_dir, split, "text"))

        with open(os.path.join(self.save_dir, split, "label_counters.json"), "w") as f:
            json.dump({
                "audio_labels_from_text": dict(Counter(audio_labels_from_text.values())),
                "text_labels_from_audio": dict(Counter(text_labels_from_audio.values())),
                "audio_size": len(audio_data),
                "text_size": len(text_data),
            }, f, indent=2)

    def run_all(self):
        for split in self.dataset:
            self.encode_split(split)
            self.cluster_and_save(split)


if __name__ == "__main__":
    from src.encoder.at_encoder import CLAP_Encoder 

    processor = AudioCapsProcessor(
        at_encoder=CLAP_Encoder(),
        save_dir="data/embeddings/audiocaps",
        batch_size=256,
        text_cluster_k=5,
        audio_cluster_k=5
    )
    processor.run_all()