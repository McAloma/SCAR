import sys, os, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

from src.encoder.text_encoders import BERT_Encoder, RoBERTa_Encoder, GPT2_Encoder


class AGNewsDataloader():
    def __init__(self, encoder_type="bert"):
        if encoder_type == "bert":
            self.encoder = BERT_Encoder()
            self.train_path = "./data/embeddings/agnews/bert/train"
            self.test_path = "./data/embeddings/agnews/bert/test"
        elif encoder_type == "roberta":
            self.encoder = RoBERTa_Encoder()
            self.train_path = "./data/embeddings/agnews/roberta/train"
            self.test_path = "./data/embeddings/agnews/roberta/test"
        elif encoder_type == "gpt2":
            self.encoder = GPT2_Encoder()
            self.train_path = "./data/embeddings/agnews/gpt2/train"
            self.test_path = "./data/embeddings/agnews/gpt2/test"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'bert', 'roberta', or 'gpt2'.")
        
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        self.label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }

    def save_to_json(self, data, out_path):
        with open(out_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def encode_and_save(self, split="train", batch_size=1024):
        print(f"Loading AG News [{split}] split...")
        dataset = load_dataset("ag_news", split=split)

        titles = dataset["text"]
        labels = dataset["label"]

        total_docs = len(titles)
        print(f"Encoding {total_docs} documents...")

        batch_texts = []
        batch_labels = []
        batch_id = 0
        label_counter = defaultdict(int)

        for idx, (text, label) in enumerate(tqdm(zip(titles, labels), total=total_docs)):
            batch_texts.append(text)
            batch_labels.append(label)

            if len(batch_texts) >= batch_size or idx == total_docs - 1:
                try:
                    embeddings = self.encoder.encode_batch(batch_texts)
                    batch_results = []
                    for t, l, e in zip(batch_texts, batch_labels, embeddings):
                        coarse_label = self.label_map[l]
                        label_counter[coarse_label] += 1
                        batch_results.append({
                            "title": t[:100],  # optional truncation
                            "label": coarse_label,
                            "embedding": e.tolist()
                        })

                    batch_id += 1
                    if split == "train":
                        out_path = os.path.join(self.train_path, f"embeddings_{batch_id:04d}.json")
                    else:
                        out_path = os.path.join(self.test_path, f"embeddings_{batch_id:04d}.json")
                    self.save_to_json(batch_results, out_path)
                except Exception as e:
                    print(f"[Error] Failed to encode batch {batch_id}: {e}")

                batch_texts = []
                batch_labels = []

        if split == "train":
            label_stat_path = os.path.join(self.train_path, f"label_distribution.json")
        else:
            label_stat_path = os.path.join(self.test_path, f"label_distribution.json")
        self.save_to_json(dict(label_counter), label_stat_path)
        print(f"[Info] {split} label distribution saved to {label_stat_path}")


if __name__ == "__main__":
    # encoder_type = "gpt2"  # 可改为 'bert' 或 'roberta'
    for encoder in ["bert", "roberta", "gpt2"]:
        dataloader = AGNewsDataloader(encoder_type=encoder)

        # 分别处理 train 和 test
        dataloader.encode_and_save(split="train", batch_size=512)
        dataloader.encode_and_save(split="test", batch_size=512)