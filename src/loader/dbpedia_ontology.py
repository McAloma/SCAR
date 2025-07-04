import sys, os, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

from src.encoder.text_encoders import BERT_Encoder, RoBERTa_Encoder, GPT2_Encoder


class DBpediaDataloader():
    def __init__(self, encoder_type="bert"):
        if encoder_type == "bert":
            self.encoder = BERT_Encoder()
            self.train_path = "./data/embeddings/dbpedia/bert/train"
            self.test_path = "./data/embeddings/dbpedia/bert/test"
        elif encoder_type == "roberta":
            self.encoder = RoBERTa_Encoder()
            self.train_path = "./data/embeddings/dbpedia/roberta/train"
            self.test_path = "./data/embeddings/dbpedia/roberta/test"
        elif encoder_type == "gpt2":
            self.encoder = GPT2_Encoder()
            self.train_path = "./data/embeddings/dbpedia/gpt2/train"
            self.test_path = "./data/embeddings/dbpedia/gpt2/test"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'bert', 'roberta', or 'gpt2'.")

        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        self.label_map = {
            0: "Company",
            1: "EducationalInstitution",
            2: "Artist",
            3: "Athlete",
            4: "OfficeHolder",
            5: "MeanOfTransportation",
            6: "Building",
            7: "NaturalPlace",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "WrittenWork"
        }

    def save_to_json(self, data, out_path):
        with open(out_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def encode_and_save(self, split="train", batch_size=1024):
        print(f"Loading DBpedia [{split}] split...")
        dataset = load_dataset("dbpedia_14", split=split)

        texts = dataset["content"]
        labels = dataset["label"]

        total_docs = len(texts)
        print(f"Encoding {total_docs} documents...")

        batch_texts = []
        batch_labels = []
        batch_id = 0
        label_counter = defaultdict(int)

        for idx, (text, label) in enumerate(tqdm(zip(texts, labels), total=total_docs)):
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
                            "text": t[:300],  # optional truncation
                            "label": coarse_label,
                            "embedding": e.tolist()
                        })

                    batch_id += 1
                    out_path = os.path.join(
                        self.train_path if split == "train" else self.test_path,
                        f"embeddings_{batch_id:04d}.json"
                    )
                    self.save_to_json(batch_results, out_path)
                except Exception as e:
                    print(f"[Error] Failed to encode batch {batch_id}: {e}")

                batch_texts = []
                batch_labels = []

        label_stat_path = os.path.join(
            self.train_path if split == "train" else self.test_path,
            f"label_distribution.json"
        )
        self.save_to_json(dict(label_counter), label_stat_path)
        print(f"[Info] {split} label distribution saved to {label_stat_path}")


if __name__ == "__main__":
    for encoder in ["bert", "roberta", "gpt2"]:
        dataloader = DBpediaDataloader(encoder_type=encoder)
        dataloader.encode_and_save(split="train", batch_size=512)
        dataloader.encode_and_save(split="test", batch_size=512)

    
    # python3 src/loader/dbpedia_ontology.py