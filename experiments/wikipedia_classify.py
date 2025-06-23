import sys, os, json, torch, copy, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.model.linear import single_train_test
from src.scar.scar_calculate import SCARcalculation

# wiki sample num 18315148


class WikipediaClassifyTest():
    def __init__(self, encoder_type="bert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        if encoder_type == "bert":
            self.embed_path = "data/embeddings/wikipedia/bert"
        elif encoder_type == "roberta":
            self.embed_path = "data/embeddings/wikipedia/roberta"
        elif encoder_type == "gpt2":
            self.embed_path = "data/embeddings/wikipedia/gpt2"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'bert', 'roberta', or 'gpt2'.")
        
    def load_json_file(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    def load_embeddings(self, path):
        data_files = [f for f in os.listdir(path) if f.endswith(".json") and f != "label_distribution.json"][:4]    # NOTE: test
        # data_files = [f for f in os.listdir(path) if f.endswith(".json") and f != "label_distribution.json"]
        data = []

        with ThreadPoolExecutor() as executor:
            for batch in executor.map(lambda f: self.load_json_file(os.path.join(path, f)), data_files):
                for item in batch:
                    if "Unknown" not in item["label"]:
                        data.append(item)

        print(f"Loaded {len(data)} samples (excluding 'Unknown') from {path}")
        return data
    
    def build_multi_hot_dataset(self, data):
        all_labels = sorted({label for item in data for label in item["label"]})
        label2idx = {label: idx for idx, label in enumerate(all_labels)}
        self.label2idx = label2idx
        num_labels = len(label2idx)

        samples = []

        for item in data:
            emb = item["embedding"]
            labels = item["label"]

            multi_hot = np.zeros(num_labels, dtype=np.float32)
            for label in labels:
                multi_hot[label2idx[label]] = 1.0

            samples.append({
                "embedding": emb,
                "label": multi_hot
            })

        print(f"Constructed {len(samples)} samples with {num_labels} multi-hot classes.")
        return samples, label2idx
    
    def split_dataset(self, samples, train_ratio=0.8, seed=42):
        labels = np.array([sample["label"] for sample in samples])
        indices = np.arange(len(samples))
        n_splits = int(1 / (1 - train_ratio))

        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for train_idx, val_idx in mskf.split(indices, labels):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            print(f"Stratified split: {len(train_samples)} train / {len(val_samples)} val")
            return train_samples, val_samples
    
    def split_prmary_reserve(self, data, split_ratio=1, seed=42):
        label_to_items = defaultdict(list)
        for item in data:
            label_to_items[item['label']].append(item)

        primary_set = []
        reserve_set = defaultdict(list)
        rng = np.random.default_rng(seed)

        for label, items in label_to_items.items():
            n_total = len(items)
            n_primary = int(n_total * split_ratio)

            indices = rng.permutation(n_total)
            primary_set.extend([items[i] for i in indices[:n_primary]])
            reserve_set[label].extend([items[i] for i in indices[n_primary:]])

        return primary_set, reserve_set

    def downsample_embeddings(self, data, ratio=1.0):
        if ratio < 1.0:
            label_to_items = defaultdict(list)
            for item in data:
                label_to_items[item['label']].append(item)

            sampled_data = []
            for label, items in label_to_items.items():
                n_samples = int(len(items) * ratio)
                indices = np.random.choice(len(items), n_samples, replace=False)
                sampled_items = [items[i] for i in indices]
                sampled_data.extend(sampled_items)

            data = sampled_data 

        X = np.array([item['embedding'] for item in data])  # (N, D)
        y = np.array([item['label'] for item in data])      # (N,)
        return X, y
    
    def training_testing_with_given_data(self, dataset, testset, label_type, sample_ratio=1.0):
        label_path = os.path.join(self.embed_path, "label_distribution.json")
        with open(label_path, "r") as f:
            label_dict = json.load(f)

        labels = label_dict.keys()
        num_class = len(labels)
        
        # -------------------- Load Sample Data --------------------
        X, y = self.downsample_embeddings(dataset, sample_ratio)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        num_samples = X.shape[0]

        # -------------------- Load Test Data --------------------
        X_test, y_test = self.downsample_embeddings(testset)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # -------------------- EMR process with linear model --------------------
        testing_acc, train_logits, train_preds, train_labels = single_train_test(self.device, X, y, X_test, y_test, num_class)
        # NOTE: testing_acc, logits_cpu, preds_cpu, labels_cpu = float, (n, k) (n,) (n,)

        return num_samples, testing_acc, train_logits, train_preds, train_labels


def main(encoder_type, label_type):
    calculation = SCARcalculation()
    classifier = WikipediaClassifyTest(encoder_type)
    ratios = [1, 2, 5, 10, 15, 20, 30, 40, 50]

    # =========================== 1. Total Set Test ===========================
    raw_data = classifier.load_embeddings(classifier.embed_path)
    samples, label2idx = classifier.build_multi_hot_dataset(raw_data)
    total_set, test_set = classifier.split_dataset(samples)

    total_test_results = classifier.training_testing_with_given_data(total_set, test_set, label_type=label_type, sample_ratio=1)  
    _, total_test_acc, _, _, _ = total_test_results
    total_foundation_size, _, _ = classifier.foundation_size_estimate(calculation, label_type, ratios, total_set, test_set)

    # =========================== 2. Primary Set Test ===========================
    primary_set, reserve_set = classifier.split_prmary_reserve(total_set, split_ratio=0.6)
    data_size, rdata_size = len(primary_set), sum([len(reserve_set[key]) for key in reserve_set])
    print(f"Primary Set Size: {data_size}, Reserve Set Size: {rdata_size}")

    primary_test_results = classifier.training_testing_with_given_data(primary_set, test_set, label_type=label_type, sample_ratio=1)  
    _, primary_test_acc, _, _, _ = primary_test_results
    primary_foundation_size, scar_indexes_with_ratio, task_res = classifier.foundation_size_estimate(calculation, label_type, ratios, primary_set, test_set, write_hs=True)

    # =========================== 3. Fill data set ===========================
    fill_size = max(0, primary_foundation_size - data_size)                     # keep positive
    subtype_fill_size = defaultdict(int)
    subtype_count = Counter([item['label'] for item in primary_set])
    for tar in scar_indexes_with_ratio[1]:
        subtype_fill_size[tar] = max(0, task_res[tar][1] - subtype_count[tar])

    total = sum(subtype_fill_size.values())
    norm_subtype_fill_size = {k: v / total for k, v in subtype_fill_size.items()}

    available_reserve_sizes = []
    for key in norm_subtype_fill_size:
        if (key in reserve_set) and norm_subtype_fill_size[key] > 0:
            num = min(len(reserve_set[key]), task_res[key][1] - subtype_count[key]) / norm_subtype_fill_size[key] 
            available_reserve_sizes.append(num)

    available_reserve_size = min([fill_size] +  available_reserve_sizes)
    print(f"Available Reserve Size: {available_reserve_size}, Fill Size: {fill_size}.")

    extend_set = copy.deepcopy(primary_set)
    for label in subtype_fill_size:
        if label in reserve_set:
            add_data_size = int(available_reserve_size * norm_subtype_fill_size[label])
            add_data = random.sample(reserve_set[label], add_data_size)
            extend_set.extend(add_data)
    print(f"Filled Data size is {len(extend_set)}")
    
    extend_test_results = classifier.training_testing_with_given_data(extend_set, test_set, label_type=label_type, sample_ratio=1)     
    _, extend_testing_acc, _, _, _  = extend_test_results

    # =============== 4. SAVE Average Results for this ratio ===============
    result_save_path = "experiments/results/imagenet_experiment_results.txt"
    with open(result_save_path, 'a') as f:
        f.write("=" * 70 + "\n")
        f.write(f"\n\n  Encoder: {encoder_type}; Label Type: {label_type}\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total data size {data_size+rdata_size}; Primary datasize: {data_size}; Extend datasize: {len(extend_set)}.\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total Foundation data size estimation: {total_foundation_size}\n")
        f.write(f"  Primary Foundation data size estimation: {primary_foundation_size}\n")
        f.write("=" * 70 + "\n")
        f.write(f"  1. Total Data Test Acc over {ratios} runs: {total_test_acc:.4f}\n")
        f.write(f"  2. Primary Data Test Acc over {ratios} runs: {primary_test_acc:.4f}\n")
        f.write(f"  3. Extend Data Test Acc over {ratios} runs: {extend_testing_acc:.4f}\n")
        f.write("=" * 70 + "\n")
