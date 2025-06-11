import sys, os, json, torch, copy, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor

from src.model.linear import single_train_test
from src.scar.scar_calculate import SCARcalculation




LABEL_INFO_DICT = {
    "origin": {
        "label_mapping": None,
        "num_class": 1000,
        "label_type": "origin"
    },
}



class Imagenet1KClassifyTest():
    def __init__(self, encoder_type="resnet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        if encoder_type == "resnet":
            self.embed_path = "data/embeddings/imagenet/resnet/train"
            self.test_embed_path = "data/embeddings/imagenet/resnet/val"
        elif encoder_type == "vit":
            self.embed_path = "data/embeddings/imagenet/vit/train"
            self.test_embed_path = "data/embeddings/imagenet/vit/val"
        elif encoder_type == "dino":
            self.embed_path = "data/embeddings/imagenet/dino/train"
            self.test_embed_path = "data/embeddings/imagenet/dino/val"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'resnet', 'vit', or 'dino'.")
        
    def load_json_file(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    def load_embeddings(self, path):
        # train_batches = [os.path.join(path, fname) for fname in os.listdir(path)][:8]   # NOTE: test
        train_batches = [os.path.join(path, fname) for fname in os.listdir(path)]
        data = []
        with ThreadPoolExecutor() as executor:
            for batch_data in executor.map(self.load_json_file, train_batches):
                data.extend(batch_data)

        print(f"Loaded Data from {path}") 
        return data
    
    def split_prmary_reserve(self, data, split_ratio=1, seed=42):
        label_to_items = defaultdict(list)
        for item in tqdm(data, desc="Split data with label", ascii=False):  # 监控遍历
            label_to_items[item['label']].append(item)

        primary_set = []
        reserve_set = defaultdict(list)
        rng = np.random.default_rng(seed)

        # 每个 label 的数据划分
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
    
    def k_flods_testing_with_label_mapping(self, dataset, testset, k_folds, label_type, sample_ratio=1.0, test_only=False):
        if label_type not in LABEL_INFO_DICT:
            raise ValueError(f"Unsupported label type: {label_type}. Available types: {list(LABEL_INFO_DICT.keys())}")
        label_info = LABEL_INFO_DICT[label_type]
        num_class = label_info["num_class"]
        # label_mapping = label_info["label_mapping"]
        
        # -------------------- Load Sample Data --------------------
        X, y = self.downsample_embeddings(dataset, sample_ratio)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        num_samples = X.shape[0]

        # -------------------- Load Test Data --------------------
        X_test, y_test = self.downsample_embeddings(testset)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        all_fold_acc = []
        x_logits_results = [0] * num_samples          
        y_positive_results = [-1] * num_samples
        y_labels = [-1] * num_samples

        if not test_only:
            # -------------------- K-fold Testing --------------------
            kf = StratifiedKFold(n_splits=k_folds, shuffle=False)    
            folds = list(kf.split(X, y))   
            fold_bar = tqdm(enumerate(folds), total=k_folds, ascii=True)
            for fold, (train_idx, val_idx) in fold_bar:
                now = datetime.now().strftime("%H:%M:%S")
                fold_bar.set_description(f"[{now}] Fold {fold+1}/{k_folds} | Task: {label_type}")

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                acc, logits_cpu, preds_cpu, labels_cpu = single_train_test(self.device, X_train, y_train, X_val, y_val, num_class)

                for idx, logits, pred, label in zip(val_idx, logits_cpu, preds_cpu, labels_cpu):
                    x_logits_results[idx] = logits
                    y_positive_results[idx] = (pred == label)
                    y_labels[idx] = label

                all_fold_acc.append(acc)

            # -------------------- K-fold Testing --------------------
        testing_acc, _, _, _ = single_train_test(self.device, X, y, X_test, y_test, num_class)
        return num_samples, all_fold_acc, testing_acc, x_logits_results, y_positive_results, y_labels
    
    

def main(encoder_type, label_type, k_folds=5):
    calculation = SCARcalculation()
    classifier = Imagenet1KClassifyTest(encoder_type)

    # =========================== Total Set Testing ===========================
    total_set = classifier.load_embeddings(classifier.embed_path)
    test_set = classifier.load_embeddings(classifier.test_embed_path)
    kfold_results = classifier.k_flods_testing_with_label_mapping(total_set, test_set, k_folds=k_folds, label_type=label_type, sample_ratio=1, test_only=True)    # NOTE: 这里取了第一个 label type
    _, _, total_test_acc, _, _, _ = kfold_results
    print(f"Total dataset test acc: {total_test_acc}")

    # =========================== Primary Set Test ===========================
    primary_set, reserve_set = classifier.split_prmary_reserve(total_set, split_ratio=0.8)   # 这里我们先测 primary_set 的数据，reserve_set 作为后续补充的内容。
    data_size, rdata_size = len(primary_set), sum([len(reserve_set[key]) for key in reserve_set])
    print(f"Primary Set Size: {data_size}, Reserve Set Size: {rdata_size}")

    indexes_with_ratio = {}
    for ratio in [1, 2, 5]:         # NOTE: test
    # for ratio in [1, 2, 5, 10, 15, 20, 30, 40, 50]:
        indexes = []
        type_kfold_accs = defaultdict(list)
        type_test_accs = defaultdict(list)
        avg_false_nums, false_all_nums = [], []
        for cur in range(ratio): 
            type_results = {}

            print(f"\n=== Rough Type: {label_type} with {encoder_type} at ratio={1/ratio} in {cur+1}/{ratio} ===")
            kfold_results = classifier.k_flods_testing_with_label_mapping(primary_set, test_set, k_folds=k_folds, label_type=label_type, sample_ratio=1/ratio)

            num_samples, all_fold_acc, testing_acc, x_logits_results, y_positive_results, y_labels = kfold_results
            type_results[label_type] = [x_logits_results, y_positive_results, y_labels]

            # —————————————— Show Results ——————————————————
            avg_acc = sum(all_fold_acc) / k_folds
            type_kfold_accs[label_type].append(avg_acc)
            type_test_accs[label_type].append(testing_acc)

            # —————————————— False Calculation ——————————————————            
            false_records_arr = np.array([type_results[key][1] for key in type_results])  # 对应 y_positive_results
            zero_counts_per_row = np.sum(false_records_arr == 0, axis=1)
            avg_false_num = np.mean(zero_counts_per_row)
            zero_counts_per_column = np.all(false_records_arr == 0, axis=0)
            all_false_num = np.sum(zero_counts_per_column)
            avg_false_nums.append(avg_false_num)
            false_all_nums.append(all_false_num)
            
            # —————————————— SCAR Calculation ——————————————————
            scar_index = calculation.calculation(type_results, num_samples, ratio)    # 确定一个下采样比例，在单次实验中所有 rough_type 下的 SCAR 指标。
            indexes.append(scar_index)

        scars = defaultdict(dict)
        for i in indexes[0][label_type]['hs_scar']:
            for key in ["scale", 'coverage', 'authenticity', 'richness']:
                task_index = np.mean([item[label_type]['hs_scar'][i][key] for item in indexes])
                scars[i][key] = task_index

        for key in ["scale", 'coverage', 'authenticity', 'richness']:
            total_index = np.mean([item[label_type]['task_scar'][key] for item in indexes])
            scars["total"][key] = total_index

        indexes_with_ratio[1/ratio] = scars
    
    # —————————————— Foundation data size Estimation ——————————————————
    ratios = [key for key in indexes_with_ratio]
    tar_res = {}
    for tar in indexes_with_ratio[1]:
        if tar == 'total':
            continue
        indexes = [[indexes_with_ratio[key][tar][index] for key in ratios] for index in ["scale", 'coverage', 'authenticity', 'richness']]
        tar_h, tar_size = calculation.predict_foudation(ratios, indexes)
        tar_res[tar] = (tar_h, tar_size)

    total_indexes = [[indexes_with_ratio[key]['total'][index] for key in ratios] for index in ["scale", 'coverage', 'authenticity', 'richness']]
    total_h, total_size = calculation.predict_foudation(ratios, total_indexes)

    # for key, (tar_h, tar_size) in tar_res.items():
    #     print(f"Estimated Foundation Data Size for {key}: {tar_size:.2f} (H: {tar_h:.4f})")
    # print(f"Estimated Total Foundation Data Size: {total_size:.2f} (H: {total_h:.4f})")

    # —————————————— Fill data with Foundation data size from reserve sample ——————————————————
    fill_size = total_size - data_size
    subtype_count = Counter([item['label'] for item in primary_set])

    subtype_fill_size = defaultdict(int)
    for label in indexes_with_ratio[1]:
        if tar == 'total':
            continue
        subtype_fill_size[label] = max(0, tar_res[label][1] - subtype_count[label])

    total = sum(subtype_fill_size.values())
    norm_subtype_fill_size = {k: v / total for k, v in subtype_fill_size.items()}

    available_reserve_size = min([fill_size] + [len(reserve_set[key] / norm_subtype_fill_size[key]) for key in subtype_fill_size])
    print(f"Available Reserve Size: {available_reserve_size}, Fill Size: {fill_size}")

    extend_set = copy.deepcopy(primary_set)
    for label in subtype_fill_size:
        add_data_size = int(available_reserve_size * norm_subtype_fill_size[label])
        add_data = random.sample(reserve_set[label], add_data_size)
        extend_set.extend(add_data)
    print(f"Filled Data size is {len(extend_set)}")
    
    # —————————————— Testing with Fill data  ——————————————————
    kfold_results = classifier.k_flods_testing_with_label_mapping(extend_set, test_set, k_folds=k_folds, label_type=label_type, sample_ratio=1, test_only=True)
    num_samples, all_fold_acc, testing_acc, x_logits_results, y_positive_results, y_labels = kfold_results

    # =============== SAVE Average Results for this ratio ===============
    result_save_path = "experiments/results/imagenet_experiment_results.txt"
    with open(result_save_path, 'a') as f:
        avg_kfold_acc = sum(type_kfold_accs[label_type]) / ratio
        avg_test_acc = sum(type_test_accs[label_type]) / ratio

        f.write(f"Label Type: {label_type}\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total data size {data_size+rdata_size}; Primary datasize: {data_size}; Extend datasize: {len(extend_set)}.\n")

        f.write("=" * 70 + "\n")
        f.write(f"  Total SCAR feature: {total_h}, {total_size}\n")
        f.write(f"  Step SCAR feature: {tar_res}\n")

        f.write("=" * 70 + "\n")
        f.write(f"  Avg K-Fold Acc over {ratio} runs: {avg_kfold_acc:.4f}\n")
        f.write(f"  1. Total Data Test Acc over {ratio} runs: {total_test_acc:.4f}\n")
        f.write(f"  2. Avg Test Acc over {ratio} runs: {avg_test_acc:.4f}\n")
        f.write(f"  3. Filled Data Test Acc over {ratio} runs: {testing_acc:.4f}\n")


if __name__ == "__main__":
    encoder_type = "resnet"  # Choose from 'resnet', 'vit', or 'dino'
    label_type = "origin"  # Define the label types you want to test
    k_folds = 5
    for encoder_type in ['resnet', 'vit', 'dino']:
        main(encoder_type, label_type, k_folds)

    # python3 experiments/imagenet1k_classify.py