import sys, os, json, torch, copy
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

from src.model.linear import single_train_test
from src.scar.scar_calculate import SCARcalculation


LABEL_MAPPINGS = {
    "what": {
        "num_classes": 2,
        "mapping": {
            0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1
        }
    },
    "where": {
        "num_classes": 3,
        "mapping": {
            0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 2, 9: 1
        }
    },
    "how": {
        "num_classes": 4,
        "mapping": {
            0: 2, 1: 0, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 3, 9: 0
        }
    },
    "origin": {
        "num_classes": 10,
        "mapping": {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9
        }
    }
}


class CIFAR10ClassifyTest():
    def __init__(self, encoder_type="resnet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        if encoder_type == "resnet":
            self.embed_path = "data/embeddings/cifar10/resnet/cifar10_embeddings.json"
            self.test_embed_path = "data/embeddings/cifar10/resnet/cifar10_test_embeddings.json"
        elif encoder_type == "vit":
            self.embed_path = "data/embeddings/cifar10/vit/cifar10_embeddings.json"
            self.test_embed_path = "data/embeddings/cifar10/vit/cifar10_test_embeddings.json"
        elif encoder_type == "dino":
            self.embed_path = "data/embeddings/cifar10/dino/cifar10_embeddings.json"
            self.test_embed_path = "data/embeddings/cifar10/dino/cifar10_test_embeddings.json"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'resnet', 'vit', or 'dino'.")

    def load_embeddings(self, json_path, ratio=1.1):
        data = []
        with open(json_path, "r") as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

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
    
    def label_mapping(self, y, rough_type):
        if rough_type not in LABEL_MAPPINGS:
            raise ValueError("Unsupported rough type. Choose from 'what', 'where', or 'how'.")
        
        config = LABEL_MAPPINGS[rough_type]
        return config["num_classes"], np.array([config["mapping"][label] for label in y])

    def k_flods_testing_with_label_mapping(self, k_folds, rough_type, sample_ratio=1.0):
        # -------------------- Load Sample Data --------------------
        X, y = self.load_embeddings(self.embed_path, sample_ratio)
        X = torch.tensor(X, dtype=torch.float32)
        num_class, map_y = self.label_mapping(y, rough_type)
        y = torch.tensor(y, dtype=torch.long)
        map_y = torch.tensor(map_y, dtype=torch.long)

        num_samples = X.shape[0]

        # -------------------- Load Test Data --------------------
        X_test, y_test = self.load_embeddings(self.test_embed_path)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        num_class, map_y_test = self.label_mapping(y_test, rough_type)
        y_test = torch.tensor(y_test, dtype=torch.long)
        map_y_test = torch.tensor(map_y_test, dtype=torch.long)

        all_fold_acc = []
        x_logits_results = [0] * len(X)             # te1
        y_positive_results = [-1] * len(X)
        y_labels = [-1] * len(X)
        # -------------------- K-fold Testing --------------------
        kf = StratifiedKFold(n_splits=k_folds, shuffle=False)    
        folds = list(kf.split(X, map_y))   
        fold_bar = tqdm(enumerate(folds), total=k_folds, ascii=True)
        for fold, (train_idx, val_idx) in fold_bar:
            now = datetime.now().strftime("%H:%M:%S")
            fold_bar.set_description(f"[{now}] Fold {fold+1}/{k_folds} | Task: {rough_type}")

            X_train, X_val = X[train_idx], X[val_idx]
            map_y_train, map_y_val = map_y[train_idx], map_y[val_idx]

            acc, logits_cpu, preds_cpu, labels_cpu = single_train_test(self.device, X_train, map_y_train, X_val, map_y_val, num_class)

            for idx, logits, pred, label in zip(val_idx, logits_cpu, preds_cpu, labels_cpu):
                x_logits_results[idx] = logits
                y_positive_results[idx] = (pred == label)
                y_labels[idx] = label

            all_fold_acc.append(acc)

        # -------------------- K-fold Testing --------------------
        testing_acc, logits_cpu, preds_cpu, labels_cpu = single_train_test(self.device, X, y, X_test, y_test, 10)

        return num_samples, all_fold_acc, testing_acc, x_logits_results, y_positive_results, y_labels
    

    def average_nested_dicts(self, dict_list):
        avg_dict = copy.deepcopy(dict_list[0])
        
        def recursive_add(d_total, d_new):
            for key in d_new:
                if isinstance(d_new[key], dict):
                    recursive_add(d_total[key], d_new[key])
                else:
                    d_total[key] += d_new[key]

        def recursive_zero(d):
            for key in d:
                if isinstance(d[key], dict):
                    recursive_zero(d[key])
                else:
                    d[key] = 0.0

        def recursive_average(d_total, n):
            for key in d_total:
                if isinstance(d_total[key], dict):
                    recursive_average(d_total[key], n)
                else:
                    d_total[key] /= n

        avg_dict = copy.deepcopy(dict_list[0])
        recursive_zero(avg_dict)
        for d in dict_list:
            recursive_add(avg_dict, d)

        recursive_average(avg_dict, len(dict_list))

        return avg_dict



def main(encoder_type, k_folds=5):
    calculation = SCARcalculation()
    classifier = CIFAR10ClassifyTest(encoder_type)

    save_path = "test/results/scar_experiment_results_pac.txt"

    # for ratio in [2]:         # test
    for ratio in [1, 2, 5, 10, 20, 50]:
        indexes = []
        type_kfold_accs = defaultdict(list)
        type_test_accs = defaultdict(list)
        avg_false_nums, false_all_nums = [], []
        for cur in range(ratio): 
            type_results = {}
            for rough_type in ["what", "where", "how", "origin"]:
                print(f"\n=== Rough Type: {rough_type} with {encoder_type} at ratio={1/ratio} in {cur+1}/{ratio} ===")
                num_samples, all_fold_acc, testing_acc, x_logits_results, y_positive_results, y_labels = classifier.k_flods_testing_with_label_mapping(k_folds=k_folds, rough_type=rough_type, sample_ratio=1/ratio)
                type_results[rough_type] = [x_logits_results, y_positive_results, y_labels]

                # —————————————— Show Results ——————————————————
                avg_acc = sum(all_fold_acc) / k_folds
                type_kfold_accs[rough_type].append(avg_acc)
                type_test_accs[rough_type].append(testing_acc)

            # —————————————— False Calculation ——————————————————            
            false_records_arr = np.array([type_results[key][1] for key in type_results])  # 对应 y_positive_results
            zero_counts_per_row = np.sum(false_records_arr == 0, axis=1)
            avg_false_num = np.mean(zero_counts_per_row)
            zero_counts_per_column = np.all(false_records_arr == 0, axis=0)
            all_false_num = np.sum(zero_counts_per_column)
            # print("Average False Indices Numbers:", avg_false_num, "All False Indices Numbers:", all_false_num)
            avg_false_nums.append(avg_false_num)
            false_all_nums.append(all_false_num)
            
            # —————————————— SCAR Calculation ——————————————————
            scar_index = calculation.calculation(type_results, num_samples, ratio)    # 确定一个下采样比例，在单次实验中所有 rough_type 下的 SCAR 指标。
            indexes.append(scar_index)

        # —————————————— Results Calculation ——————————————————
        avg_false_sample_num = np.mean(avg_false_nums)
        avg_false_total_sample_num = np.mean(false_all_nums)

        avg_index_results = classifier.average_nested_dicts(indexes)

        avg_fold_acc_what = np.mean(type_kfold_accs["what"])
        avg_fold_acc_where = np.mean(type_kfold_accs["where"])
        avg_fold_acc_how = np.mean(type_kfold_accs["how"])
        avg_fold_acc_origin = np.mean(type_kfold_accs["origin"])

        avg_test_acc_what = np.mean(type_test_accs["what"])
        avg_test_acc_where = np.mean(type_test_accs["where"])
        avg_test_acc_how = np.mean(type_test_accs["how"])
        avg_test_acc_origin = np.mean(type_test_accs["origin"])
        # —————————————— Writing Results ——————————————————
        with open(save_path, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write("=" * 60 + "\n")
            f.write(f"[Timestamp]      {current_time}\n")
            f.write(f"[Encoder]        {encoder_type}\n")
            f.write(f"[Ratio]          1/{ratio:.0f} = {1/ratio:.2f}\n")
            
            f.write(f"\n[False Sample Statistics]\n")
            f.write(f"  - Avg False Sample per Fold     : {avg_false_sample_num:.4f}\n")
            f.write(f"  - Avg False Sample (Total)      : {avg_false_total_sample_num:.4f}\n")

            f.write(f"\n[Average SCAR Index per Fold]\n")
            f.write(f"{avg_index_results}\n")

            f.write(f"\n[Average K-Fold Accuracies]\n")
            f.write(f"  - WHAT                           : {avg_fold_acc_what:.4f}\n")
            f.write(f"  - WHERE                          : {avg_fold_acc_where:.4f}\n")
            f.write(f"  - HOW                            : {avg_fold_acc_how:.4f}\n")
            f.write(f"  - Origin                         : {avg_fold_acc_origin:.4f}\n")

            f.write(f"\n[Average Test Accuracies]\n")
            f.write(f"  - WHAT                           : {avg_test_acc_what:.4f}\n")
            f.write(f"  - WHERE                          : {avg_test_acc_where:.4f}\n")
            f.write(f"  - HOW                            : {avg_test_acc_how:.4f}\n")
            f.write(f"  - Origin                         : {avg_test_acc_origin:.4f}\n")

    # —————————————— Completion Experiments ——————————————————
    with open(save_path, "a") as f:
        f.write("=" * 60 + "\n")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Experiment completed at {current_time}\n")
        f.write("=" * 60 + "\n")





if __name__ == "__main__":
    # main("vit", k_folds=5)
    for encoder_type in ["resnet", "vit", "dino"]:
        main(encoder_type, k_folds=5)

    # python3 test/cifar10_rough_classify_pac.py