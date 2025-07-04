import sys, os, json, torch, copy, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from collections import defaultdict, Counter

from src.model.linear import single_train_test
from src.scar.scar_calculate import SCARcalculation



LABEL_INFO_DICT = {
    "origin": {
        "label_mapping": None,
        "num_class": 10,
        "label_type": "origin"
    },
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
        
        self.calculation = SCARcalculation()
        
    def load_embeddings(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        
        print(f"Loaded Data from {path}") 
        return data

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
        if label_type not in LABEL_INFO_DICT:
            raise ValueError(f"Unsupported label type: {label_type}. Available types: {list(LABEL_INFO_DICT.keys())}")
        label_info = LABEL_INFO_DICT[label_type]
        num_class = label_info["num_class"]
        
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

        return num_samples, num_class, testing_acc, train_logits, train_preds, train_labels

    def foundation_size_estimate(self, sample_num, encoder_type, label_type, ratios, train_set, test_set, write_hs=False):
        scar_indexes_with_ratio = {}
        primary_train_acc = 0.0
        for ratio in ratios:  
            indexes = []
            for cur in range(ratio): 
                type_results = {}
                print(f"=== Rough Type: {label_type} with {encoder_type} at ratio={1/ratio} in {cur+1}/{ratio} ===")
                cur_total_results = self.training_testing_with_given_data(train_set, test_set, label_type=label_type, sample_ratio=1/ratio)      
                num_samples, num_class, _, train_logits, train_preds, train_labels = cur_total_results

                type_results[label_type] = [train_logits, train_preds, train_labels]
                scar_index, train_acc = self.calculation.calculation(type_results, num_samples, ratio)
                if ratio == 1:
                    primary_train_acc = train_acc
                indexes.append(scar_index)     

            scars = defaultdict(dict)
            for task_index in indexes[0][label_type]:
                for key in ["scale", 'coverage', 'authenticity', 'richness']:
                    task_mean_index = np.mean([item[label_type][task_index][key] for item in indexes])
                    scars[task_index][key] = task_mean_index

            scar_indexes_with_ratio[ratio] = dict(scars)

        task_res = {}
        for task in scar_indexes_with_ratio[1]:   
            indexes = [[scar_indexes_with_ratio[key][task][index] for key in ratios] 
                    for index in ["scale", 'coverage', 'authenticity', 'richness']]

            task_h, task_fd_size = self.calculation.predict_foudation_step(sample_num, num_class, ratios, indexes)
            task_res[task] = (task_h, task_fd_size)

        step_hs = [v[0] for v in task_res.values()]                         # 所有 step function 所估计的目标建设空间大小

        step_avg_acc = np.mean([scar_indexes_with_ratio[1][task]["authenticity"] for task in scar_indexes_with_ratio[1]])
        ratio = (1 - primary_train_acc) / (1 - step_avg_acc) if step_avg_acc < 1 else np.e          # min value is E

        foundation_size = self.calculation.predict_foundation_total_fast(step_hs, ratio, max_order=3)

        if not foundation_size:
            foundation_size = sum([v[1] for v in task_res.values()])  # Fallback to sum of step sizes if prediction fails

        if write_hs:
            with open(f"src/scar/hs_list_cifar10_{encoder_type}_{write_hs}.json", "w") as f:
                json.dump(ratio, f)
                f.write("\n")
                json.dump(step_hs, f)

        return foundation_size, scar_indexes_with_ratio, task_res



def main(encoder_type, label_type, split_ratio=0.6):
    classifier = CIFAR10ClassifyTest(encoder_type)
    ratios = [1, 2, 5, 10, 15, 20, 30]

    # =========================== 1. Total Set Test ===========================
    total_set = classifier.load_embeddings(classifier.embed_path)
    test_set = classifier.load_embeddings(classifier.test_embed_path)

    total_test_results = classifier.training_testing_with_given_data(total_set, test_set, label_type=label_type, sample_ratio=1)  
    total_sample_num, _, total_test_acc, _, _, _ = total_test_results
    total_foundation_size, total_scar_indexes, _ = classifier.foundation_size_estimate(total_sample_num, encoder_type, label_type, ratios, total_set, test_set, write_hs="total")

    # =========================== 2. Primary Set Test ===========================
    primary_set, reserve_set = classifier.split_prmary_reserve(total_set, split_ratio=split_ratio)
    data_size, rdata_size = len(primary_set), sum([len(reserve_set[key]) for key in reserve_set])
    print(f"Primary Set Size: {data_size}, Reserve Set Size: {rdata_size}")

    primary_test_results = classifier.training_testing_with_given_data(primary_set, test_set, label_type=label_type, sample_ratio=1)  
    primary_sample_num, _, primary_test_acc, _, _, _ = primary_test_results
    primary_foundation_size, scar_indexes_with_ratio, task_res = classifier.foundation_size_estimate(primary_sample_num, encoder_type, label_type, ratios, primary_set, test_set, write_hs="primary")

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
    _, _, extend_testing_acc, _, _, _  = extend_test_results

    # =========================== 3.1 Random Fill data set ===========================
    avaliable_reserve_set = []
    for label in subtype_fill_size:
        if label in reserve_set:
            avaliable_reserve_set.extend(reserve_set[label])

    average_extend_set = copy.deepcopy(primary_set)
    extend_random = random.sample(avaliable_reserve_set, len(extend_set) - len(primary_set))
    average_extend_set.extend(extend_random)

    print(f"Random Filled Data size is {len(average_extend_set)}")
    random_extend_test_results = classifier.training_testing_with_given_data(average_extend_set, test_set, label_type=label_type, sample_ratio=1)     
    _, _, random_extend_testing_acc, _, _, _  = random_extend_test_results

    # =========================== 3.2 Average Fill data set ===========================
    average_extend_set = copy.deepcopy(primary_set)
    add_num = (len(extend_set) - len(primary_set)) / len([label for label in subtype_fill_size if label in reserve_set])
    for label in subtype_fill_size:
        if label in reserve_set:
            add_data = random.sample(reserve_set[label], int(add_num))
            average_extend_set.extend(add_data)
            
    print(f"Average Filled Data size is {len(average_extend_set)}")
    average_extend_test_results = classifier.training_testing_with_given_data(average_extend_set, test_set, label_type=label_type, sample_ratio=1)     
    _, _, average_extend_testing_acc, _, _, _  = average_extend_test_results

    # =============== 4. SAVE Average Results for this ratio ===============
    result_save_path = "experiments/results/cifar10_experiment_results.txt"
    with open(result_save_path, 'a') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  CIFAR10 - Encoder: {encoder_type}; Label Type: {label_type}\n")
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
        f.write(f"  4. Random Extend Data Test Acc over {ratios} runs: {random_extend_testing_acc:.4f}\n")
        f.write(f"  5. Average Extend Data Test Acc over {ratios} runs: {average_extend_testing_acc:.4f}\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total SCAR indexes: {total_scar_indexes}\n")
        f.write("=" * 70 + "\n\n")





if __name__ == "__main__":
    label_type = "origin"  # Define the label types you want to test
    for i in range(3):
        print(f"Running experiments for label type: {label_type} in range {i}.")
        for encoder_type in ['resnet', 'vit', 'dino']:
            main(encoder_type, label_type)

    # main("vit", label_type)


    # python3 experiments/cifar10_classify.py