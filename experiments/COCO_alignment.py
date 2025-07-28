import sys, os, json, torch, copy, random, time, argparse
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

from src.model.linear import single_train_test
from src.scar.scar_calculate import SCARcalculation






class COCOCaptionMultiModalTest():
    def __init__(self, encoder_type="clip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        if encoder_type:
            self.image_embed_path = f"data/embeddings/coco_caption/{encoder_type}/image/train"
            self.image_test_embed_path = f"data/embeddings/coco_caption/{encoder_type}/image/val"
            self.text_embed_path = f"data/embeddings/coco_caption/{encoder_type}/text/train"
            self.text_test_embed_path = f"data/embeddings/coco_caption/{encoder_type}/text/val"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'clip', 'siglip', or 'coca'.")
        
        self.calculation = SCARcalculation()

    def load_json_file(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)
        
    def load_embeddings(self, path):
        # train_batches = [os.path.join(path, fname) for fname in os.listdir(path)][:20]   # NOTE: test
        train_batches = [os.path.join(path, fname) for fname in os.listdir(path)]
        data = []
        with ThreadPoolExecutor() as executor:
            for batch_data in executor.map(self.load_json_file, train_batches):
                data.extend(batch_data)

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
    
    def training_testing_with_given_data(self, dataset, testset, label_type, num_class, sample_ratio=1.0):
        X, y = self.downsample_embeddings(dataset, sample_ratio)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        num_samples = X.shape[0]
        
        X_test, y_test = self.downsample_embeddings(testset)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        testing_acc, train_logits, train_preds, train_labels = single_train_test(self.device, X, y, X_test, y_test, num_class)
        # NOTE: testing_acc, logits_cpu, preds_cpu, labels_cpu = float, (n, k) (n,) (n,)

        return num_samples, num_class, testing_acc, train_logits, train_preds, train_labels
    
    def foundation_size_estimate(self, sample_num, encoder_type, label_type, ratios, train_set, test_set, num_class, write_hs=False):
        scar_indexes_with_ratio = {}
        primary_train_acc = 0.0
        for ratio in ratios:  
            indexes = []
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Rough Type: {label_type} with {encoder_type} at ratio={1/ratio} ===")
            for _ in range(ratio): 
                type_results = {}
                cur_total_results = self.training_testing_with_given_data(train_set, test_set, label_type, num_class, sample_ratio=1/ratio)      
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
        ratio = (1 - primary_train_acc) / (1 - step_avg_acc) if step_avg_acc < 1 else 0
        ratio = max(ratio, np.e)  # Ensure ratio is at least e to avoid division by zero

        foundation_size = self.calculation.predict_foundation_total_fast(step_hs, ratio, max_order=3)

        if not foundation_size:
            foundation_size = sum([v[1] for v in task_res.values()])  # Fallback to sum of step sizes if prediction fails

        if write_hs:
            with open(f"src/scar/hs_list_COCOcaption_{encoder_type}_{write_hs}.json", "w") as f:
                json.dump(ratio, f)
                f.write("\n")
                json.dump(step_hs, f)

        return foundation_size, scar_indexes_with_ratio, task_res
    

def single_modal_main(modal_type, encoder_type, label_type):
    print(f"\nProcessing COCO_Caption dataset with encoder {encoder_type} in {modal_type} modal.")
    classifier = COCOCaptionMultiModalTest(encoder_type)
    num_class = 50
    # ratios = [1, 2, 5]      # NOTE: test
    ratios = [1, 2, 5, 10, 15, 20, 30]

    # =========================== 1. Total Set Test ===========================
    if modal_type == "image":
        total_set = classifier.load_embeddings(classifier.image_embed_path)
        test_set = classifier.load_embeddings(classifier.image_test_embed_path)
    elif modal_type == "text":
        total_set = classifier.load_embeddings(classifier.text_embed_path)
        test_set = classifier.load_embeddings(classifier.text_test_embed_path)
    else:
        raise ValueError("Unsupported modal type. Choose from 'image' or 'dino'.")

    total_test_results = classifier.training_testing_with_given_data(total_set, test_set, label_type, num_class, sample_ratio=1)  
    total_sample_num, _, total_test_acc, _, _, _ = total_test_results
    total_foundation_size, total_scar_indexes, _ = classifier.foundation_size_estimate(total_sample_num, encoder_type, label_type, ratios, total_set, test_set, num_class, write_hs="Total")

    # =========================== 2. Primary Set Test ===========================
    primary_set, reserve_set = classifier.split_prmary_reserve(total_set, split_ratio=0.6)
    data_size, rdata_size = len(primary_set), sum([len(reserve_set[key]) for key in reserve_set])
    print(f"Primary Set Size: {data_size}, Reserve Set Size: {rdata_size}")

    primary_test_results = classifier.training_testing_with_given_data(primary_set, test_set, label_type, num_class, sample_ratio=1)  
    primary_sample_num, _, primary_test_acc, _, _, _ = primary_test_results
    primary_foundation_size, scar_indexes_with_ratio, task_res = classifier.foundation_size_estimate(primary_sample_num, encoder_type, label_type, ratios, primary_set, test_set, num_class, write_hs="Primary")
    print(f"Primary Foundatoin Set Size: {primary_foundation_size}")

    # =========================== 3. Fill data set ===========================
    fill_size = rdata_size                    # keep fill
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
    # print(f"Normalized Subtype Fill Size: {norm_subtype_fill_size}")

    extend_set = copy.deepcopy(primary_set)
    for label in subtype_fill_size:
        if label in reserve_set:
            add_data_size = int(available_reserve_size * norm_subtype_fill_size[label])
            add_data = random.sample(reserve_set[label], add_data_size)
            extend_set.extend(add_data)
    extend_size = len(extend_set)
    print(f"Filled Data size is {extend_size}")
    
    extend_test_results = classifier.training_testing_with_given_data(extend_set, test_set, label_type, num_class, sample_ratio=1)     
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
    random_extend_test_results = classifier.training_testing_with_given_data(average_extend_set, test_set, label_type, num_class, sample_ratio=1)     
    _, _, random_extend_testing_acc, _, _, _  = random_extend_test_results

    # =========================== 3.2 Average Fill data set ===========================
    average_extend_set = copy.deepcopy(primary_set)
    add_num = (len(extend_set) - len(primary_set)) / len([label for label in subtype_fill_size if label in reserve_set])
    for label in subtype_fill_size:
        if label in reserve_set:
            add_data = random.sample(reserve_set[label], int(add_num))
            average_extend_set.extend(add_data)
            
    print(f"Average Filled Data size is {len(average_extend_set)}")
    average_extend_test_results = classifier.training_testing_with_given_data(average_extend_set, test_set, label_type, num_class, sample_ratio=1)     
    _, _, average_extend_testing_acc, _, _, _  = average_extend_test_results

    size_results = {
        "primary": data_size,
        "reserve": rdata_size,
        "extend": extend_size,
        "total_foundation": total_foundation_size,
        "primary_foundation": primary_foundation_size
    }

    acc_results = {
        "total": total_test_acc,
        "primary": primary_test_acc,
        "extent": extend_testing_acc,
        "random": random_extend_testing_acc,
        "average": average_extend_testing_acc,
    }

    return size_results, acc_results


def main(encoder_type, label_type):
    text_size_results, text_acc_results = single_modal_main("text", encoder_type, label_type)
    image_size_results, image_acc_results = single_modal_main("image", encoder_type, label_type)

    result_save_path = f"experiments/results/COCO_Caption_experiment_results.txt"
    with open(result_save_path, 'a') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  COCO_Caption - Encoder: {encoder_type}; Label Type: {label_type}\n")
        f.write("=" * 70 + "\n")

        # TEXT MODALITY
        f.write(f"  [TEXT MODALITY]\n")
        f.write(f"  Total data size {text_size_results['primary'] + text_size_results['reserve']}; "
                f"Primary datasize: {text_size_results['primary']}; "
                f"Extend datasize: {text_size_results['extend']}.\n")
        f.write(f"  Total Foundation data size estimation: {text_size_results['total_foundation']}\n")
        f.write(f"  Primary Foundation data size estimation: {text_size_results['primary_foundation']}\n")
        f.write(f"  1. Total Data Test Acc: {text_acc_results['total']:.4f}\n")
        f.write(f"  2. Primary Data Test Acc: {text_acc_results['primary']:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"  3. Extend Data Test Acc: {text_acc_results['extent']:.4f}\n")
        f.write(f"  4. Random Extend Data Test Acc: {text_acc_results['random']:.4f}\n")
        f.write(f"  5. Average Extend Data Test Acc: {text_acc_results['average']:.4f}\n")
        f.write("=" * 70 + "\n")

        # IMAGE MODALITY
        f.write(f"  [IMAGE MODALITY]\n")
        f.write(f"  Total data size {image_size_results['primary'] + image_size_results['reserve']}; "
                f"Primary datasize: {image_size_results['primary']}; "
                f"Extend datasize: {image_size_results['extend']}.\n")
        f.write(f"  Total Foundation data size estimation: {image_size_results['total_foundation']}\n")
        f.write(f"  Primary Foundation data size estimation: {image_size_results['primary_foundation']}\n")
        f.write(f"  1. Total Data Test Acc: {image_acc_results['total']:.4f}\n")
        f.write(f"  2. Primary Data Test Acc: {image_acc_results['primary']:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"  3. Extend Data Test Acc: {image_acc_results['extent']:.4f}\n")
        f.write(f"  4. Random Extend Data Test Acc: {image_acc_results['random']:.4f}\n")
        f.write(f"  5. Average Extend Data Test Acc: {image_acc_results['average']:.4f}\n")
        f.write("=" * 70 + "\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COCO Caption Alignment experiments.")
    parser.add_argument("--encoder", type=str, default="clip", choices=["clip", "siglip", "coca"])
    parser.add_argument("--label", type=str, default="origin")

    args = parser.parse_args()
    main(args.encoder, args.label)

    # python3 experiments/COCO_alignment.py --encoder clip --label origin