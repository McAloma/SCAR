import sys, os, json, torch, copy, random, argparse
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

from src.model.linear import single_train_test
from src.scar.scar_calculate import SCARcalculation


class AudioCapsMultiModalTest():
    def __init__(self, encoder_type="clap"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.audio_embed_path = f"data/embeddings/audiocaps/{encoder_type}/train/audio"
        self.text_embed_path = f"data/embeddings/audiocaps/{encoder_type}/train/text"
        self.audio_test_embed_path = f"data/embeddings/audiocaps/{encoder_type}/val/audio"
        self.text_test_embed_path = f"data/embeddings/audiocaps/{encoder_type}/val/text"

        self.calculation = SCARcalculation()

    def load_json_file(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    def load_embeddings(self, path):
        files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.json')]
        data = []
        with ThreadPoolExecutor() as executor:
            for batch in executor.map(self.load_json_file, files):
                data.extend(batch)
        print(f"Loaded {len(data)} samples from {path}")
        return data

    def split_primary_reserve(self, data, split_ratio=0.6):
        label_to_items = defaultdict(list)
        for item in data:
            label_to_items[item['label']].append(item)

        primary_set = []
        reserve_set = defaultdict(list)
        rng = np.random.default_rng()

        for label, items in label_to_items.items():
            n = len(items)
            n_primary = int(n * split_ratio)
            indices = rng.permutation(n)
            primary_set.extend([items[i] for i in indices[:n_primary]])
            reserve_set[label] = [items[i] for i in indices[n_primary:]]

        return primary_set, reserve_set

    def downsample_embeddings(self, data, ratio=1.0):
        if ratio < 1.0:
            label_to_items = defaultdict(list)
            for item in data:
                label_to_items[item['label']].append(item)
            sampled_data = []
            for label, items in label_to_items.items():
                k = int(len(items) * ratio)
                sampled_data.extend(random.sample(items, k))
            data = sampled_data

        X = np.array([item['embedding'] for item in data])
        y = np.array([item['label'] for item in data])
        return X, y

    def training_testing_with_given_data(self, train_set, test_set, label_type, num_class, sample_ratio=1.0):
        X_train, y_train = self.downsample_embeddings(train_set, sample_ratio)
        X_test, y_test = self.downsample_embeddings(test_set)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        acc, logits, preds, labels = single_train_test(self.device, X_train, y_train, X_test, y_test, num_class)
        return len(X_train), num_class, acc, logits, preds, labels

    def foundation_size_estimate(self, sample_num, encoder_type, label_type, ratios, train_set, test_set, num_class, write_hs=False):
        scar_indexes = {}
        primary_acc = 0

        for ratio in ratios:
            cur_indexes = []
            print(f"[{datetime.now()}] Ratio: {1/ratio}")
            for _ in range(ratio):
                result = self.training_testing_with_given_data(train_set, test_set, label_type, num_class, sample_ratio=1/ratio)
                num_samples, _, _, logits, preds, labels = result
                type_results = {label_type: [logits, preds, labels]}
                scar, acc = self.calculation.calculation(type_results, num_samples, ratio)
                if ratio == 1:
                    primary_acc = acc
                cur_indexes.append(scar)

            avg_scar = defaultdict(dict)
            for task in cur_indexes[0][label_type]:
                for key in ['scale', 'coverage', 'authenticity', 'richness']:
                    avg = np.mean([sc[label_type][task][key] for sc in cur_indexes])
                    avg_scar[task][key] = avg
            scar_indexes[ratio] = dict(avg_scar)

        task_res = {}
        for task in scar_indexes[1]:
            series = [[scar_indexes[r][task][k] for r in ratios] for k in ['scale', 'coverage', 'authenticity', 'richness']]
            h, size = self.calculation.predict_foudation_step(sample_num, num_class, ratios, series)
            task_res[task] = (h, size)

        hs = [v[0] for v in task_res.values()]
        avg_auth = np.mean([scar_indexes[1][task]["authenticity"] for task in scar_indexes[1]])
        ratio = (1 - primary_acc) / (1 - avg_auth) if avg_auth < 1 else 0
        ratio = max(ratio, np.e)

        foundation_size = self.calculation.predict_foundation_total_fast(hs, ratio, max_order=3)
        if not foundation_size:
            foundation_size = sum([v[1] for v in task_res.values()])

        if write_hs:
            with open(f"src/scar/hs_list_audiocaps_{encoder_type}_{write_hs}.json", "w") as f:
                json.dump({"ratio": ratio, "hs": hs}, f)

        return foundation_size, scar_indexes, task_res


def single_modal_main(modal_type, encoder_type, label_type):
    print(f"\nProcessing AudioCaps dataset with encoder {encoder_type} in {modal_type} modal.")
    classifier = AudioCapsMultiModalTest(encoder_type)
    num_class = 5
    ratios = [1, 2, 5, 10, 15, 20, 30]

    # =========================== 1. Total Set Test ===========================
    if modal_type == "audio":
        total_set = classifier.load_embeddings(classifier.audio_embed_path)
        test_set = classifier.load_embeddings(classifier.audio_test_embed_path)
    elif modal_type == "text":
        total_set = classifier.load_embeddings(classifier.text_embed_path)
        test_set = classifier.load_embeddings(classifier.text_test_embed_path)
    else:
        raise ValueError("Unsupported modal type. Choose from 'audio' or 'text'.")

    total_test_results = classifier.training_testing_with_given_data(total_set, test_set, label_type, num_class, sample_ratio=1)
    total_sample_num, _, total_test_acc, _, _, _ = total_test_results
    total_foundation_size, _, _ = classifier.foundation_size_estimate(total_sample_num, encoder_type, label_type, ratios, total_set, test_set, num_class, write_hs="Total")

    # =========================== 2. Primary Set Test ===========================
    primary_set, reserve_set = classifier.split_primary_reserve(total_set, split_ratio=0.6)
    data_size, rdata_size = len(primary_set), sum([len(reserve_set[key]) for key in reserve_set])
    print(f"Primary Set Size: {data_size}, Reserve Set Size: {rdata_size}")

    primary_test_results = classifier.training_testing_with_given_data(primary_set, test_set, label_type, num_class, sample_ratio=1)
    primary_sample_num, _, primary_test_acc, _, _, _ = primary_test_results
    primary_foundation_size, scar_indexes_with_ratio, task_res = classifier.foundation_size_estimate(primary_sample_num, encoder_type, label_type, ratios, primary_set, test_set, num_class, write_hs="Primary")
    print(f"Primary Foundation Set Size: {primary_foundation_size}")

    # =========================== 3. Fill data set ===========================
    fill_size = rdata_size
    subtype_count = Counter([item['label'] for item in primary_set])
    subtype_fill_size = defaultdict(int)
    for tar in scar_indexes_with_ratio[1]:
        subtype_fill_size[tar] = max(0, task_res[tar][1] - subtype_count[tar])

    total = sum(subtype_fill_size.values())
    norm_subtype_fill_size = {k: v / total for k, v in subtype_fill_size.items()}

    available_reserve_sizes = []
    for key in norm_subtype_fill_size:
        if (key in reserve_set) and norm_subtype_fill_size[key] > 0:
            num = min(len(reserve_set[key]), task_res[key][1] - subtype_count[key]) / norm_subtype_fill_size[key]
            available_reserve_sizes.append(num)

    available_reserve_size = min([fill_size] + available_reserve_sizes)
    print(f"Available Reserve Size: {available_reserve_size}, Fill Size: {fill_size}.")
    print(f"Normalized Subtype Fill Size: {norm_subtype_fill_size}")

    extend_set = copy.deepcopy(primary_set)
    for label in subtype_fill_size:
        if label in reserve_set:
            add_data_size = int(available_reserve_size * norm_subtype_fill_size[label])
            add_data = random.sample(reserve_set[label], add_data_size)
            extend_set.extend(add_data)
    extend_size = len(extend_set)
    print(f"Filled Data size is {extend_size}")

    extend_test_results = classifier.training_testing_with_given_data(extend_set, test_set, label_type, num_class, sample_ratio=1)
    _, _, extend_testing_acc, _, _, _ = extend_test_results

    # =========================== 3.1 Random Fill data set ===========================
    avaliable_reserve_set = []
    for label in subtype_fill_size:
        if label in reserve_set:
            avaliable_reserve_set.extend(reserve_set[label])

    random_extend_set = copy.deepcopy(primary_set)
    extend_random = random.sample(avaliable_reserve_set, len(extend_set) - len(primary_set))
    random_extend_set.extend(extend_random)

    print(f"Random Filled Data size is {len(random_extend_set)}")
    random_extend_test_results = classifier.training_testing_with_given_data(random_extend_set, test_set, label_type, num_class, sample_ratio=1)
    _, _, random_extend_testing_acc, _, _, _ = random_extend_test_results

    # =========================== 3.2 Average Fill data set ===========================
    average_extend_set = copy.deepcopy(primary_set)
    add_num = (len(extend_set) - len(primary_set)) / len([label for label in subtype_fill_size if label in reserve_set])
    for label in subtype_fill_size:
        if label in reserve_set:
            add_data = random.sample(reserve_set[label], min(int(add_num), len(reserve_set[label])))
            average_extend_set.extend(add_data)

    print(f"Average Filled Data size is {len(average_extend_set)}")
    average_extend_test_results = classifier.training_testing_with_given_data(average_extend_set, test_set, label_type, num_class, sample_ratio=1)
    _, _, average_extend_testing_acc, _, _, _ = average_extend_test_results

    # =========================== 4. Result Record ===========================
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
    audio_size_results, audio_acc_results = single_modal_main("audio", encoder_type, label_type)


    result_save_path = f"experiments/results/audiocaps_experiment_results.txt"
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    with open(result_save_path, 'a') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  AudioCaps - Encoder: {encoder_type}; Label Type: {label_type}\n")
        f.write("=" * 70 + "\n")

        # TEXT MODALITY
        f.write(f"  [TEXT MODALITY]\n")
        f.write(f"  Total data size {text_size_results['primary'] + text_size_results.get('reserve', 0)}; "
                f"Primary datasize: {text_size_results['primary']}; "
                f"Extend datasize: {text_size_results['extend']}.\n")
        f.write(f"  Total Foundation data size estimation: {text_size_results['total_foundation']}\n")
        f.write(f"  Primary Foundation data size estimation: {text_size_results['primary_foundation']}\n")
        f.write(f"  1. Total Data Test Acc: {text_acc_results['total']:.4f}\n")
        f.write(f"  2. Primary Data Test Acc: {text_acc_results['primary']:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"  3. Extend Data Test Acc: {text_acc_results.get('extent', 0):.4f}\n")
        f.write(f"  4. Random Extend Data Test Acc: {text_acc_results.get('random', 0):.4f}\n")
        f.write(f"  5. Average Extend Data Test Acc: {text_acc_results.get('average', 0):.4f}\n")
        f.write("=" * 70 + "\n")

        # AUDIO MODALITY
        f.write(f"  [AUDIO MODALITY]\n")
        f.write(f"  Total data size {audio_size_results['primary'] + audio_size_results.get('reserve', 0)}; "
                f"Primary datasize: {audio_size_results['primary']}; "
                f"Extend datasize: {audio_size_results['extend']}.\n")
        f.write(f"  Total Foundation data size estimation: {audio_size_results['total_foundation']}\n")
        f.write(f"  Primary Foundation data size estimation: {audio_size_results['primary_foundation']}\n")
        f.write(f"  1. Total Data Test Acc: {audio_acc_results['total']:.4f}\n")
        f.write(f"  2. Primary Data Test Acc: {audio_acc_results['primary']:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"  3. Extend Data Test Acc: {audio_acc_results.get('extent', 0):.4f}\n")
        f.write(f"  4. Random Extend Data Test Acc: {audio_acc_results.get('random', 0):.4f}\n")
        f.write(f"  5. Average Extend Data Test Acc: {audio_acc_results.get('average', 0):.4f}\n")
        f.write("=" * 70 + "\n\n")


if __name__ == "__main__":
    # encoder = "clap"
    # encoder = "pengi"
    # label = "origin"
    # for _ in range(1):
    #     main(encoder, label)

    for encoder in ["clap", "pengi"]:
        label = "origin"
        for _ in range(3):
            print(f"Running with Encoder: {encoder}, Label: {label}")
            main(encoder, label)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--encoder_type", type=str, required=True)
    # parser.add_argument("--label_type", type=str, default="origin")
    # parser.add_argument("--round", type=int, default=0)  # 可选：打印轮数或用于种子等
    # args = parser.parse_args()

    # main(args.encoder_type, args.label_type)