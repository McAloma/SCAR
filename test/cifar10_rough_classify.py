import sys, os, json, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import chi2
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from src.model.linear import LinearClassifier

from sklearn.neighbors import NearestNeighbors


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
    
        
    def single_train_test(self, X_train, y_train, X_val, y_val, num_class=10):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

        model = LinearClassifier(X_train.shape[1], num_class).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # -------- Training --------
        for epoch in range(20):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            logits = model(X_val)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_val).float().mean().item()

        preds_cpu = preds.cpu().tolist()
        labels_cpu = y_val.cpu().tolist()

        return acc, preds_cpu, labels_cpu


    def calculate_nearest_neighbors_label_similarity(self, X_train, y_train, X_val, y_val, k=10):
        X_train = X_train.cpu().numpy()
        X_val = X_val.cpu().numpy()

        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X_train)

        _, indices = neigh.kneighbors(X_val)
        nearest_labels = y_train[indices]
        similarity_scores = (nearest_labels == y_val[:, None]).sum(axis=1) / k

        return similarity_scores

    
    def k_flods_testing_with_label_mapping(self, k_folds, rough_type, sample_ratio=1.0):
        # -------------------- Load Sample Data --------------------
        X, y = self.load_embeddings(self.embed_path, sample_ratio)
        X = torch.tensor(X, dtype=torch.float32)
        num_class, map_y = self.label_mapping(y, rough_type)
        y = torch.tensor(y, dtype=torch.long)
        map_y = torch.tensor(map_y, dtype=torch.long)

        # -------------------- Load Test Data --------------------
        X_test, y_test = self.load_embeddings(self.test_embed_path)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        num_class, map_y_test = self.label_mapping(y_test, rough_type)
        y_test = torch.tensor(y_test, dtype=torch.long)
        map_y_test = torch.tensor(map_y_test, dtype=torch.long)

        kf = StratifiedKFold(n_splits=k_folds, shuffle=False)       # for label ratio

        all_fold_acc = []
        sample_x_results = [0] * len(X)
        sample_y_results = [-1] * len(X)
        folds = list(kf.split(X, map_y))

        fold_bar = tqdm(enumerate(folds), total=k_folds, ascii=True)
        for fold, (train_idx, val_idx) in fold_bar:
            now = datetime.now().strftime("%H:%M:%S")
            fold_bar.set_description(f"[{now}] Fold {fold+1}/{k_folds} | Task: {rough_type}")

            X_train, X_val = X[train_idx], X[val_idx]
            map_y_train, map_y_val = map_y[train_idx], map_y[val_idx]

            # print("1. Feature Hypothesis Testing")
            x_values = self.calculate_nearest_neighbors_label_similarity(X_test, map_y_test, X_val, map_y_val)
            # print("2. Training-Validation Results")
            acc, preds_cpu, labels_cpu = self.single_train_test(X_train, map_y_train, X_val, map_y_val, num_class)

            for idx, x_value, pred, label in zip(val_idx, x_values, preds_cpu, labels_cpu):
                sample_x_results[idx] = x_value
                sample_y_results[idx] = (pred == label)

            all_fold_acc.append(acc)

        # -------------------- Final Evaluation --------------------
        testing_acc, preds_cpu, labels_cpu = self.single_train_test(X, y, X_test, y_test)

        return all_fold_acc, testing_acc, sample_x_results, sample_y_results



if __name__ == "__main__":
    from collections import defaultdict, Counter
    from src.scar.scar_calculate import SCARcalculation

    calculation = SCARcalculation()

    save_path = "test/results/scar_experiment_results.txt"
    # encoder_type = "resnet"
    k_folds = 5
    for encoder_type in ["resnet", "vit", "dino"]:

        classifier = CIFAR10ClassifyTest(encoder_type)
        # for ratio in [1, 2, 5, 10, 20, 50]:
        for ratio in [50]:
            # NOTE: ratio = 1 means using all data
            indexes, falses, total_false_num = [], [], []
            kfold_accs, testing_accs = [], []

            for cur in range(1):
                results_list = defaultdict(list)  
                counter_dict = {}
                for rough_type in ["what", "where", "how"]:
                    print(f"\n=== Rough Type: {rough_type} with {encoder_type} at ratio={1/ratio} in {cur+1}/{ratio} ===")
                    all_fold_acc, testing_acc, sample_x_results, sample_y_results = classifier.k_flods_testing_with_label_mapping(k_folds=k_folds, rough_type=rough_type, sample_ratio=1/ratio)
                    results_list[rough_type].append(sample_x_results)
                    results_list[rough_type].append(sample_y_results)

                    print("=== Fold Accuracies ===")
                    for i, acc in enumerate(all_fold_acc):
                        print(f"Fold {i+1}: {acc:.4f}")
                    avg_acc = sum(all_fold_acc) / k_folds
                    print(f"===== Average Accuracy over {k_folds} folds: {avg_acc:.4f} =====")
                    kfold_accs.append(avg_acc)
                    print(f"===== Testing Accuracy: {testing_acc:.4f} =====")
                    testing_accs.append(testing_acc)

                    counter = Counter(sample_y_results)
                    print(f"===== Sample Results Counter:{counter} =====\n")
                    counter_dict[rough_type] = counter

                for key in results_list.keys():
                    task_results = results_list[key]

                false_indices = calculation.calculate_all_false_index(results_list)
                print("False Indices:", false_indices)
                total_false_num.append(len(false_indices))
                print("Number of False Indices:", len(false_indices))

                scar_index = calculation.calculate_scar(results_list, ratio)
                indexes.append(scar_index)

                falses = [counter_dict[key][False] for key in results_list.keys()]
                avg_false = sum(falses) / len(falses)
                falses.append(avg_false)
            
            print(indexes)

            avg_kfold_acc = sum(kfold_accs) / len(kfold_accs)
            avg_testing_acc = sum(testing_accs) / len(testing_accs)
            avg_falses = sum(falses) / len(falses)
            avg_total_false = sum(total_false_num) / len(total_false_num)

            with open(save_path, "a") as f:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("=" * 60 + "\n")
                f.write(f"[Timestamp]      {current_time}\n")
                f.write(f"[Encoder]        {encoder_type}\n")
                f.write(f"[Ratio]          1/{ratio:.0f} = {1/ratio:.2f}\n")
                f.write(f"[False Avg.]     avg_false = {avg_false:.4f}, avg_total_false = {avg_total_false:.4f}\n")
                f.write("\n")
                f.write("[Average Metric Results per K-Fold]:\n")
                for index in indexes:
                    for rough_type, metrics in index.items():
                        f.write(f"  [{rough_type.upper()}]\n")
                        for metric_name, value in metrics.items():
                            f.write(f"    - {metric_name:<15}: {value:.4f}\n")
                f.write("\n")
                f.write(f"[Average K-Fold Accuracy]     : {avg_kfold_acc:.4f}\n")
                f.write(f"[Average Final Test Accuracy] : {avg_testing_acc:.4f}\n")
        
        with open(save_path, "a") as f:
            f.write("=" * 60 + "\n")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Experiment completed at {current_time}\n")
            f.write("=" * 60 + "\n")


    
    # ————————————————————————————————————————————————————————————————————————————————————————————

    # 我们讨论什么是 foundation data 的时候，我们就是想知道什么样的数据集能够完全代表这个 domain 中的所有可能性，换句话说，即什么样的 dataset 能够近似穷举 domian 中的情况。
    # 然而现实是，我们无法穷举。第一，我们不知道某个 domian 下具体的表现是什么。第二，我们没有一个极其强大的算法能够完全拟合所有的情况。
    # 从机器学习的角度出发，数据质量从另一个角度上来说是数据的可分性，当不同类别数据表征之间的距离越大时，可分性越好，即越容易被学习到规则。

    # 从计算角度来说 SCAR 用来评价数据集是针对模型的，那当数据量越小，则 SCAR 越容易大，因为其可分性越好。
    # 所以自回归过程为了说明真实的 foundation data 的 SCAR，一定要引入数据量的影响。

    # Idea：基于判断一个数据集的质量本质上是一个“信号任务”。因此我们基于 sigmoid 函数来估计 Foundation data。
    # 具体来说，我们首先计算全量数据集的 SCAR， 然后通过 sigmoid 逆函数的方式来估计 SCAR 的x值。
    # 根据向下采样的比例，我们在同比例的 x 值下获得子集在 sigmoid 函数上的比值，然后作为规模函数进行相乘。
    # 最后用相乘的结果进行自回归，最后获得真实情况下，当 SCAR 接近 100 时，foundation data 的具体数据。

    # ——————————————————————————————————————————————————————————————————————————————————————————————
    # 从 Data centric 的角度出发，我们考虑数据对于 hypothesis 的影响。
    # 假设在特征空间中移除一条 sample，如同 SVM 一样，当其不对边界起到决定性作用时，则移除这个 sample 不会影响模型的性能。
    # 所以我们考虑多个针对同一个数据集的粗粒度任务来观察数据在不同任务下的表现。
    # 1. 当一个样本在所有粗粒度任务下都表现正确时，我们可以认为他在知识空间的核心区域，即距离每个粗粒度任务的边界都很远。
    # 2. 当一个样本在不同的粗粒度任务下表现不一致，但是有对也有错时，则我们可以认为他在知识空间的边界处，其一定程度上决定了模型能够学习到什么样的 hypothesis。
    # 3. 当一个样本在所有粗粒度任务下都表现不正确时，我们可以认为这个数据在现有模型下是无效的。即他所提供的表征无法帮助模型进行知识表达，所以不能算作基于当前模型的 foundation data。

    # ——————————————————————————————————————————————————————————————————————————————————————————————
    # 一个直觉的感觉是，对于同一条数据来说，当其插入小样本和大样本中时，其对于模型的贡献程度是不一样的。
    # 数据对于数据质量的贡献应该服从一个类似“无记忆性”的“同质性”，即假定所有数据的地位都是同等的，那么我往大数据集中插入 1000 条数据，和在小数据集中插入同样数据，这个数据的贡献程度应该是一样的。（前提是数据是有效的）