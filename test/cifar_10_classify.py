import sys, json, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from src.model.linear import LinearClassifier



class CIFAR10ClassifyTest():
    def __init__(self, encoder_type="resnet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        if encoder_type == "resnet":
            self.embed_path = "data/embeddings/resnet/cifar10_embeddings.json"
        elif encoder_type == "vit":
            self.embed_path = "data/embeddings/vit/cifar10_embeddings.json"
        elif encoder_type == "dino":
            self.embed_path = "data/embeddings/dino/cifar10_embeddings.json"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'resnet', 'vit', or 'dino'.")

    def load_embeddings(self, json_path):
        data = []
        with open(json_path, "r") as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

        X = np.array([item['embedding'] for item in data])  # (N, D)
        y = np.array([item['label'] for item in data])      # (N,)
        return X, y
    
    def train_test(self):
        X, y = self.load_embeddings(self.embed_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # -------- Training --------
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=500, shuffle=True)

        model = LinearClassifier(X_train.shape[1], 10).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(20):
            model.train()
            total_loss = 0.0
            epoch_bar = tqdm(loader, desc=f"Epoch {epoch+1}", ascii=True, leave=True)

            for xb, yb in epoch_bar:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                avg_loss = total_loss / (len(loader))
                now = datetime.now().strftime("%H:%M:%S")
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1024 / 1024
                    mem_str = f"{mem:.1f} MB"
                else:
                    mem_str = "CPU"

                epoch_bar.set_postfix({
                    "time": now,
                    "gpu": mem_str,
                    "loss": f"{avg_loss:.4f}"
                })

        # -------- Evaluation --------
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_test_tensor).float().mean().item()

        return acc
    
    def train_test_k_flods(self, k_folds=5):
        X, y = self.load_embeddings(self.embed_path)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        all_fold_acc = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n----- Fold {fold+1}/{k_folds} -----")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

            model = LinearClassifier(X.shape[1], 10).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # -------- Training --------
            for epoch in range(20):
                model.train()
                total_loss = 0.0
                epoch_bar = tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch+1}", ascii=True, leave=True)

                for xb, yb in epoch_bar:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    avg_loss = total_loss / len(train_loader)
                    now = datetime.now().strftime("%H:%M:%S")
                    if torch.cuda.is_available():
                        mem = torch.cuda.memory_allocated() / 1024 / 1024
                        mem_str = f"{mem:.1f} MB"
                    else:
                        mem_str = "CPU"

                    epoch_bar.set_postfix({
                        "time": now,
                        "gpu": mem_str,
                        "loss": f"{avg_loss:.4f}"
                    })

            # -------- Validation --------
            model.eval()
            with torch.no_grad():
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                logits = model(X_val)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_val).float().mean().item()

            print(f"Fold {fold+1} Accuracy: {acc:.4f}")
            all_fold_acc.append(acc)

        return all_fold_acc




if __name__ == "__main__":
    # for encoder_type in ["resnet", "vit", "dino"]:   
    #     classifier = CIFAR10ClassifyTest(encoder_type)
    #     acc = classifier.train_test()
    #     print(f"✅ {encoder_type} Results:", acc)

    encoder_type = "dino"
    k_folds = 5

    classifier = CIFAR10ClassifyTest(encoder_type)
    all_fold_acc = classifier.train_test_k_flods(k_folds=k_folds)

    print("\n=== Fold Accuracies ===")
    for i, acc in enumerate(all_fold_acc):
        print(f"Fold {i+1}: {acc:.4f}")

    avg_acc = sum(all_fold_acc) / k_folds
    print(f"\n===== Average Accuracy over {k_folds} folds: {avg_acc:.4f} =====")