import copy, datetime, psutil
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)







def single_train_test(device, X_train, y_train, X_val, y_val, num_class=10, patience=5, max_epochs=100):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=500000, shuffle=True)

    model = LinearClassifier(X_train.shape[1], num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    with tqdm(range(max_epochs), ascii=True) as pbar:
        for _ in pbar:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            mem_usage = psutil.virtual_memory().used / (1024 ** 3)  # 转换为GB
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # 当前分配的显存
            else:
                gpu_mem = 0.0

            pbar.set_postfix_str(f"Time: {current_time} | RAM: {mem_usage:.1f}GB | GPU: {gpu_mem:.1f}GB")

            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            train_acc = correct / total

            # -------- Early Stopping Check based on TRAIN acc --------
            if train_acc > best_acc:
                best_acc = train_acc
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    # -------- Load best model --------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        # -------- 1. Collect training set outputs --------
        train_logits_list = []
        train_preds_list = []
        train_labels_list = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)

            train_logits_list.append(out.cpu())
            train_preds_list.append(preds.cpu())
            train_labels_list.append(yb.cpu())

        train_logits = torch.cat(train_logits_list, dim=0).numpy()
        train_preds = torch.cat(train_preds_list, dim=0).numpy()
        train_labels = torch.cat(train_labels_list, dim=0).numpy()

        # -------- 2. Evaluate on validation set --------
        X_val_device = X_val.to(device)
        y_val_device = y_val.to(device)
        logits_val = model(X_val_device)
        preds_val = torch.argmax(logits_val, dim=1)
        acc_val = (preds_val == y_val_device).float().mean().item()

    return acc_val, train_logits, train_preds, train_labels



def multi_label_train_test(device, X_train, y_train, X_val, y_val, num_class=10, patience=5, max_epochs=100, threshold=0.5):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)

    model = LinearClassifier(X_train.shape[1], num_class).to(device)
    criterion = nn.BCEWithLogitsLoss()  # ✅ 多标签用 BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for _ in tqdm(range(max_epochs), ascii=True):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float())  # ✅ 标签要是 float
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = (torch.sigmoid(out) > threshold).int()  # ✅ 多标签预测
            correct += ((preds == yb).all(dim=1).sum().item())  # ✅ 完全匹配正确的样本数
            total += yb.size(0)

        train_acc = correct / total

        # -------- Early Stopping Check based on TRAIN acc --------
        if train_acc > best_acc:
            best_acc = train_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # -------- Load best model --------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        # -------- 1. Collect training set outputs --------
        train_logits_list = []
        train_preds_list = []
        train_labels_list = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = (torch.sigmoid(out) > threshold).int()  # ✅ 预测为 multi-hot

            train_logits_list.append(out.cpu())
            train_preds_list.append(preds.cpu())
            train_labels_list.append(yb.cpu())

        train_logits = torch.cat(train_logits_list, dim=0).numpy()
        train_preds = torch.cat(train_preds_list, dim=0).numpy()
        train_labels = torch.cat(train_labels_list, dim=0).numpy()

        # -------- 2. Evaluate on validation set --------
        X_val_device = X_val.to(device)
        y_val_device = y_val.to(device)
        logits_val = model(X_val_device)
        preds_val = (torch.sigmoid(logits_val) > threshold).int()
        acc_val = ((preds_val == y_val_device).all(dim=1).sum().item()) / y_val_device.size(0)

    return acc_val, train_logits, train_preds, train_labels



def kfold_train_test(device, X_train, y_train, X_test, y_test, num_class=10, patience=5, max_epochs=100, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    saved_state_dicts = []

    for _, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

        valid_dataset = TensorDataset(X_val, y_val)
        valid_loader = DataLoader(valid_dataset, batch_size=1000, shuffle=True)

        model = LinearClassifier(X_train.shape[1], num_class).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for _ in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            # Early stopping based on train acc on this fold
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = torch.argmax(model(xb), dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            train_acc = correct / total

            if train_acc > best_acc:
                best_acc = train_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        saved_state_dicts.append(best_model_state)

    # Step 2: average models
    avg_state_dict = copy.deepcopy(saved_state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(saved_state_dicts)):
            avg_state_dict[key] += saved_state_dicts[i][key]
        avg_state_dict[key] /= len(saved_state_dicts)

    # Step 3: evaluate on the whole training set using avg model
    final_model = LinearClassifier(X_train.shape[1], num_class).to(device)
    final_model.load_state_dict(avg_state_dict)
    final_model.eval()

    with torch.no_grad():
        # -------- 1. Collect training set outputs --------
        train_logits_list = []
        train_preds_list = []
        train_labels_list = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = final_model(xb)
            preds = torch.argmax(out, dim=1)

            train_logits_list.append(out.cpu())
            train_preds_list.append(preds.cpu())
            train_labels_list.append(yb.cpu())

        train_logits = torch.cat(train_logits_list, dim=0).numpy()
        train_preds = torch.cat(train_preds_list, dim=0).numpy()
        train_labels = torch.cat(train_labels_list, dim=0).numpy()

        # -------- 2. Evaluate on validation set --------
        X_test_device = X_test.to(device)
        y_test_device = y_test.to(device)
        logits_val = final_model(X_test_device)
        preds_val = torch.argmax(logits_val, dim=1)
        acc_val = (preds_val == y_test_device).float().mean().item()

    return acc_val, train_logits, train_preds, train_labels