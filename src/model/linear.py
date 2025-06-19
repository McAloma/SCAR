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
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

    model = LinearClassifier(X_train.shape[1], num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(device)
            y_val_device = y_val.to(device)
            logits = model(X_val_device)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_val_device).float().mean().item()

        # -------- Early Stopping Check --------
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # print(f"Early stopping at epoch {epoch+1} (best val acc = {best_acc:.4f})")
                break

    # -------- Load best model --------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        X_val_device = X_val.to(device)
        y_val_device = y_val.to(device)
        logits = model(X_val_device)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_val_device).float().mean().item()

    logits_cpu = logits.cpu().numpy()
    preds_cpu = preds.cpu().tolist()
    labels_cpu = y_val_device.cpu().tolist()

    return acc, logits_cpu, preds_cpu, labels_cpu, model



# def single_train_test(device, X_train, y_train, X_val, y_val, num_class=10):
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

#     model = LinearClassifier(X_train.shape[1], num_class).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     # -------- Training --------
#     for _ in range(100):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             out = model(xb)
#             loss = criterion(out, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#     # -------- Validation --------
#     model.eval()
#     with torch.no_grad():
#         X_val = X_val.to(device)
#         y_val = y_val.to(device)
#         logits = model(X_val)
#         preds = torch.argmax(logits, dim=1)
#         acc = (preds == y_val).float().mean().item()

#     logits_cpu = logits.cpu().numpy()  # [B, D]
#     preds_cpu = preds.cpu().tolist()    # [B]
#     labels_cpu = y_val.cpu().tolist()   # [B]
    
#     return acc, logits_cpu, preds_cpu, labels_cpu
