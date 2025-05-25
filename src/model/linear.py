import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    

def single_train_test(device, X_train, y_train, X_val, y_val, num_class=10):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

    model = LinearClassifier(X_train.shape[1], num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------- Training --------
    for _ in range(20):
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
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        logits = model(X_val)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_val).float().mean().item()

    logits_cpu = logits.cpu().numpy()  # [B, D]
    preds_cpu = preds.cpu().tolist()    # [B]
    labels_cpu = y_val.cpu().tolist()   # [B]
    
    return acc, logits_cpu, preds_cpu, labels_cpu