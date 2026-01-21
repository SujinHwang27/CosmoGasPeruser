import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Any

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim=20, num_classes=4, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.input_projection(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            # Ensure yb is 0-indexed if it comes as 1-4
            if yb.min() > 0:
                yb = yb - 1
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                if yb.min() > 0:
                    yb = yb - 1
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())

        val_acc = accuracy_score(labels, preds)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss/len(val_loader))
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, history
