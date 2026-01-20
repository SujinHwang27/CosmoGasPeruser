import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



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
        # x: (batch, features)

        # x = x.unsqueeze(1)                  # -> (batch, seq=1, features)
        # x = self.input_projection(x)        # -> (batch, 1, d_model)
        # x = self.encoder(x)                 # -> (batch, 1, d_model)
        # x = x.squeeze(1)                    # -> (batch, d_model)
        
        x = x.unsqueeze(-1)               # (batch, 20, 1)
        x = self.input_projection(x)      # (batch, 20, d_model)
        x = self.encoder(x)               # (batch, 20, d_model)
        x = x.mean(dim=1)                 # pooled representation

        return self.classifier(x)


def create_dataloader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


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
            optimizer.zero_grad()

            logits = model(Xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        preds, labels = [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)

                logits = model(Xb)
                loss = criterion(logits, yb)

                val_loss += loss.item()
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(labels, preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, history



def evaluate_model(model, X, y, device="cpu"):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(1).cpu().numpy()
        labels = y.cpu().numpy()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
