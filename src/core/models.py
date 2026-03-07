import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import Optional, Any, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.core.base import BaseModel

class BaselineRFClassifier(BaseModel):
    """
    Random Forest Baseline Classifier.
    """
    def __init__(self, n_estimators=100, max_depth=25, min_samples_split=100, 
                 min_samples_leaf=50, max_features=0.1, random_seed=42, n_jobs=8):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            **self.params, 
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """
        Trains the RF model with internal scaling.
        """
        X_sc = self.scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        self.model.fit(X_sc, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the trained model and scaler.
        """
        X_sc = self.scaler.transform(X.astype(np.float64)).astype(np.float32)
        return self.model.predict(X_sc)

    @staticmethod
    def find_best_params(X: np.ndarray, y: np.ndarray, param_dist: Dict, n_iter: int = 20, cv: int = 3, n_jobs: int = 8):
        """
        Static method to perform hyperparameter search.
        """
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=n_jobs, class_weight='balanced'),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            random_state=42,
            n_jobs=n_jobs,
            verbose=1
        )
        search.fit(X_sc, y)
        return search.best_params_

class MicroProbingClassifier(BaseModel):
    """
    Implements the logic for 'probes': training n_classes*(n_classes-1)/2 
    OVO (One-Vs-One) individual classifiers for a single observation point.
    Originally from Stage 2 logic.
    """
    def __init__(self, classifier_type: str = "LinearSVC", penalty: str = "l1", C: float = 1.0, random_seed: int = 42):
        self.classifier_type = classifier_type
        self.penalty = penalty
        self.C = C
        self.random_seed = random_seed
        self.weights_ = None
        self.intercepts_ = None

    def train_probe(self, X_probe: np.ndarray):
        """
        X_probe: shape (n_classes, n_features)
        Returns flattened weight vector and intercepts.
        """
        n_classes = X_probe.shape[0]
        n_features = X_probe.shape[1]
        all_weights = []
        all_intercepts = []
        
        # OVO loop
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                X_pair = X_probe[[i, j]]
                y_pair = np.array([0, 1])
                
                if self.classifier_type == "LinearSVC":
                    from sklearn.svm import LinearSVC
                    clf = LinearSVC(penalty=self.penalty, C=self.C, dual=False, 
                                   random_state=self.random_seed, max_iter=10000)
                else:
                    raise ValueError(f"Classifier {self.classifier_type} not implemented for probing.")
                    
                clf.fit(X_pair, y_pair)
                all_weights.append(clf.coef_.flatten())
                all_intercepts.append(clf.intercept_)
                
        self.weights_ = np.concatenate(all_weights)
        self.intercepts_ = np.concatenate(all_intercepts)
        return np.concatenate([self.weights_, self.intercepts_])

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Any:
        # For probing, we usually iterate over probes. 
        # This wrapper can handle parallel execution if needed.
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MicroProbingClassifier is for weight extraction, not direct prediction.")

class SimpleTransformerClassifier(nn.Module, BaseModel):
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

    def train(self, train_loader: Any, val_loader: Optional[Any] = None, epochs: int = 20, lr: float = 1e-3, device: str = "cpu") -> Any:
        return train_model(self, train_loader, val_loader, epochs, lr, device)

    def predict(self, X: Any) -> Any:
        self.eval()
        with torch.no_grad():
            return self.forward(X)

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
