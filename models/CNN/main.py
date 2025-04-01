import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import logging
import psutil
from src.data_loader import load_data, prepare_dataset
from sklearn.model_selection import train_test_split


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILTER1 = 64
FILTER2 = 64

class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()
        
        # Parameters
        self.kernel_size = 5
        self.padding = 1
        self.pool_size = 2
        
        # First conv layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=FILTER1, kernel_size=self.kernel_size, padding=self.padding)        
        self.conv2 = nn.Conv1d(in_channels=FILTER1, out_channels=FILTER2, kernel_size=self.kernel_size, padding=self.padding)
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size, stride=2)



        
        # Calculate the size after convolutions and pooling
        # # After two pooling layers with stride 2, the size is reduced by factor of 4
        # self.flattened_size = FILTER2 * (input_size // 4)
        conv1_out = (input_size - self.kernel_size + 2*self.padding + 1)
        pool1_out = conv1_out // 2
        conv2_out = (pool1_out - self.kernel_size + 2*self.padding + 1)
        pool2_out = conv2_out // 2
        
        self.flattened_size = FILTER2 * pool2_out
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_memory_usage():
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert bytes to MB
        return cpu_mem, gpu_mem
    return cpu_mem, 0

def main():
    # Measure initial memory usage
    cpu_mem_before, gpu_mem_before = get_memory_usage()
    print(f"Initial CPU Memory Usage: {cpu_mem_before:.2f} MB")
    if torch.cuda.is_available():
        print(f"Initial GPU Memory Usage: {gpu_mem_before:.2f} MB")

    data = load_data()
    spectra, labels = prepare_dataset(data)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Split into train and test datasets
    num_samples = len(spectra)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D(input_size=X_tensor.shape[2], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses = []
    val_losses = []

    # Measure memory before training
    cpu_mem_train_start, gpu_mem_train_start = get_memory_usage()
    print(f"\nMemory Usage Before Training:")
    print(f"CPU Memory: {cpu_mem_train_start:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {gpu_mem_train_start:.2f} MB")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # Print memory usage every 10 epochs
        if (epoch + 1) % 10 == 0:
            cpu_mem, gpu_mem = get_memory_usage()
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Memory Usage:")
            print(f"CPU Memory: {cpu_mem:.2f} MB")
            if torch.cuda.is_available():
                print(f"GPU Memory: {gpu_mem:.2f} MB")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Measure final memory usage
    cpu_mem_after, gpu_mem_after = get_memory_usage()
    print(f"\nFinal Memory Usage:")
    print(f"CPU Memory: {cpu_mem_after:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {gpu_mem_after:.2f} MB")
    
    print(f"\nMemory Increase During Training:")
    print(f"CPU Memory Increase: {cpu_mem_after - cpu_mem_train_start:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory Increase: {gpu_mem_after - gpu_mem_train_start:.2f} MB")

    # Plot training vs validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker='o', linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    main() 


# kernel size 3, filters 32&64
# Epoch [20/20], Loss: 0.5556
# Test Accuracy: 0.6625

# Epoch [50/50], Loss: 0.3768
# Test Accuracy: 0.6713
# main_50.png

# Epoch [100/100], Loss: 0.2044
# Test Accuracy: 0.6471

# kernel 11, filters 16&32
# Epoch [50/50], Train Loss: 0.0656, Val Loss: 2.0182
# Test Accuracy: 0.6633

# kernel 5, filters 32&32
# Epoch [50/50], Train Loss: 0.1800, Val Loss: 1.3057
# Test Accuracy: 0.6633

# kernel 5, filters 64&64
# Epoch [50/50], Train Loss: 0.2131, Val Loss: 1.4602
# Test Accuracy: 0.6329
























# ### **Model Architecture:**

# 1. **Input Layer:**
#    - The input has a shape of **(batch_size, 1, 194)** where:
#      - `batch_size` is the number of samples in a batch.
#      - `1` is the number of input channels (since it's a single-channel 1D signal).
#      - `194` is the number of features per sample.

# 2. **First Convolutional Block:**
#    - `Conv1D(1 → 32, kernel_size=3, padding=1)`: Applies **32 filters** of size **3** to the input.
#    - `ReLU Activation`: Applies a non-linearity to introduce complexity.
#    - `MaxPool1D(kernel_size=2, stride=2)`: Reduces the feature length by **half (from 194 → 97).**

# 3. **Second Convolutional Block:**
#    - `Conv1D(32 → 64, kernel_size=3, padding=1)`: Applies **64 filters** of size **3**.
#    - `ReLU Activation`: Applies non-linearity again.
#    - `MaxPool1D(kernel_size=2, stride=2)`: Again reduces the feature length by **half (from 97 → 48).**

# 4. **Flattening Layer:**
#    - The output from the second **max-pooling layer** has a shape of **(batch_size, 64, 48)**.
#    - Flattening reshapes it into a vector of size **64 × 48 = 3072** (for each sample).

# 5. **Fully Connected Layers (FC):**
#    - `FC1 (3072 → 128)`: First fully connected layer, reduces dimensionality to **128 neurons**.
#    - `ReLU Activation`: Applies non-linearity.
#    - `FC2 (128 → 2)`: The final layer that outputs **2 values** (one per class).

# 6. **Output Layer:**
#    - No activation function at the last layer since **CrossEntropyLoss** expects raw logits.
#    - The highest logit determines the predicted class.

# ### **Model Flow:**
# 1. Input: `(batch_size, 1, 194)`
# 2. **Conv1 + ReLU + MaxPool** → `(batch_size, 32, 97)`
# 3. **Conv2 + ReLU + MaxPool** → `(batch_size, 64, 48)`
# 4. **Flatten** → `(batch_size, 3072)`
# 5. **FC1 + ReLU** → `(batch_size, 128)`
# 6. **FC2 (Logits for classification)** → `(batch_size, 2)`

# ---

# ### **Why This Architecture?**
# - **1D Convolutions** are great for analyzing **spectral or sequential data** because they capture **local patterns** along the feature axis.
# - **Pooling Layers** help **reduce dimensionality** while retaining important features.
# - **Fully Connected Layers** learn **higher-level representations** and perform classification.

# This model is a **shallow CNN**, meaning it's relatively simple and computationally efficient. If needed, you could experiment with **more convolutional layers**, **dropout**, or **batch normalization** to improve performance.

# ---

