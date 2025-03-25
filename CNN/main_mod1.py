import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.data_loader import load_data, prepare_dataset
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define global constants for filter sizes
FILTER1 = 64
FILTER2 = 128


class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=FILTER1, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv1d(in_channels=FILTER1, out_channels=FILTER2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(FILTER1)  # Batch normalization after conv1
        self.bn2 = nn.BatchNorm1d(FILTER2)  # Batch normalization after conv2
        self.dropout = nn.Dropout(0.6)  # Dropout layer to prevent overfitting
        
        # Calculate the size after convolutions and pooling
        self.flattened_size = FILTER2 * (input_size // 4)
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # First conv + pool + batch norm
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Second conv + pool + batch norm
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first FC layer
        x = self.fc2(x)
        return x

# Data loading
num_samples = 12000
input_size = 194  # Number of features per sample
num_classes = 2

def main():
    data = load_data()
    spectra, labels = prepare_dataset(data)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Split into train and test datasets
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 50
    best_val_loss = float('inf')  # Track the best validation loss for early stopping
    train_losses, val_losses = [], []  # To track the losses during training

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Average training loss for this epoch
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "cnn_model_best.pth")  # Save the best model

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plot training and validation loss
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()

    # Final test accuracy
    model.load_state_dict(torch.load("cnn_model_best.pth"))
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

if __name__ == "__main__":
    main()


# Smaller Kernel Size: Changed to 3 for better capturing of local features.

# More Filters: Increased the number of filters to 32 and 64.

# Dropout: Added dropout layers after the fully connected layers to prevent overfitting.

# Batch Normalization: Added after the convolution layers.

# Early Stopping: Monitors validation loss to stop training early.



# Batch Normalization (BatchNorm1d) after each convolution layer. This stabilizes training and helps with faster convergence.

# Dropout (Dropout(0.5)) after the first fully connected layer to reduce overfitting.

# Tracking both training and validation loss during each epoch and plotting them.

# Early Stopping: If the validation loss improves, we save the model’s state; otherwise, training continues.


# kernel 3, filters 32&64, dropout 0.5, lr 0.001
# Epoch [50/50], Train Loss: 0.4870, Val Loss: 0.6686
# Test Accuracy: 0.6687

# kernel 3, filters 32&64, dropout 0.5, lr 0.0005
# Epoch [50/50], Train Loss: 0.3294, Val Loss: 0.8310
# Test Accuracy: 0.6637

# kernel 3, filters 32&64, dropout 0.3, lr 0.0005
# Epoch [50/50], Train Loss: 0.2013, Val Loss: 1.1406
# Test Accuracy: 0.6575

# kernel 3, filters 64&128, dropout 0.5, lr 0.0005
# Epoch [20/20], Train Loss: 0.5615, Val Loss: 0.5893
# Test Accuracy: 0.6637

# kernel 3, filters 64&128, dropout 0.6, lr 0.0005
# Epoch [20/20], Train Loss: 0.5755, Val Loss: 0.5809
# Test Accuracy: 0.6792



