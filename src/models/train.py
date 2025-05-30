"""
train.py
--------
Trains the MotorImageryCNN model on preprocessed EEG data using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn import MotorImageryCNN
import numpy as np

def train_model(X_train, y_train, X_val, y_val, input_shape, n_classes, batch_size=32, epochs=30, lr=1e-5, device=None):
    """
    Train the MotorImageryCNN model on EEG data.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        input_shape (tuple): Shape of the input tensor (channels, height, width).
        n_classes (int): Number of output classes.
        batch_size (int, optional): Batch size. Defaults to 32.
        epochs (int, optional): Number of training epochs. Defaults to 30.
        lr (float, optional): Learning rate. Defaults to 1e-5.
        device (torch.device, optional): Device to use ('cuda' or 'cpu'). Defaults to auto-detect.

    Returns:
        MotorImageryCNN: Trained model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert numpy arrays to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    # Add channel dimension if needed
    if X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = MotorImageryCNN(input_shape, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model

def export_to_onnx(model, input_shape, export_path="model.onnx"):
    """
    Export a trained PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        input_shape (tuple): Shape of the input tensor (batch, channels, height, width).
        export_path (str, optional): Path to save the ONNX file. Defaults to "model.onnx".
    """
    import torch
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Model exported to {export_path}")
