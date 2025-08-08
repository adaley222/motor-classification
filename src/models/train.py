"""
train.py
--------
Trains the MotorImageryCNN model on preprocessed EEG data using PyTorch.
Now includes MLflow experiment tracking for comprehensive model management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn import MotorImageryCNN
import numpy as np
# MLflow imports for experiment tracking and model management
import mlflow
import mlflow.pytorch
import os
from datetime import datetime

def train_model(X_train, y_train, X_val, y_val, input_shape, n_classes, batch_size=32, epochs=30, lr=1e-5, device=None, experiment_name="motor-imagery-cnn", run_name=None):
    """
    Train the MotorImageryCNN model on EEG data with MLflow tracking.

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
        experiment_name (str, optional): MLflow experiment name. Defaults to "motor-imagery-cnn".
        run_name (str, optional): MLflow run name. Defaults to timestamp-based name.

    Returns:
        MotorImageryCNN: Trained model.
    """
    # Set up device for training
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MLflow experiment tracking
    mlflow.set_experiment(experiment_name)
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"cnn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run to track this training session
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters for reproducibility
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs, 
            "learning_rate": lr,
            "input_shape": str(input_shape),
            "n_classes": n_classes,
            "device": str(device)
        })
    
        # Convert numpy arrays to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Add channel dimension if needed
        if X_train.ndim == 3:
            X_train = X_train.unsqueeze(1)
            X_val = X_val.unsqueeze(1)
        
        # Create datasets and data loaders
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        # Initialize model, loss function, and optimizer
        model = MotorImageryCNN(input_shape, n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Enable MLflow autologging for PyTorch (logs model architecture, parameters)
        mlflow.pytorch.autolog()
        
        best_val_acc = 0
        
        # Training loop with MLflow metric logging
        for epoch in range(epochs):
            # Training phase
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
            
            # Validation phase
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
            
            # Log metrics to MLflow for each epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "epoch": epoch + 1
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model locally and log to MLflow
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pt')
                
                # Log the best model to MLflow
                mlflow.pytorch.log_model(
                    model, 
                    "best_model",
                    registered_model_name="MotorImageryCNN"
                )
        
        # Log final metrics
        mlflow.log_metrics({
            "best_val_accuracy": best_val_acc,
            "total_epochs": epochs
        })
        
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
