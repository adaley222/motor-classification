"""
train_sagemaker.py
------------------
SageMaker-compatible training script for EEG motor imagery CNN.
Follows SageMaker training container conventions.
"""

import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import boto3

# Import model from src
from src.models.cnn import MotorImageryCNN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_s3_or_local(data_dir):
    """
    Load preprocessed data from either S3 or local directory.
    SageMaker will download S3 data to local paths automatically.
    
    Args:
        data_dir (str): Path to directory containing the data files
        
    Returns:
        tuple: (X data array, y labels array, subject_ids array or None)
        
    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data is malformed or empty
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Define required file paths
    x_path = os.path.join(data_dir, 'X_processed.npy')
    y_path = os.path.join(data_dir, 'y_processed.npy')
    subject_ids_path = os.path.join(data_dir, 'subject_ids.npy')
    
    # Check for required files
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"X data file not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y labels file not found: {y_path}")
    
    try:
        X = np.load(x_path)
        y = np.load(y_path)
        
        # Validate data
        if X.size == 0:
            raise ValueError("X data array is empty")
        if y.size == 0:
            raise ValueError("y labels array is empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch between X samples ({X.shape[0]}) and y labels ({y.shape[0]})")
            
        # Load subject IDs if available
        subject_ids = None
        if os.path.exists(subject_ids_path):
            try:
                subject_ids = np.load(subject_ids_path)
                if subject_ids.shape[0] != X.shape[0]:
                    logger.warning(f"Mismatch between subject_ids ({subject_ids.shape[0]}) and X samples ({X.shape[0]})")
                    subject_ids = None
            except Exception as e:
                logger.warning(f"Failed to load subject IDs: {str(e)}")
                subject_ids = None
        
        logger.info(f"Loaded data shape: {X.shape}, labels shape: {y.shape}")
        logger.info(f"Unique labels: {np.unique(y)}")
        
        return X, y, subject_ids
        
    except ValueError as ve:
        raise ValueError(f"Error loading data: {str(ve)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading data: {str(e)}")

def prepare_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train/val/test sets."""
    logger.info("Splitting data into train/val/test sets")
    
    # Convert labels to 0-based indexing if needed
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_mapped, test_size=(test_size + val_size), 
        random_state=random_state, stratify=y_mapped
    )
    
    # Second split: val vs test
    val_relative = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_relative), 
        random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, len(unique_labels)

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch data loaders."""
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    # Add channel dimension if needed
    if X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-5):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save to SageMaker model directory
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
    
    logger.info(f'Best validation accuracy: {best_val_acc:.4f}')
    
    return train_losses, val_accuracies, best_val_acc

def save_model_artifacts(model, input_shape, model_dir):
    """Save model and metadata for SageMaker deployment."""
    
    # Save the complete model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    # Save model metadata
    metadata = {
        'input_shape': input_shape,
        'model_class': 'MotorImageryCNN',
        'framework': 'pytorch'
    }
    
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    # Export to ONNX for inference optimization
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(model_dir, 'model.onnx'),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info("Model exported to ONNX format")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    X, y, subject_ids = load_data_from_s3_or_local(args.data_dir)
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test, n_classes = prepare_data(
        X, y, test_size=args.test_size, val_size=args.val_size
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )
    
    # Define input shape for model
    input_shape = (1, X_train.shape[1], X_train.shape[2])  # (channels, height, width)
    logger.info(f"Input shape: {input_shape}, Number of classes: {n_classes}")
    
    # Initialize model
    model = MotorImageryCNN(input_shape, n_classes).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, device, 
        epochs=args.epochs, lr=args.lr
    )
    
    # Save model artifacts
    save_model_artifacts(model, input_shape, args.model_dir)
    
    # Save training metrics
    metrics = {
        'best_validation_accuracy': best_val_acc,
        'final_train_loss': train_losses[-1],
        'final_val_accuracy': val_accuracies[-1],
        'epochs_trained': args.epochs
    }
    
    with open(os.path.join(args.output_data_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1)
    
    args = parser.parse_args()
    
    main(args)