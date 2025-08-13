"""
cnn.py
------
Defines a PyTorch CNN model for EEG motor imagery classification, exactly replicating 
the architecture from "A Simplified CNN Classification Method for MI-EEG via the Electrode Pairs Signals"
by Lun et al. (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotorImageryCNN(nn.Module):
    """
    Simplified CNN for EEG motor imagery classification using electrode pairs.
    
    Adapted from Lun et al. (2020) architecture for shorter epochs:
    - Input: 40 × 2 (40 time points, 2 electrodes from symmetric pair)
    - Modified architecture for 0.25-second epochs instead of 4-second epochs
    - Separated temporal and spatial filters
    - Reduced pooling to accommodate smaller input size
    - Spatial dropout and batch normalization
    - Leaky ReLU activation
    - 4-class output (T1, T2, T3, T4)
    
    Args:
        n_classes (int): Number of output classes (default: 4 for T1, T2, T3, T4).
    """
    def __init__(self, n_classes=4):
        super(MotorImageryCNN, self).__init__()
        
        # Input shape: (batch_size, 1, 40, 2) - adapted for 0.25-second epochs
        # Modified architecture for smaller temporal dimension
        
        # L1: Temporal convolution along time axis
        # Kernel: [5, 1], Output: (batch, 25, 36, 2) 
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(5, 1), stride=1, padding=0)
        self.spatial_dropout1 = nn.Dropout2d(0.5)
        
        # L2: Spatial convolution along electrode axis  
        # Kernel: [1, 2], Output: (batch, 25, 36, 1)
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(1, 2), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(25)
        
        # L3: Max-Pooling1
        # Kernel: [2, 1], Stride: [2, 1], Output: (batch, 25, 18, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # L4: Temporal convolution
        # Kernel: [5, 1], Output: (batch, 50, 14, 1)
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(5, 1), stride=1, padding=0)
        self.spatial_dropout3 = nn.Dropout2d(0.5)
        
        # L5: Max-Pooling2
        # Kernel: [2, 1], Stride: [2, 1], Output: (batch, 50, 7, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # L6: Temporal convolution
        # Kernel: [3, 1], Output: (batch, 100, 5, 1)
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(3, 1), stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(100)
        self.spatial_dropout4 = nn.Dropout2d(0.5)
        
        # L7: Final temporal convolution
        # Kernel: [3, 1], Output: (batch, 200, 3, 1)
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(3, 1), stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(200)
        
        # L8: Global Average Pooling to handle remaining temporal dimension
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # L9: Flatten + Fully Connected
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(200, n_classes)  # 200 features after global pooling
        
    def forward(self, x):
        """
        Forward pass adapted for 0.25-second epochs.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, 40, 2)
        Returns:
            torch.Tensor: Output logits for each class (batch, n_classes)
        """
        # L1: Temporal convolution + Activation + Spatial Dropout
        x = self.conv1(x)  # (batch, 25, 36, 2)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.spatial_dropout1(x)
        
        # L2: Spatial convolution + Batch Normalization + Activation  
        x = self.conv2(x)  # (batch, 25, 36, 1)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        # L3: Max-Pooling1
        x = self.pool1(x)  # (batch, 25, 18, 1)
        
        # L4: Temporal convolution + Activation + Spatial Dropout
        x = self.conv3(x)  # (batch, 50, 14, 1)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.spatial_dropout3(x)
        
        # L5: Max-Pooling2
        x = self.pool2(x)  # (batch, 50, 7, 1)
        
        # L6: Temporal convolution + Batch Normalization + Activation + Spatial Dropout
        x = self.conv4(x)  # (batch, 100, 5, 1)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.spatial_dropout4(x)
        
        # L7: Final temporal convolution + Batch Normalization + Activation
        x = self.conv5(x)  # (batch, 200, 3, 1)
        x = self.bn5(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        # L8: Global Average Pooling
        x = self.global_avg_pool(x)  # (batch, 200, 1, 1)
        
        # L9: Flatten + Fully Connected
        x = self.flatten(x)  # (batch, 200)
        x = self.fc(x)  # (batch, n_classes)
        
        return x
