"""
cnn.py
------
Defines a PyTorch CNN model for EEG motor imagery classification, matching the architecture described in the project notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotorImageryCNN(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(MotorImageryCNN, self).__init__()
        # input_shape: (channels, height, width) = (1, sample_size, 64)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=0)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.dropout3 = nn.Dropout(0.5)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.pool5 = nn.MaxPool2d(2)
        # Compute the output size after all conv/pool layers
        self._to_linear = self._get_conv_output(input_shape)
        self.fc = nn.Linear(self._to_linear, n_classes)

    def _get_conv_output(self, shape):
        # shape: (channels, height, width)
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
