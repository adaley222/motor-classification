#!/usr/bin/env python3
"""
Test script for the new electrode pairs CNN model implementation.
"""

import torch
import numpy as np
from src.models.cnn import MotorImageryCNN

def test_model_architecture():
    """Test the new CNN model with correct input dimensions."""
    print("Testing new MotorImageryCNN architecture...")
    
    # Create model (4 classes: T1, T2, T3, T4)
    model = MotorImageryCNN(n_classes=4)
    
    # Create dummy input: (batch_size, channels, time_points, electrodes)
    # Expected input: (batch_size, 1, 40, 2) for 0.25-second epochs
    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, 40, 2)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 4)")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, 4), f"Expected shape ({batch_size}, 4), got {output.shape}"
        
        print("Model architecture test PASSED!")
        return True
        
    except Exception as e:
        print(f"Model architecture test FAILED: {str(e)}")
        return False

def test_preprocessing_function():
    """Test the preprocessing function import."""
    print("\nTesting preprocessing function import...")
    
    try:
        from src.data.preprocess import process_subjects_electrode_pairs, ELECTRODE_PAIRS
        
        print(f"Preprocessing functions imported successfully")
        print(f"Available electrode pairs: {len(ELECTRODE_PAIRS)}")
        print(f"Electrode pairs: {ELECTRODE_PAIRS}")
        
        return True
        
    except Exception as e:
        print(f"Preprocessing import FAILED: {str(e)}")
        return False

def print_model_summary():
    """Print model summary."""
    print("\nModel Summary:")
    print("="*50)
    
    model = MotorImageryCNN(n_classes=4)
    
    print(f"Model: MotorImageryCNN")
    print(f"Input: (batch, 1, 40, 2) - electrode pairs (0.25s epochs)")
    print(f"Output: (batch, 4) - T1, T2, T3, T4 classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nLayer structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            print(f"  {name}: {module}")

if __name__ == "__main__":
    print("Testing new motor imagery model implementation")
    print("=" * 60)
    
    # Run tests
    architecture_ok = test_model_architecture()
    preprocessing_ok = test_preprocessing_function()
    
    if architecture_ok and preprocessing_ok:
        print_model_summary()
        print("\nAll tests PASSED! Ready to proceed with training.")
    else:
        print("\nSome tests FAILED. Please fix issues before proceeding.")