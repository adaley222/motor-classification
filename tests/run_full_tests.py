#!/usr/bin/env python3
"""
run_full_tests.py
-----------------
Run comprehensive tests for the motor imagery electrode pairs pipeline.
This script validates the entire pipeline before training.
"""

import subprocess
import sys
import os
import torch
import numpy as np
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'-'*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command) if isinstance(command, list) else command}")
    print(f"{'-'*60}")
    
    try:
        if isinstance(command, list):
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print(f"FAILED: {description}")
            print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        print(f"ERROR: {description} - {str(e)}")
        return False

def test_model_architecture():
    """Test the CNN model architecture."""
    print(f"\n{'-'*60}")
    print("Testing CNN Model Architecture")
    print(f"{'-'*60}")
    
    try:
        from src.models.cnn import MotorImageryCNN
        
        # Test model creation
        model = MotorImageryCNN(n_classes=4)
        print("Model creation: SUCCESS")
        
        # Test forward pass with correct input shape
        dummy_input = torch.randn(8, 1, 40, 2)  # (batch, channels, time, electrodes)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Validate output shape
        expected_shape = (8, 4)
        if output.shape == expected_shape:
            print(f"Model forward pass: SUCCESS - Output shape {output.shape}")
        else:
            print(f"Model forward pass: FAILED - Expected {expected_shape}, got {output.shape}")
            return False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"Model architecture test FAILED: {str(e)}")
        return False

def test_mock_data():
    """Test mock data generation."""
    print(f"\n{'-'*60}")
    print("Testing Mock Data Generation")
    print(f"{'-'*60}")
    
    try:
        from tests.fixtures.mock_data import create_mock_eeg_data, create_mock_processed_data
        
        # Test electrode pairs data
        X, y = create_mock_eeg_data(n_samples=100, n_timepoints=40, n_classes=4)
        
        if X.shape != (100, 40, 2):
            print(f"Mock data shape FAILED: Expected (100, 40, 2), got {X.shape}")
            return False
            
        if y.shape != (100,):
            print(f"Mock labels shape FAILED: Expected (100,), got {y.shape}")
            return False
            
        # Test label values
        unique_labels = np.unique(y)
        if not all(label in [2, 3, 4, 5] for label in unique_labels):
            print(f"Mock labels FAILED: Expected T1-T4 (2-5), got {unique_labels}")
            return False
            
        print("Mock electrode pairs data: SUCCESS")
        
        # Test processed data
        X_proc, y_proc, subjects = create_mock_processed_data()
        print(f"Mock processed data: SUCCESS - Shape {X_proc.shape}")
        
        return True
        
    except Exception as e:
        print(f"Mock data test FAILED: {str(e)}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print(f"\n{'-'*60}")
    print("Testing Configuration Loading")
    print(f"{'-'*60}")
    
    try:
        import yaml
        
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print("Config file FAILED: config/config.yaml not found")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key parameters
        required_keys = ['n_classes', 'input_shape', 'tmax', 'event_id', 'electrode_pairs']
        for key in required_keys:
            if key not in config:
                print(f"Config FAILED: Missing key '{key}'")
                return False
        
        # Validate values
        if config['n_classes'] != 4:
            print(f"Config FAILED: Expected 4 classes, got {config['n_classes']}")
            return False
            
        if config['input_shape'] != [1, 40, 2]:
            print(f"Config FAILED: Expected [1, 40, 2] input shape, got {config['input_shape']}")
            return False
            
        if config['tmax'] != 0.25:
            print(f"Config FAILED: Expected 0.25s epochs, got {config['tmax']}")
            return False
            
        print("Configuration loading: SUCCESS")
        print(f"Classes: {config['n_classes']}")
        print(f"Input shape: {config['input_shape']}")
        print(f"Epoch duration: {config['tmax']}s")
        print(f"Electrode pairs: {len(config['electrode_pairs'])}")
        
        return True
        
    except Exception as e:
        print(f"Config loading FAILED: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Motor Imagery Electrode Pairs Pipeline - Full Test Suite")
    print("="*70)
    
    tests = []
    
    # Test 1: Model Architecture
    tests.append(("Model Architecture", test_model_architecture()))
    
    # Test 2: Mock Data Generation
    tests.append(("Mock Data Generation", test_mock_data()))
    
    # Test 3: Configuration Loading
    tests.append(("Configuration Loading", test_config_loading()))
    
    # Test 4: Unit Tests (preprocessing)
    tests.append(("Unit Tests - Preprocessing", 
                 run_command([".venv\\Scripts\\python.exe", "-m", "pytest", "tests/unit/test_preprocessing.py", "-v"], 
                           "Unit tests for preprocessing")))
    
    # Test 5: Import Tests
    import_tests = [
        "from src.models.cnn import MotorImageryCNN",
        "from src.data.preprocess import preprocess_edf_electrode_pairs, ELECTRODE_PAIRS", 
        "from tests.fixtures.mock_data import create_mock_eeg_data"
    ]
    
    import_success = True
    for test in import_tests:
        try:
            exec(test)
            print(f"Import SUCCESS: {test}")
        except Exception as e:
            print(f"Import FAILED: {test} - {str(e)}")
            import_success = False
    
    tests.append(("Import Tests", import_success))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\nALL TESTS PASSED! Pipeline is ready for training.")
        return True
    else:
        print(f"\n{failed} TEST(S) FAILED! Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)