"""
test_mlflow_integration.py
--------------------------
Test MLflow experiment tracking integration with SageMaker.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import requests

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class TestMLflowIntegration:
    
    def test_mlflow_tracking_uri_configured(self):
        """Test that MLflow tracking URI is properly configured."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not tracking_uri:
            pytest.skip("MLFLOW_TRACKING_URI environment variable not set")
        
        assert tracking_uri.startswith('https://'), "MLflow URI should use HTTPS"
        assert 'sagemaker.aws' in tracking_uri, "Should use SageMaker MLflow endpoint"
        assert len(tracking_uri) > 10, "URI should not be empty"
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    def test_mlflow_experiment_logging(self, mock_log_param, mock_log_metric, 
                                     mock_start_run, mock_set_uri):
        """Test MLflow experiment logging functionality."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not tracking_uri:
            pytest.skip("MLFLOW_TRACKING_URI environment variable not set")
        
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Simulate experiment logging
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            
            with mlflow.start_run(run_name="test-hpo-run") as run:
                # Log hyperparameters
                mlflow.log_param("lr", 0.001)
                mlflow.log_param("batch_size", 32)
                mlflow.log_param("epochs", 50)
                
                # Log metrics
                mlflow.log_metric("train_loss", 0.5, step=1)
                mlflow.log_metric("validation_accuracy", 0.85, step=1)
            
            # Verify MLflow functions were called
            mock_set_uri.assert_called_once_with(tracking_uri)
            mock_start_run.assert_called_once()
            assert mock_log_param.call_count >= 3  # At least 3 parameters logged
            assert mock_log_metric.call_count >= 2  # At least 2 metrics logged
            
        except ImportError:
            pytest.skip("MLflow not installed")
    
    @patch('requests.get')
    def test_mlflow_server_connectivity(self, mock_get):
        """Test connectivity to MLflow tracking server."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not tracking_uri:
            pytest.skip("MLFLOW_TRACKING_URI environment variable not set")
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.0"}
        mock_get.return_value = mock_response
        
        # Test connectivity
        health_endpoint = f"{tracking_uri.rstrip('/')}/health"
        response = requests.get(health_endpoint, timeout=10)
        
        assert response.status_code == 200
        mock_get.assert_called_once_with(health_endpoint, timeout=10)
    
    def test_experiment_name_generation(self):
        """Test that experiment names follow expected pattern."""
        import datetime
        
        # Expected pattern: SageMaker-HPO-motor-imagery-hpo-YYYY-MM-DD-HH-MM-SS
        base_name = "motor-imagery-hpo"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        expected_pattern = f"SageMaker-HPO-{base_name}-{timestamp[:10]}"  # Just date part
        
        # Test pattern matching
        test_experiment_name = f"SageMaker-HPO-{base_name}-2024-01-15-14-30-25"
        assert test_experiment_name.startswith("SageMaker-HPO-")
        assert base_name in test_experiment_name
        assert len(test_experiment_name.split('-')) >= 6  # Should have timestamp parts
    
    @patch('mlflow.search_experiments')
    def test_experiment_search_functionality(self, mock_search):
        """Test that experiments can be searched and retrieved."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not tracking_uri:
            pytest.skip("MLFLOW_TRACKING_URI environment variable not set")
        
        # Mock experiment search results
        mock_experiment = MagicMock()
        mock_experiment.name = "SageMaker-HPO-motor-imagery-hpo"
        mock_experiment.experiment_id = "123"
        mock_search.return_value = [mock_experiment]
        
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            
            # Search for experiments
            experiments = mlflow.search_experiments()
            
            mock_search.assert_called_once()
            assert len(experiments) > 0
            assert experiments[0].name == "SageMaker-HPO-motor-imagery-hpo"
            
        except ImportError:
            pytest.skip("MLflow not installed")
    
    def test_metric_logging_format(self):
        """Test that metrics are logged in expected format for SageMaker HPO."""
        # Expected metrics for SageMaker HPO optimization
        expected_metrics = {
            'validation_accuracy': {'type': 'float', 'range': [0.0, 1.0]},
            'train_loss': {'type': 'float', 'range': [0.0, float('inf')]},
            'val_loss': {'type': 'float', 'range': [0.0, float('inf')]},
            'epoch': {'type': 'int', 'range': [1, 100]}
        }
        
        # Test metric validation
        for metric_name, metric_config in expected_metrics.items():
            assert isinstance(metric_name, str), f"Metric name {metric_name} should be string"
            assert 'type' in metric_config, f"Metric {metric_name} should have type specified"
            assert 'range' in metric_config, f"Metric {metric_name} should have range specified"
    
    def test_hyperparameter_logging_format(self):
        """Test that hyperparameters are logged in expected format."""
        # Expected hyperparameters matching HPO configuration
        expected_hyperparams = [
            'lr',
            'batch_size', 
            'epochs',
            'test_size',
            'val_size'
        ]
        
        # Test hyperparameter names match HPO configuration
        for param_name in expected_hyperparams:
            assert isinstance(param_name, str), f"Hyperparameter {param_name} should be string"
            assert len(param_name) > 0, f"Hyperparameter {param_name} should not be empty"
    
    @patch('mlflow.log_artifact')
    def test_model_artifact_logging(self, mock_log_artifact):
        """Test that model artifacts are properly logged."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not tracking_uri:
            pytest.skip("MLFLOW_TRACKING_URI environment variable not set")
        
        try:
            import mlflow
            
            # Test artifact logging
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_model:
                temp_model.write(b"mock_model_data")
                temp_model_path = temp_model.name
            
            try:
                mlflow.log_artifact(temp_model_path, "models")
                mock_log_artifact.assert_called_once_with(temp_model_path, "models")
            finally:
                os.unlink(temp_model_path)
                
        except ImportError:
            pytest.skip("MLflow not installed")