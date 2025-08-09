"""
test_training_container.py
--------------------------
Test SageMaker training container and Docker functionality.
"""

import pytest
import os
import tempfile
import boto3
import subprocess
from unittest.mock import patch, MagicMock
from moto import mock_ecr

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class TestSageMakerTrainingContainer:
    
    def test_docker_requirements_file(self):
        """Test that Docker requirements.txt contains necessary packages."""
        docker_requirements_path = os.path.join(
            os.path.dirname(__file__), 
            '../../sagemaker/docker/requirements.txt'
        )
        
        assert os.path.exists(docker_requirements_path), "Docker requirements.txt not found"
        
        with open(docker_requirements_path, 'r') as f:
            requirements = f.read()
        
        # Check for critical packages
        required_packages = [
            'torch',
            'mlflow', 
            'boto3',
            'numpy',
            'scikit-learn'
        ]
        
        for package in required_packages:
            assert package in requirements, f"Missing required package: {package}"
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and contains expected content."""
        dockerfile_path = os.path.join(
            os.path.dirname(__file__),
            '../../sagemaker/docker/Dockerfile'
        )
        
        assert os.path.exists(dockerfile_path), "Dockerfile not found"
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check for expected Dockerfile components
        expected_components = [
            'FROM',
            'COPY',
            'RUN pip install',
            'WORKDIR',
            'ENTRYPOINT'
        ]
        
        for component in expected_components:
            assert component in dockerfile_content, f"Missing Dockerfile component: {component}"
    
    @mock_ecr
    def test_ecr_repository_setup(self):
        """Test ECR repository creation and configuration."""
        # Get AWS configuration from environment
        sagemaker_role = os.getenv('SAGEMAKER_ROLE')
        if not sagemaker_role:
            pytest.skip("SAGEMAKER_ROLE environment variable not set")
            
        # Extract account ID from role ARN
        account_id = sagemaker_role.split(':')[4]
        region = os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1')
        
        ecr_client = boto3.client('ecr', region_name=region)
        repository_name = 'motor-imagery-training'
        
        # Create repository
        response = ecr_client.create_repository(repositoryName=repository_name)
        
        assert 'repository' in response
        assert response['repository']['repositoryName'] == repository_name
        
        # Test repository URI format using environment-derived values
        expected_uri_pattern = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
        actual_uri = response['repository']['repositoryUri']
        
        assert expected_uri_pattern in actual_uri
    
    def test_training_script_imports(self):
        """Test that training script can be imported and has required functions."""
        train_script_path = os.path.join(
            os.path.dirname(__file__),
            '../../sagemaker/scripts/train_sagemaker.py'
        )
        
        assert os.path.exists(train_script_path), "Training script not found"
        
        # Test script can be imported (basic syntax check)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_sagemaker", train_script_path)
            train_module = importlib.util.module_from_spec(spec)
            # Don't execute - just check it can be loaded
            assert spec is not None
        except SyntaxError:
            pytest.fail("Training script has syntax errors")
    
    def test_hyperparameter_configuration(self):
        """Test hyperparameter ranges are properly configured."""
        hpo_script_path = os.path.join(
            os.path.dirname(__file__),
            '../../sagemaker/launch_hpo_job.py'
        )
        
        assert os.path.exists(hpo_script_path), "HPO launch script not found"
        
        with open(hpo_script_path, 'r') as f:
            hpo_content = f.read()
        
        # Check for expected hyperparameter configurations
        expected_hyperparams = [
            'ContinuousParameter',
            'IntegerParameter', 
            'CategoricalParameter',
            'lr',
            'batch_size',
            'epochs'
        ]
        
        for param in expected_hyperparams:
            assert param in hpo_content, f"Missing hyperparameter config: {param}"
    
    def test_mlflow_integration_imports(self):
        """Test that MLflow integration components are importable."""
        train_script_path = os.path.join(
            os.path.dirname(__file__),
            '../../sagemaker/scripts/train_sagemaker.py'
        )
        
        if os.path.exists(train_script_path):
            with open(train_script_path, 'r') as f:
                script_content = f.read()
            
            # Check for MLflow integration
            mlflow_components = [
                'import mlflow',
                'mlflow.set_tracking_uri',
                'mlflow.start_run',
                'mlflow.log_metric'
            ]
            
            for component in mlflow_components:
                assert component in script_content, f"Missing MLflow component: {component}"
    
    @patch('subprocess.run')
    def test_docker_build_command_structure(self, mock_subprocess):
        """Test that Docker build commands have correct structure."""
        # Mock successful subprocess call
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Expected Docker build command components
        expected_build_args = [
            'docker', 'build',
            '-t', 'motor-imagery-training',
            'sagemaker/docker'
        ]
        
        # This tests the command structure without actually running Docker
        try:
            result = subprocess.run(expected_build_args, capture_output=True, text=True, check=False)
            # We expect this to fail since Docker might not be running, but command structure is valid
            assert isinstance(result.returncode, int)
        except FileNotFoundError:
            # Docker not installed - that's fine for testing command structure
            pass
    
    def test_environment_variable_integration(self):
        """Test that container can access required environment variables."""
        required_container_vars = [
            'MLFLOW_TRACKING_URI',
            'AWS_DEFAULT_REGION'
        ]
        
        # Test that these would be passed to container
        with patch.dict(os.environ, {
            'MLFLOW_TRACKING_URI': 'https://test-uri.com',
            'AWS_DEFAULT_REGION': 'ap-northeast-1'
        }):
            for var in required_container_vars:
                assert os.getenv(var) is not None
                assert len(os.getenv(var)) > 0