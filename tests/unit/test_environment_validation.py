"""
test_environment_validation.py
------------------------------
Test environment setup and AWS credentials validation.
"""

import pytest
import os
import boto3
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Load environment variables from .env file
load_dotenv()


class TestEnvironmentValidation:
    
    def test_required_environment_variables(self):
        """Test that all required environment variables are set."""
        required_vars = [
            'MLFLOW_TRACKING_URI',
            'SAGEMAKER_ROLE',
            'SAGEMAKER_INSTANCE_TYPE',
            'AWS_DEFAULT_REGION',
            'SAGEMAKER_S3_BUCKET',
            'ECR_REPOSITORY_NAME'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.skip(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Verify values are not empty
        for var in required_vars:
            value = os.getenv(var)
            assert value and len(value.strip()) > 0, f"Environment variable {var} is empty"
    
    def test_aws_region_format(self):
        """Test that AWS region follows expected format."""
        region = os.getenv('AWS_DEFAULT_REGION')
        
        if not region:
            pytest.skip("AWS_DEFAULT_REGION not set")
        
        # Basic region format validation
        assert len(region) >= 9, f"Region {region} too short"
        assert '-' in region, f"Region {region} should contain hyphens"
        
        # Should match pattern like: us-east-1, ap-northeast-1, etc.
        parts = region.split('-')
        assert len(parts) >= 3, f"Region {region} should have at least 3 parts"
    
    def test_sagemaker_role_arn_format(self):
        """Test that SageMaker role ARN is properly formatted."""
        role_arn = os.getenv('SAGEMAKER_ROLE')
        
        if not role_arn:
            pytest.skip("SAGEMAKER_ROLE not set")
        
        # ARN format validation
        assert role_arn.startswith('arn:aws:iam::'), f"Role ARN {role_arn} has invalid format"
        assert ':role/' in role_arn, f"Role ARN {role_arn} missing role path"
        
        # Extract account ID from ARN
        arn_parts = role_arn.split(':')
        assert len(arn_parts) >= 6, f"Role ARN {role_arn} has insufficient parts"
        
        account_id = arn_parts[4]
        assert account_id.isdigit(), f"Account ID {account_id} should be numeric"
        assert len(account_id) == 12, f"Account ID {account_id} should be 12 digits"
    
    def test_mlflow_tracking_uri_format(self):
        """Test that MLflow tracking URI is properly formatted."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not tracking_uri:
            pytest.skip("MLFLOW_TRACKING_URI not set")
        
        assert tracking_uri.startswith('https://'), f"MLflow URI {tracking_uri} should use HTTPS"
        assert '.sagemaker.aws' in tracking_uri, f"MLflow URI {tracking_uri} should be SageMaker endpoint"
        assert len(tracking_uri) > 20, f"MLflow URI {tracking_uri} seems too short"
    
    def test_s3_bucket_name_format(self):
        """Test that S3 bucket name follows AWS naming rules."""
        bucket_name = os.getenv('SAGEMAKER_S3_BUCKET')
        
        if not bucket_name:
            pytest.skip("SAGEMAKER_S3_BUCKET not set")
        
        # AWS S3 bucket naming rules
        assert 3 <= len(bucket_name) <= 63, f"Bucket name {bucket_name} length invalid"
        assert bucket_name.islower(), f"Bucket name {bucket_name} should be lowercase"
        assert not bucket_name.startswith('-'), f"Bucket name {bucket_name} cannot start with hyphen"
        assert not bucket_name.endswith('-'), f"Bucket name {bucket_name} cannot end with hyphen"
        assert '..' not in bucket_name, f"Bucket name {bucket_name} cannot contain consecutive dots"
    
    @patch('boto3.client')
    def test_aws_credentials_configured(self, mock_boto_client):
        """Test that AWS credentials are properly configured."""
        # Mock successful STS call
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {
            'Account': '123456789012',
            'UserId': 'AIDACKCEVSQ6C2EXAMPLE',
            'Arn': 'arn:aws:iam::123456789012:user/testuser'
        }
        mock_boto_client.return_value = mock_sts
        
        try:
            sts_client = boto3.client('sts')
            response = sts_client.get_caller_identity()
            
            assert 'Account' in response
            assert 'UserId' in response
            assert 'Arn' in response
            
            mock_boto_client.assert_called_with('sts')
            mock_sts.get_caller_identity.assert_called_once()
            
        except (NoCredentialsError, ClientError) as e:
            pytest.skip(f"AWS credentials not configured: {e}")
    
    def test_sagemaker_instance_type_valid(self):
        """Test that SageMaker instance type is valid."""
        instance_type = os.getenv('SAGEMAKER_INSTANCE_TYPE')
        
        if not instance_type:
            pytest.skip("SAGEMAKER_INSTANCE_TYPE not set")
        
        # Valid SageMaker instance type patterns
        valid_prefixes = ['ml.m5', 'ml.m4', 'ml.c5', 'ml.c4', 'ml.p3', 'ml.p2', 'ml.g4dn']
        valid_sizes = ['large', 'xlarge', '2xlarge', '4xlarge', '8xlarge', '12xlarge', '16xlarge', '24xlarge']
        
        assert instance_type.startswith('ml.'), f"Instance type {instance_type} should start with 'ml.'"
        
        # Check if it matches expected pattern
        instance_family = '.'.join(instance_type.split('.')[:2])  # e.g., ml.p3
        instance_size = instance_type.split('.')[-1]  # e.g., 2xlarge
        
        assert any(instance_family.startswith(prefix) for prefix in valid_prefixes), \
            f"Instance family {instance_family} not in valid list"
        assert instance_size in valid_sizes, f"Instance size {instance_size} not in valid list"
    
    def test_hpo_configuration_values(self):
        """Test that HPO configuration values are reasonable."""
        max_parallel = os.getenv('HPO_MAX_PARALLEL_JOBS')
        max_total = os.getenv('HPO_MAX_TOTAL_JOBS')
        
        if max_parallel:
            parallel_jobs = int(max_parallel)
            assert 1 <= parallel_jobs <= 10, f"Max parallel jobs {parallel_jobs} should be 1-10"
        
        if max_total:
            total_jobs = int(max_total)
            assert 1 <= total_jobs <= 100, f"Max total jobs {total_jobs} should be 1-100"
            
            if max_parallel:
                assert total_jobs >= parallel_jobs, \
                    f"Total jobs {total_jobs} should be >= parallel jobs {parallel_jobs}"
    
    def test_ecr_repository_name_format(self):
        """Test that ECR repository name follows AWS naming rules."""
        repo_name = os.getenv('ECR_REPOSITORY_NAME')
        
        if not repo_name:
            pytest.skip("ECR_REPOSITORY_NAME not set")
        
        # AWS ECR repository naming rules
        assert 2 <= len(repo_name) <= 256, f"Repository name {repo_name} length invalid"
        assert repo_name.replace('-', '').replace('_', '').replace('/', '').isalnum(), \
            f"Repository name {repo_name} contains invalid characters"
        assert not repo_name.startswith('-'), f"Repository name {repo_name} cannot start with hyphen"
        assert not repo_name.endswith('-'), f"Repository name {repo_name} cannot end with hyphen"