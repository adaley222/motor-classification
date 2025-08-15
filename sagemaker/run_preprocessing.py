"""
run_preprocessing.py
-------------------
Run SageMaker processing job to download and preprocess PhysioNet EEG data.
This prepares the data for HPO training jobs.
"""

import boto3
import sagemaker
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from dotenv import load_dotenv
import os
import yaml
import argparse

# Load environment variables
load_dotenv()

# Configuration
REGION = 'ap-northeast-1'
sagemaker_session = sagemaker.Session()
role = os.environ.get('SAGEMAKER_ROLE')

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_data_preprocessing():
    """
    Run SageMaker processing job to preprocess EEG data and upload to S3.
    Uses subjects and runs from config.yaml
    """
    
    # Load configuration
    config = load_config()
    subjects = config.get('subjects', [1, 2])
    runs = config.get('runs', [3])
    
    account_id = sagemaker_session.account_id()
    
    # S3 paths for data storage
    s3_bucket = f'sagemaker-{REGION}-{account_id}'
    input_path = f's3://{s3_bucket}/motor-imagery-data/raw'
    output_path = f's3://{s3_bucket}/motor-imagery-data/processed'
    
    print(f"Running data preprocessing...")
    print(f"Subjects: {subjects}")
    print(f"Runs: {runs}")
    print(f"Output will be saved to: {output_path}")
    
    # Create processing job using PyTorch processor
    # This gives us a better Python environment with more scientific libraries
    processor = PyTorchProcessor(
        framework_version='1.12',
        py_version='py38',
        role=role,
        instance_type='ml.m5.xlarge',  # CPU instance sufficient for preprocessing
        instance_count=1,
        base_job_name='motor-imagery-preprocessing',
        sagemaker_session=sagemaker_session
    )
    
    # Convert subjects and runs to strings for arguments
    subject_args = [str(s) for s in subjects]
    run_args = [str(r) for r in runs]
    
    # Run processing job
    processor.run(
        # Use our preprocessing script
        code='scripts/preprocess_sagemaker.py',
        
        # No input data needed - script downloads from PhysioNet
        inputs=[],
        
        # Output processed data to S3
        outputs=[
            ProcessingOutput(
                output_name='processed-data',
                source='/opt/ml/processing/output',
                destination=output_path
            )
        ],
        
        # Processing job arguments - read from config
        arguments=[
            '--subjects'] + subject_args + [
            '--runs'] + run_args + [
            '--s3-bucket', s3_bucket,
            '--s3-prefix', 'motor-imagery-data/processed'
        ],
        
        # Wait for completion
        wait=True,
        logs=True
    )
    
    print("Data preprocessing completed!")
    print(f"Processed data available at: {output_path}")
    
    return output_path

def verify_processed_data(s3_path):
    """
    Verify that the processed data files exist in S3.
    """
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    bucket_name = s3_path.replace('s3://', '').split('/')[0]
    prefix = '/'.join(s3_path.replace('s3://', '').split('/')[1:])
    
    print(f"Verifying processed data in s3://{bucket_name}/{prefix}/")
    
    try:
        # List objects in the processed data directory
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            data_files = [f for f in files if f.endswith('.npy')]
            
            print(f"Found {len(data_files)} data files:")
            for file in data_files:
                # Get file size
                obj_info = s3_client.head_object(Bucket=bucket_name, Key=file)
                size_mb = obj_info['ContentLength'] / (1024 * 1024)
                print(f"   {file.split('/')[-1]} ({size_mb:.1f} MB)")
            
            # Check for required files
            required_files = ['X_processed.npy', 'y_processed.npy', 'subject_ids.npy']
            found_files = [f.split('/')[-1] for f in files]
            
            missing_files = [f for f in required_files if f not in found_files]
            if missing_files:
                print(f"Missing required files: {missing_files}")
                return False
            else:
                print("All required data files present!")
                return True
        else:
            print("No processed data found in S3")
            return False
            
    except Exception as e:
        print(f"Error verifying data: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SAGEMAKER DATA PREPROCESSING")
    print("=" * 60)
    
    if not role:
        print("SAGEMAKER_ROLE environment variable not set")
        print("Please add your SageMaker execution role ARN to your .env file")
        exit(1)
    
    try:
        # Run preprocessing
        output_path = run_data_preprocessing()
        
        # Verify results
        if verify_processed_data(output_path):
            print("\n" + "=" * 60)
            print("PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Data location: {output_path}")
            print("\nNext steps:")
            print("1. Build and push Docker container")
            print("2. Launch HPO job")
        else:
            print("Data verification failed")
            
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        print("Check SageMaker console for detailed logs")