"""
launch_hpo_job.py
-----------------
Script to configure and launch SageMaker Hyperparameter Optimization job
for the Motor Imagery CNN model with MLflow integration.

This script defines the hyperparameter ranges and creates the HPO job.
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter
)
from sagemaker.inputs import TrainingInput
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize SageMaker session and get execution role
sagemaker_session = sagemaker.Session()

# Get IAM role from environment variable or try to auto-detect
role = os.environ.get('SAGEMAKER_ROLE')
if not role:
    try:
        role = sagemaker.get_execution_role()  # Works inside SageMaker
    except:
        # Must be specified when running outside SageMaker
        raise ValueError(
            "SAGEMAKER_ROLE environment variable is required when running outside SageMaker. "
            "Set it to your SageMaker execution role ARN in your .env file."
        )

# Configuration - Make sure this matches your MLflow server region
REGION = 'ap-northeast-1'  # Match your MLflow server region
ACCOUNT_ID = sagemaker_session.account_id()

def setup_hpo_job():
    """
    Configure and launch the SageMaker HPO job for motor imagery classification.
    """
    
    # Validate required environment variables
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if not mlflow_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI environment variable is required. "
            "Please set it in your .env file."
        )
    
    print(f"Using MLflow tracking URI: {mlflow_uri}")
    print(f"CloudWatch metrics enabled: {os.environ.get('ENABLE_CLOUDWATCH_METRICS', 'false')}")
    
    # Get configurable parameters from environment variables
    instance_type = os.environ.get('SAGEMAKER_INSTANCE_TYPE', 'ml.p3.2xlarge')  # Default to V100 for performance
    max_parallel_jobs = int(os.environ.get('HPO_MAX_PARALLEL_JOBS', '3'))
    max_total_jobs = int(os.environ.get('HPO_MAX_TOTAL_JOBS', '20'))
    
    print(f"Instance type: {instance_type}")
    print(f"HPO configuration: {max_total_jobs} total jobs, {max_parallel_jobs} parallel")
    
    # 1. DEFINE HYPERPARAMETER RANGES
    # This is where YOU specify what SageMaker should try
    hyperparameter_ranges = {
        # Learning rate: Try values between 0.000001 and 0.01
        'lr': ContinuousParameter(1e-6, 1e-2),
        
        # Batch size: Try these specific values
        'batch_size': CategoricalParameter([16, 32, 64, 128]),
        
        # Number of epochs: Try integer values between 20 and 100
        'epochs': IntegerParameter(20, 100),
        
        # Data split ratios: Fine-tune your train/val/test splits
        'test_size': ContinuousParameter(0.15, 0.25),
        'val_size': ContinuousParameter(0.05, 0.15)
    }
    
    # 2. DEFINE FIXED HYPERPARAMETERS
    # These stay the same across all HPO trials
    static_hyperparameters = {
        # You can add any fixed parameters here
        # 'dropout_rate': 0.5,  # If you want to add this later
    }
    
    # 3. CONFIGURE THE TRAINING ESTIMATOR
    # This tells SageMaker how to run your training container
    
    # ECR image URI for your training container
    # Format: {account}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}
    image_repo_name = os.environ.get('ECR_REPOSITORY_NAME', 'motor-imagery-training')
    image_tag = os.environ.get('ECR_IMAGE_TAG', 'latest')
    training_image_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{image_repo_name}:{image_tag}"
    
    print(f"Using Docker image: {training_image_uri}")
    
    estimator = Estimator(
        # Docker image containing your training script
        image_uri=training_image_uri,
        
        # IAM role for SageMaker execution
        role=role,
        
        # Instance configuration (configurable via environment variables)
        instance_count=1,
        instance_type=instance_type,
        
        # Fixed hyperparameters (same for all trials)
        hyperparameters=static_hyperparameters,
        
        # Output configuration
        output_path=f's3://sagemaker-{REGION}-{ACCOUNT_ID}/motor-imagery-hpo-output',
        
        # Environment variables for MLflow integration
        environment={
            # Load MLflow tracking URI from environment variable
            'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', ''),
            'ENABLE_CLOUDWATCH_METRICS': os.environ.get('ENABLE_CLOUDWATCH_METRICS', 'false')
        },
        
        # SageMaker configuration
        sagemaker_session=sagemaker_session
    )
    
    # 4. CONFIGURE THE HPO TUNER
    # This defines the optimization strategy
    tuner = HyperparameterTuner(
        # The estimator to tune
        estimator=estimator,
        
        # OBJECTIVE METRIC: What SageMaker should optimize
        # This must match the metric name you print in train_sagemaker.py
        objective_metric_name='validation_accuracy',
        objective_type='Maximize',  # Try to maximize validation accuracy
        
        # Hyperparameter ranges defined above
        hyperparameter_ranges=hyperparameter_ranges,
        
        # HPO job configuration (configurable via environment variables)
        max_jobs=max_total_jobs,
        max_parallel_jobs=max_parallel_jobs,
        
        # Early stopping to save costs
        early_stopping_type='Auto',
        
        # Job naming
        base_tuning_job_name='motor-imagery-hpo'
    )
    
    # 5. SPECIFY TRAINING DATA LOCATION
    # Point to your preprocessed data in S3
    training_data = TrainingInput(
        s3_data=f's3://sagemaker-{REGION}-{ACCOUNT_ID}/motor-imagery-data/processed/',
        content_type='application/x-npy'  # Since we're using .npy files
    )
    
    # 6. LAUNCH THE HPO JOB
    job_name = f"motor-imagery-hpo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    print(f"Launching HPO job: {job_name}")
    print(f"Hyperparameter ranges: {hyperparameter_ranges}")
    print(f"Max jobs: {tuner.max_jobs}, Max parallel: {tuner.max_parallel_jobs}")
    print(f"Objective: {tuner.objective_type} {tuner.objective_metric_name}")
    
    # This starts the HPO job!
    tuner.fit(
        inputs={'training': training_data},
        job_name=job_name,
        wait=False  # Don't block - job runs in background
    )
    
    print(f"HPO job launched! Monitor progress in SageMaker console.")
    print(f"Job name: {job_name}")
    
    return tuner, job_name

def monitor_hpo_job(job_name):
    """
    Helper function to check HPO job status and results.
    """
    sm_client = boto3.client('sagemaker')
    
    try:
        response = sm_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )
        
        status = response['HyperParameterTuningJobStatus']
        print(f"HPO Job Status: {status}")
        
        if 'BestTrainingJob' in response:
            best_job = response['BestTrainingJob']
            print(f"Best Training Job: {best_job['TrainingJobName']}")
            print(f"Best Objective Value: {best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']}")
            print(f"Best Hyperparameters: {best_job['TunedHyperParameters']}")
        
        return response
        
    except Exception as e:
        print(f"Error checking job status: {e}")
        return None

if __name__ == "__main__":
    # Launch the HPO job
    tuner, job_name = setup_hpo_job()
    
    print("\n" + "="*50)
    print("WHAT HAPPENS NEXT:")
    print("="*50)
    print("1. SageMaker will create 20 training jobs")
    print("2. Each job gets different hyperparameters from the ranges you defined")
    print("3. Each job runs: python train_sagemaker.py --lr X --batch-size Y --epochs Z")
    print("4. SageMaker tracks which combination gives best validation_accuracy")
    print("5. All experiments are logged to your MLflow server")
    print("6. After completion, you get the best hyperparameter combination")
    print("="*50)
    
    # Optionally monitor the job
    # Uncomment this if you want to check status programmatically
    # import time
    # time.sleep(60)  # Wait a minute for job to start
    # monitor_hpo_job(job_name)