"""
build_and_push.py
-----------------
Build and push the Docker container for SageMaker training.
This script handles ECR repository creation, Docker build, and image push.
"""

import boto3
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
REGION = 'ap-northeast-1'  # Match your MLflow region
REPOSITORY_NAME = 'motor-imagery-training'
IMAGE_TAG = 'latest'

def get_account_id():
    """Get AWS account ID"""
    sts_client = boto3.client('sts')
    return sts_client.get_caller_identity()['Account']

def create_ecr_repository(repository_name, region):
    """Create ECR repository if it doesn't exist"""
    ecr_client = boto3.client('ecr', region_name=region)
    
    try:
        # Check if repository exists
        ecr_client.describe_repositories(repositoryNames=[repository_name])
        print(f"ECR repository '{repository_name}' already exists")
        
    except ecr_client.exceptions.RepositoryNotFoundException:
        # Create repository
        print(f"Creating ECR repository '{repository_name}'...")
        ecr_client.create_repository(
            repositoryName=repository_name,
            imageScanningConfiguration={'scanOnPush': True}
        )
        print(f"ECR repository '{repository_name}' created successfully")

def get_docker_login_command(region, account_id):
    """Get Docker login command for ECR"""
    ecr_client = boto3.client('ecr', region_name=region)
    
    # Get login token
    response = ecr_client.get_authorization_token()
    token = response['authorizationData'][0]['authorizationToken']
    endpoint = response['authorizationData'][0]['proxyEndpoint']
    
    # Decode token (it's base64 encoded username:password)
    import base64
    username, password = base64.b64decode(token).decode().split(':')
    
    return f"docker login --username {username} --password {password} {endpoint}"

def build_and_push_image():
    """Build and push Docker image to ECR"""
    
    account_id = get_account_id()
    print(f"Building and pushing Docker image for account: {account_id}")
    
    # Create ECR repository
    create_ecr_repository(REPOSITORY_NAME, REGION)
    
    # Docker image URI
    image_uri = f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{REPOSITORY_NAME}:{IMAGE_TAG}"
    
    print(f"Target image URI: {image_uri}")
    
    # Change to sagemaker directory for Docker build context
    os.chdir('sagemaker')
    
    try:
        # Step 1: Build Docker image
        print("Building Docker image...")
        build_cmd = f"docker build -t {REPOSITORY_NAME}:{IMAGE_TAG} -f docker/Dockerfile ."
        result = subprocess.run(build_cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Docker build failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
        print("Docker image built successfully")
        
        # Step 2: Tag image for ECR
        print("Tagging image for ECR...")
        tag_cmd = f"docker tag {REPOSITORY_NAME}:{IMAGE_TAG} {image_uri}"
        result = subprocess.run(tag_cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Docker tag failed: {result.stderr}")
            return False
            
        # Step 3: Login to ECR
        print("Logging into ECR...")
        login_cmd = get_docker_login_command(REGION, account_id)
        result = subprocess.run(login_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ECR login failed: {result.stderr}")
            return False
            
        print("ECR login successful")
        
        # Step 4: Push image
        print("Pushing image to ECR...")
        push_cmd = f"docker push {image_uri}"
        result = subprocess.run(push_cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Docker push failed: {result.stderr}")
            return False
            
        print("Image pushed to ECR successfully!")
        print(f"Image URI: {image_uri}")
        
        return image_uri
        
    except Exception as e:
        print(f"Build and push failed: {e}")
        return False
    
    finally:
        # Return to original directory
        os.chdir('..')

def verify_image_exists(image_uri):
    """Verify that the image exists in ECR"""
    try:
        ecr_client = boto3.client('ecr', region_name=REGION)
        account_id = get_account_id()
        
        response = ecr_client.describe_images(
            repositoryName=REPOSITORY_NAME,
            imageIds=[{'imageTag': IMAGE_TAG}]
        )
        
        if response['imageDetails']:
            image_size_mb = response['imageDetails'][0]['imageSizeInBytes'] / (1024 * 1024)
            push_time = response['imageDetails'][0]['imagePushedAt']
            print(f"Image verified in ECR:")
            print(f"   Size: {image_size_mb:.1f} MB")
            print(f"   Pushed: {push_time}")
            return True
        else:
            print("Image not found in ECR")
            return False
            
    except Exception as e:
        print(f"Image verification failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SAGEMAKER DOCKER BUILD AND PUSH")
    print("=" * 60)
    
    # Build and push
    image_uri = build_and_push_image()
    
    if image_uri:
        # Verify image exists
        if verify_image_exists(image_uri):
            print("\n" + "=" * 60)
            print("BUILD AND PUSH COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Image URI: {image_uri}")
            print("\nNext steps:")
            print("1. Update your launch_hpo_job.py with this image URI")
            print("2. Run data preprocessing")
            print("3. Launch HPO job")
        else:
            print("Image verification failed")
    else:
        print("Build and push failed")