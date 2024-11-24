from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
import os
import sagemaker
import boto3
from datetime import datetime

# ECR 이미지 URI
image_uri = "..."

# 하이픈 제거한 Training Job Name 생성
training_job_name = f"llmproj{datetime.now().strftime('%Y%m%d%H%M%S')}"

# 리전명시
session = sagemaker.Session(
    boto_session=boto3.Session(region_name="...")
)

estimator = PyTorch(
    image_uri=image_uri,
    entry_point='main.py',       # 실행할 스크립트
    source_dir = os.path.normpath(r"..."), # 로컬 경로
    role='...', # IAM 역할
    instance_count=4,           # 인스턴스 개수 (분산 학습 필요)
    instance_type='ml.p3.16xlarge',  # EC2 인스턴스 유형
    framework_version='1.9.1',
    py_version='py310',
    job_name = training_job_name,
    distribution={
        "smdistributed": {
            "dataparallel": {
                "enabled": True
            }
        }} ,
    output_path='...',
    hyperparameters={
        'epochs': 4,
        'learning_rate': 0.001
    }
)

#학습 작업 실행
estimator.fit({
    'train': '...'
},
job_name=training_job_name # 명시적으로 job_name 전달
)
