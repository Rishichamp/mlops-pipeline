import boto3
from botocore.client import Config
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://localhost:9000'),
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def ensure_bucket(bucket: str):
    client = get_minio_client()
    try:
        client.head_bucket(Bucket=bucket)
        logger.info(f'Bucket {bucket} exists')
    except Exception:
        client.create_bucket(Bucket=bucket)
        logger.info(f'Bucket {bucket} created')

def upload_file(local_path: str, bucket: str, key: str):
    client = get_minio_client()
    ensure_bucket(bucket)
    client.upload_file(local_path, bucket, key)
    logger.info(f'Uploaded {local_path} to s3://{bucket}/{key}')

def download_file(bucket: str, key: str, local_path: str):
    client = get_minio_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.download_file(bucket, key, local_path)
    logger.info(f'Downloaded s3://{bucket}/{key} to {local_path}')

if __name__ == '__main__':
    ensure_bucket('mlops-features')
    ensure_bucket('mlops-models')
    ensure_bucket('mlflow-artifacts')
    print('All buckets ready')
