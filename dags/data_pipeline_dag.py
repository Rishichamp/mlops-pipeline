from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False,
}

def task_ingest(**context):
    from src.ingestion.data_loader import load_raw_data
    df = load_raw_data(output_path='/opt/airflow/data/raw/iris.csv')
    context['ti'].xcom_push(key='row_count', value=len(df))
    print(f'Ingested {len(df)} rows')

def task_validate(**context):
    from src.ingestion.validator import validate_from_csv
    import pandas as pd
    df = validate_from_csv('/opt/airflow/data/raw/iris.csv')
    print(f'Validation passed for {len(df)} rows')

def task_engineer(**context):
    from src.ingestion.validator import validate_from_csv
    from src.features.feature_engineering import engineer_features, save_features
    df = validate_from_csv('/opt/airflow/data/raw/iris.csv')
    features = engineer_features(
        df,
        scaler_path='/opt/airflow/artifacts/scaler.pkl',
        fit_scaler=True
    )
    save_features(features, output_path='/opt/airflow/data/features/features.parquet')
    print(f'Features engineered: {features.shape}')

def task_upload(**context):
    from src.ingestion.minio_client import upload_file, ensure_bucket
    from datetime import datetime
    ensure_bucket('mlops-features')
    date_str = datetime.now().strftime('%Y%m%d')
    upload_file(
        local_path='/opt/airflow/data/features/features.parquet',
        bucket='mlops-features',
        key=f'iris/features_{date_str}.parquet'
    )
    print('Features uploaded to MinIO')

def task_trigger_training(**context):
    print('Training trigger: feature pipeline complete')
    print('Phase 3 will replace this with actual MLflow training run')
    row_count = context['ti'].xcom_pull(key='row_count', task_ids='ingest')
    print(f'Training data size: {row_count} rows')

with DAG(
    dag_id='mlops_data_pipeline',
    default_args=default_args,
    description='Ingest, validate, engineer features, upload to MinIO',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'data', 'features'],
) as dag:

    ingest = PythonOperator(
        task_id='ingest',
        python_callable=task_ingest,
    )

    validate = PythonOperator(
        task_id='validate',
        python_callable=task_validate,
    )

    engineer = PythonOperator(
        task_id='engineer_features',
        python_callable=task_engineer,
    )

    upload = PythonOperator(
        task_id='upload_to_minio',
        python_callable=task_upload,
    )

    trigger = PythonOperator(
        task_id='trigger_training',
        python_callable=task_trigger_training,
    )

    ingest >> validate >> engineer >> upload >> trigger
