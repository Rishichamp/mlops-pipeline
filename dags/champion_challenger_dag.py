from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys, os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

def task_get_champion_metrics(**context):
    os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
    import mlflow

    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions('iris-classifier')
        prod_versions = [v for v in versions if v.tags.get('stage') == 'Production']
        if not prod_versions:
            prod_versions = versions

        latest = sorted(prod_versions, key=lambda v: int(v.version))[-1]
        run    = client.get_run(latest.run_id)
        metrics = {
            'version':  latest.version,
            'accuracy': run.data.metrics.get('accuracy', 0),
            'f1_macro': run.data.metrics.get('f1_macro', 0),
            'auc_ovr':  run.data.metrics.get('auc_ovr',  0),
        }
        context['ti'].xcom_push(key='champion_metrics', value=metrics)
        print(f'Champion: version={metrics["version"]}  accuracy={metrics["accuracy"]:.4f}  F1={metrics["f1_macro"]:.4f}')
    except Exception as e:
        print(f'Could not get champion metrics: {e}')
        context['ti'].xcom_push(key='champion_metrics', value={'version': 'unknown', 'accuracy': 0, 'f1_macro': 0})

def task_train_challenger(**context):
    os.environ['MLFLOW_TRACKING_URI']    = 'http://mlflow:5000'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    os.environ['AWS_ACCESS_KEY_ID']      = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY']  = 'minioadmin'
    os.environ['AWS_DEFAULT_REGION']     = 'us-east-1'
    os.environ['GIT_PYTHON_REFRESH']     = 'quiet'

    from src.training.train_mlflow import (
        setup_mlflow, load_features, run_hyperparameter_search,
        get_best_run, register_best_model
    )
    from sklearn.model_selection import train_test_split

    setup_mlflow()
    X, y, cols = load_features('/opt/airflow/data/features/features.parquet')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=99, stratify=y)

    run_ids         = run_hyperparameter_search(X_train, X_test, y_train, y_test, cols)
    best_run_id, f1 = get_best_run(run_ids)
    version         = register_best_model(best_run_id)

    import mlflow
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(best_run_id)
    challenger_metrics = {
        'version':  version,
        'run_id':   best_run_id,
        'accuracy': run.data.metrics.get('accuracy', 0),
        'f1_macro': float(f1),
        'auc_ovr':  run.data.metrics.get('auc_ovr', 0),
    }
    context['ti'].xcom_push(key='challenger_metrics', value=challenger_metrics)
    print(f'Challenger trained: version={version}  F1={f1:.4f}')

def task_compare(**context):
    champ  = context['ti'].xcom_pull(key='champion_metrics',   task_ids='get_champion')
    chall  = context['ti'].xcom_pull(key='challenger_metrics', task_ids='train_challenger')

    champ_f1 = float(champ.get('f1_macro', 0))
    chall_f1 = float(chall.get('f1_macro', 0))
    improvement = (chall_f1 - champ_f1) / max(champ_f1, 1e-9) * 100

    print(f'Champion  F1={champ_f1:.4f}')
    print(f'Challenger F1={chall_f1:.4f}')
    print(f'Improvement: {improvement:.2f}%')

    context['ti'].xcom_push(key='challenger_wins', value=chall_f1 > champ_f1)
    context['ti'].xcom_push(key='improvement_pct', value=round(improvement, 2))

    if chall_f1 > champ_f1:
        print('CHALLENGER WINS — will promote')
        return 'promote_challenger'
    print('CHAMPION HOLDS — no change')
    return 'keep_champion'

def task_promote(**context):
    os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
    import mlflow

    chall   = context['ti'].xcom_pull(key='challenger_metrics', task_ids='train_challenger')
    version = chall['version']
    client  = mlflow.tracking.MlflowClient()
    client.set_model_version_tag('iris-classifier', version, 'stage', 'Production')
    print(f'Challenger v{version} promoted to Production')

def task_keep(**context):
    champ = context['ti'].xcom_pull(key='champion_metrics', task_ids='get_champion')
    impr  = context['ti'].xcom_pull(key='improvement_pct', task_ids='compare')
    print(f'Champion v{champ["version"]} retained. Challenger improvement: {impr:.1f}%')

def task_report(**context):
    champ  = context['ti'].xcom_pull(key='champion_metrics',   task_ids='get_champion')
    chall  = context['ti'].xcom_pull(key='challenger_metrics', task_ids='train_challenger')
    winner = context['ti'].xcom_pull(key='challenger_wins',    task_ids='compare')
    impr   = context['ti'].xcom_pull(key='improvement_pct',   task_ids='compare')
    print('=' * 50)
    print('CHAMPION vs CHALLENGER REPORT')
    print('=' * 50)
    print(f'Champion   v{champ.get("version")}  F1={champ.get("f1_macro",0):.4f}  acc={champ.get("accuracy",0):.4f}')
    print(f'Challenger v{chall.get("version")}  F1={chall.get("f1_macro",0):.4f}  acc={chall.get("accuracy",0):.4f}')
    print(f'Winner:     {"CHALLENGER" if winner else "CHAMPION"}')
    print(f'Improvement:{impr:.1f}%')
    print('=' * 50)

with DAG(
    dag_id='champion_challenger_dag',
    default_args=default_args,
    description='Weekly champion vs challenger comparison and auto-promotion',
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'ab-testing', 'champion-challenger'],
) as dag:

    get_champ   = PythonOperator(task_id='get_champion',       python_callable=task_get_champion_metrics)
    train_chall = PythonOperator(task_id='train_challenger',   python_callable=task_train_challenger)
    compare     = BranchPythonOperator(task_id='compare',     python_callable=task_compare)
    promote     = PythonOperator(task_id='promote_challenger', python_callable=task_promote)
    keep        = PythonOperator(task_id='keep_champion',      python_callable=task_keep)
    report      = PythonOperator(task_id='report',             python_callable=task_report,
                                  trigger_rule='none_failed_min_one_success')

    [get_champ, train_chall] >> compare >> [promote, keep] >> report
