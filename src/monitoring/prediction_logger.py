import json
import os
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_LOG_PATH = 'data/predictions/predictions.jsonl'

def log_prediction(features: list, prediction: int, confidence: float):
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    record = {
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'prediction': prediction,
        'confidence': confidence,
    }
    with open(PREDICTION_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')

def load_predictions_as_df(path: str = PREDICTION_LOG_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    features_df = pd.DataFrame(df['features'].tolist(),
        columns=['sepal_length','sepal_width','petal_length','petal_width'])
    return pd.concat([features_df, df[['timestamp','prediction','confidence']]], axis=1)

def upload_predictions_to_minio():
    from src.ingestion.minio_client import upload_file, ensure_bucket
    ensure_bucket('mlops-predictions')
    date_str = datetime.now().strftime('%Y%m%d')
    upload_file(
        local_path=PREDICTION_LOG_PATH,
        bucket='mlops-predictions',
        key=f'logs/predictions_{date_str}.jsonl'
    )
    logger.info(f'Uploaded predictions to MinIO')

if __name__ == '__main__':
    log_prediction([5.1, 3.5, 1.4, 0.2], 0, 0.97)
    log_prediction([6.3, 3.3, 6.0, 2.5], 2, 0.89)
    log_prediction([5.7, 2.8, 4.5, 1.3], 1, 0.78)
    df = load_predictions_as_df()
    print(df)
    print(f'Logged {len(df)} predictions')
