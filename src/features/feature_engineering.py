import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_COL = 'target'

def engineer_features(
    df: pd.DataFrame,
    scaler_path: str = 'artifacts/scaler.pkl',
    fit_scaler: bool = True
) -> pd.DataFrame:
    logger.info('Engineering features...')
    features = df.copy()

    # Derived features — domain knowledge applied
    features['sepal_ratio'] = features['sepal_length'] / features['sepal_width']
    features['petal_ratio'] = features['petal_length'] / features['petal_width']
    features['petal_area'] = features['petal_length'] * features['petal_width']
    features['sepal_area'] = features['sepal_length'] * features['sepal_width']

    all_feature_cols = FEATURE_COLS + ['sepal_ratio', 'petal_ratio', 'petal_area', 'sepal_area']

    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        features[all_feature_cols] = scaler.fit_transform(features[all_feature_cols])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f'Scaler fitted and saved to {scaler_path}')
    else:
        scaler = joblib.load(scaler_path)
        features[all_feature_cols] = scaler.transform(features[all_feature_cols])
        logger.info('Scaler loaded and applied')

    result = features[all_feature_cols + [TARGET_COL, 'species']].copy()
    logger.info(f'Feature engineering complete. Shape: {result.shape}')
    return result

def save_features(df: pd.DataFrame, output_path: str = 'data/features/features.parquet'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f'Features saved to {output_path} ({len(df)} rows)')

def load_features(path: str = 'data/features/features.parquet') -> pd.DataFrame:
    return pd.read_parquet(path)

if __name__ == '__main__':
    from src.ingestion.data_loader import load_raw_data
    from src.ingestion.validator import validate_raw_data
    df = load_raw_data()
    df = validate_raw_data(df)
    features = engineer_features(df)
    save_features(features)
    print(features.head())
    print(f'Features shape: {features.shape}')
    print(f'Columns: {list(features.columns)}')
