import pandas as pd
import numpy as np
import os
import logging
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(output_path: str = 'data/raw/iris.csv') -> pd.DataFrame:
    logger.info('Loading Iris dataset...')
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(label_map)

    # Add realistic noise to simulate real data
    np.random.seed(42)
    df['sepal_length'] = df['sepal_length'] + np.random.normal(0, 0.05, len(df))
    df['timestamp'] = pd.Timestamp.now().isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f'Saved {len(df)} rows to {output_path}')
    return df

if __name__ == '__main__':
    df = load_raw_data()
    print(df.head())
    print(f'Shape: {df.shape}')
