import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the schema — this is the contract for your data
raw_schema = DataFrameSchema({
    'sepal_length': Column(float, [
        Check.greater_than(0),
        Check.less_than(20),
        Check(lambda s: s.isna().sum() == 0, error='sepal_length has nulls'),
    ]),
    'sepal_width': Column(float, [
        Check.greater_than(0),
        Check.less_than(20),
        Check(lambda s: s.isna().sum() == 0, error='sepal_width has nulls'),
    ]),
    'petal_length': Column(float, [
        Check.greater_than(0),
        Check.less_than(20),
    ]),
    'petal_width': Column(float, [
        Check.greater_than(0),
        Check.less_than(10),
    ]),
    'target': Column(int, Check.isin([0, 1, 2])),
    'species': Column(str, Check.isin(['setosa', 'versicolor', 'virginica'])),
})

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f'Validating dataframe with shape {df.shape}')
    try:
        validated = raw_schema.validate(df, lazy=True)
        logger.info('Validation passed')
        return validated
    except pa.errors.SchemaErrors as e:
        logger.error(f'Validation FAILED:\n{e.failure_cases}')
        raise

def validate_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return validate_raw_data(df)

if __name__ == '__main__':
    from src.ingestion.data_loader import load_raw_data
    df = load_raw_data()
    validated = validate_raw_data(df)
    print('Validation passed!')
    print(validated.dtypes)
