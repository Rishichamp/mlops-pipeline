import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

raw_schema = DataFrameSchema({
    'sepal_length': Column(float, [
        Check(lambda x: x > 0, element_wise=True),
        Check(lambda x: x < 20, element_wise=True),
    ]),
    'sepal_width': Column(float, [
        Check(lambda x: x > 0, element_wise=True),
        Check(lambda x: x < 20, element_wise=True),
    ]),
    'petal_length': Column(float, [
        Check(lambda x: x > 0, element_wise=True),
        Check(lambda x: x < 20, element_wise=True),
    ]),
    'petal_width': Column(float, [
        Check(lambda x: x > 0, element_wise=True),
        Check(lambda x: x < 10, element_wise=True),
    ]),
    'target': Column(int, Check(lambda x: x in [0, 1, 2], element_wise=True)),
    'species': Column(str, Check(lambda x: x in ['setosa', 'versicolor', 'virginica'], element_wise=True)),
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
    import sys
    sys.path.insert(0, '.')
    from src.ingestion.data_loader import load_raw_data
    df = load_raw_data()
    validated = validate_raw_data(df)
    print('Validation passed!')
    print(validated.dtypes)
