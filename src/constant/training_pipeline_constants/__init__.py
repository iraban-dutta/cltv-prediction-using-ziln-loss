import os
import sys
import numpy as np

from dotenv import load_dotenv
load_dotenv()


"""
defining common constant variables for training pipeline
"""
ARTIFACT_DIR:str = "artifacts"
FILE_NAME:str = "cltv_data_{}.csv"
TRAIN_FILE_NAME:str = "train_{}.csv"
TEST_FILE_NAME:str = "test_{}.csv"
SCHEMA_FILE_PATH = os.path.join('data_schema', 'schema.yaml')



"""
Data Ingestion related constants start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_PSQL_USER: str = os.getenv("psql_user")
DATA_INGESTION_PSQL_PSWD: str = os.getenv("psql_password")
DATA_INGESTION_PSQL_HOST: str = os.getenv("psql_host")
DATA_INGESTION_PSQL_PORT: str = os.getenv("psql_port")
DATA_INGESTION_PSQL_DB_NAME: str =  os.getenv("psql_database")

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report_{}.yaml"



"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_PREPROCESS_OBJ_FILE_NAME = "preprocessing_{}.pkl"
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train_{}.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test_{}.npy"