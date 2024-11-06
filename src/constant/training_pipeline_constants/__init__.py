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

# TARGET_COLUMN = "cltv"
# PIPELINE_NAME: str = "src"



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
