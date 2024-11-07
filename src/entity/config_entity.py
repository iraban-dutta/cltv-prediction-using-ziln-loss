import os
import sys
from datetime import datetime
from src.constant import training_pipeline_constants



class TrainingPipelineConfig:
    def __init__(self, company_id:int, timestamp:datetime=datetime.now()):
        self.run_name = f"{company_id}_{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}"
        self.artifact_dir_name = os.path.join(training_pipeline_constants.ARTIFACT_DIR, self.run_name)


class DataIngestionConfig:
    def __init__(self, company_id:int, training_pipeline_config:TrainingPipelineConfig):

        self.psql_user=training_pipeline_constants.DATA_INGESTION_PSQL_USER
        self.psql_pswd=training_pipeline_constants.DATA_INGESTION_PSQL_PSWD
        self.psql_host=training_pipeline_constants.DATA_INGESTION_PSQL_HOST
        self.psql_port=training_pipeline_constants.DATA_INGESTION_PSQL_PORT
        self.psql_db=training_pipeline_constants.DATA_INGESTION_PSQL_DB_NAME
        self.psql_engine_string = (
            f'postgresql+psycopg2://{self.psql_user}:{self.psql_pswd}'
            f'@{self.psql_host}:{self.psql_port}/{self.psql_db}'
        )

        self.train_test_split_ratio: float = training_pipeline_constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir_name, 
            training_pipeline_constants.DATA_INGESTION_DIR
        )
        self.feature_store_file_path:str=os.path.join(
            self.data_ingestion_dir, 
            training_pipeline_constants.DATA_INGESTION_FEATURE_STORE_DIR, 
            training_pipeline_constants.FILE_NAME.format(company_id)
        )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                training_pipeline_constants.DATA_INGESTION_INGESTED_DIR, 
                training_pipeline_constants.TRAIN_FILE_NAME.format(company_id)
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                training_pipeline_constants.DATA_INGESTION_INGESTED_DIR, 
                training_pipeline_constants.TEST_FILE_NAME.format(company_id)
            )
        



class DataValidationConfig:
    def __init__(self, company_id:int, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir_name,
            training_pipeline_constants.DATA_VALIDATION_DIR
        )
        self.valid_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_training_file_path: str = os.path.join(
            self.valid_dir,
            training_pipeline_constants.TRAIN_FILE_NAME.format(company_id)
        )
        self.valid_testing_file_path: str = os.path.join(
            self.valid_dir,
            training_pipeline_constants.TEST_FILE_NAME.format(company_id)
        )
        self.invalid_training_file_path: str = os.path.join(
            self.invalid_dir,
            training_pipeline_constants.TRAIN_FILE_NAME.format(company_id)
        )
        self.invalid_testing_file_path: str = os.path.join(
            self.invalid_dir,
            training_pipeline_constants.TEST_FILE_NAME.format(company_id)
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME.format(company_id)
        )





if __name__=='__main__':
    pass
    # obj1 = TrainingPipelineConfig(company_id=123)
    # obj2 = DataValidationConfig(company_id=123, training_pipeline_config=obj1)
    # attributes = [attr for attr in dir(obj2) if not attr.startswith('__')]
    # for key, val in {attr:getattr(obj2, attr) for attr in attributes}.items():
    #     print(f'{key}: {val}')
