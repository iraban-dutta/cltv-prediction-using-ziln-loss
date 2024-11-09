import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.constant.training_pipeline_constants import SCHEMA_FILE_PATH
from src.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.logging.logger import logging 
from src.exception.exception import CustomException



NUM_COLS = ['start_month', 'start_day_isweekend', 'first_purchase_amount_log', 'first_purchase_txns_cnt', 'first_purchase_productsize']
CAT_COLS = ['first_purchase_chain', 'first_purchase_dept', 'first_purchase_category', 'first_purchase_brand']
CALIBRATE_COL = ['first_purchase_amount']
TARGET_COL = ['cltv']



class DataValidation:
    def __init__(self, 
                 company_id:int, 
                 data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.company_id = company_id
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            print(CustomException(e, sys))


    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(filepath_or_buffer=file_path)
        except Exception as e:
            print(CustomException(e, sys))



    def validate_num_of_cols(self, df:pd.DataFrame)->bool:
        """
        Validates if the number of columns in the input DataFrame matches the expected schema.

        Compares the number of columns in the provided DataFrame with the expected number of columns 
        defined in the schema configuration.

        Args:
            df (pd.DataFrame): The DataFrame whose column count is to be validated.

        Returns:
            bool: True if the number of columns matches the expected schema, False otherwise.
        """
        try:
            number_of_columns=len(self._schema_config['columns'])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(df.columns)}")
            if len(df.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            print(CustomException(e, sys))


    def detect_data_drift(self, df_base:pd.DataFrame, df_curr:pd.DataFrame, threshold:float=0.05)->bool:
        """
        Detects data drift by comparing numerical columns in two datasets using the Kolmogorov-Smirnov test.

        This method compares the base and current datasets' numerical columns to identify any drift 
        in the data. It returns a boolean value indicating whether any drift is detected based on the 
        specified threshold.

        Args:
            df_base (pd.DataFrame): The original dataset to compare against.
            df_curr (pd.DataFrame): The current dataset to check for drift.
            threshold (float, optional): The p-value threshold for detecting drift. Default is 0.05.

        Returns:
            bool: True if data drift is detected in any numerical column, False otherwise.
        """
        try:
            drift_status=[]
            drift_found=False
            drift_report = {}

            for col in NUM_COLS+CALIBRATE_COL+TARGET_COL:
                col_base = df_base[col]
                col_curr = df_curr[col]
                ks_2samp_result = ks_2samp(col_base, col_curr)
                if ks_2samp_result.pvalue < threshold:
                    drift_found=True
                
                
                drift_report.update({col:{
                    'p_value':float(np.round(ks_2samp_result.pvalue, 4)),
                    'drift_status':drift_found
                }})

                drift_status.append(drift_found)

            # Compile final drift status
            drift_status = np.any(np.array(drift_status))

            # Make dir
            drift_report_file_dir = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(drift_report_file_dir, exist_ok=True)

            # Write Drift Report
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=drift_report, replace=False)

            return drift_status

        except Exception as e:
            print(CustomException(e, sys))



    def initiate_data_validation(self)->DataValidationArtifact:
        """
        Executes the data validation process for the training and testing datasets.

        This method validates the number of columns, checks for data drift between the training 
        and testing datasets, and saves the validated data. It returns an artifact containing 
        the validation status and file paths for both valid and invalid data.

        Returns:
            DataValidationArtifact: An object containing the validation status and file paths 
            for the valid and invalid training and testing datasets.
        """
        try:
            logging.info("Data Validation initiated.")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            df_train = DataValidation.read_data(file_path=train_file_path)
            df_test = DataValidation.read_data(file_path=test_file_path)



            # Validate num of columns
            logging.info("Validate number of columns: Train")
            train_num_col_status=self.validate_num_of_cols(df=df_train)
            if not train_num_col_status:
                error_message=f'Train dataframe does not contain all columns.\n'
                logging.info(error_message)

            
            logging.info("Validate number of columns: Test")
            test_num_col_status=self.validate_num_of_cols(df=df_test)
            if not test_num_col_status:
                error_message=f'Test dataframe does not contain all columns.\n'
                logging.info(error_message)



            # Validate numerical columns: Skipped
            # Validate categorical columns: Skipped


            # Validate data drift
            logging.info("Validate drift in numerical columns")
            drift_status=self.detect_data_drift(df_base=df_train, df_curr=df_test)
            logging.info(f"Drift Detected: {drift_status}")
            if drift_status:
                error_message=f'Drift detected in numerical columns.\n'
                logging.info(error_message)



            
            if train_num_col_status and test_num_col_status and ~drift_status:
                # If no validation discrepancies occur 
                valid_dir_name = os.path.dirname(self.data_validation_config.valid_training_file_path)
                os.makedirs(valid_dir_name, exist_ok=True)
                df_train.to_csv(path_or_buf=self.data_validation_config.valid_training_file_path, index=False, header=True)
                df_test.to_csv(path_or_buf=self.data_validation_config.valid_testing_file_path, index=False, header=True)

            else:
                # If validation discrepancies occur 
                invalid_dir_name = os.path.dirname(self.data_validation_config.invalid_training_file_path)
                os.makedirs(invalid_dir_name, exist_ok=True)
                df_train.to_csv(path_or_buf=self.data_validation_config.invalid_training_file_path, index=False, header=True)
                df_test.to_csv(path_or_buf=self.data_validation_config.invalid_testing_file_path, index=False, header=True)


            data_validation_artifact = DataValidationArtifact(
                validation_status=(train_num_col_status and test_num_col_status and ~drift_status),
                valid_train_file_path=self.data_validation_config.valid_training_file_path,
                valid_test_file_path=self.data_validation_config.valid_testing_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_training_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_testing_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info("Data Validation completed.")

            return data_validation_artifact

        except Exception as e:
            print(CustomException(e, sys))




if __name__=='__main__':
    
    # _schema_yaml_file = read_yaml_file(SCHEMA_FILE_PATH)
    # print(_schema_yaml_file)
    # print(len(_schema_yaml_file['columns']))

    pass