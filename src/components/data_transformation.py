import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from src.utils.main_utils.utils import save_numpy_array_data, save_object

from src.entity.config_entity import TrainingPipelineConfig, DataTransformationConfig
from src.entity.artifact_entity import  DataValidationArtifact, DataTransformationArtifact
from src.logging.logger import logging 
from src.exception.exception import CustomException




NUM_COLS = ['start_month', 'start_day_isweekend', 'first_purchase_amount_log', 'first_purchase_txns_cnt', 'first_purchase_productsize']
CAT_COLS = ['first_purchase_chain', 'first_purchase_dept', 'first_purchase_category', 'first_purchase_brand']
CALIBRATE_COL = ['first_purchase_amount']
TARGET_COL = ['cltv']



class DataTransformation:
    def __init__(self, 
                 company_id:int, 
                 data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.company_id = company_id
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact

        except Exception as e:
            print(CustomException(e, sys))


    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(filepath_or_buffer=file_path)
        except Exception as e:
            print(CustomException(e, sys))


    def get_categorical_feature_pipeline(self, perform_scaling=False)->Pipeline:
        """
        Creates a transformation pipeline for categorical features.

        The pipeline includes missing value imputation using the most frequent value (mode), 
        followed by ordinal encoding for converting categorical values to numerical. 
        Optionally, it applies standard scaling based on the `perform_scaling` parameter.

        Args:
            perform_scaling (bool): If True, applies standard scaling to the encoded features. 
                                    Default is False.

        Returns:
            Pipeline: A scikit-learn Pipeline object for transforming categorical features.
        """
        try:
            mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            std_scaler = StandardScaler()


            if perform_scaling:
                pipeline = Pipeline([
                    ('mode_imputer', mode_imputer),
                    ('ord_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-123)),
                    ('std_scaler',std_scaler)
                ])
            else:
                pipeline = Pipeline([
                    ('mode_imputer', mode_imputer),
                    ('ord_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-123)),
                    # ('std_scaler',std_scaler)
                ])

            logging.info("Pipeline object created for categorical features")
            return pipeline

        except Exception as e:
            print(CustomException(e, sys))



    def get_numeric_feature_pipeline(self, perform_scaling=False)->Pipeline:
        """
        Creates a transformation pipeline for numerical features.

        The pipeline performs missing value imputation using KNN Imputer. 
        Optionally, it applies standard scaling if `perform_scaling` is set to True.

        Args:
            perform_scaling (bool, optional): If True, applies StandardScaler for feature scaling. 
                                            Default is False.

        Returns:
            Pipeline: A scikit-learn Pipeline object for transforming numerical features.
        """
        try:
            knn_imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)
            std_scaler = StandardScaler()
            if perform_scaling:
                pipeline = Pipeline(steps=[
                    ('knn_imputer', knn_imputer),
                    ('std_scaler',std_scaler)
                ])
            else:
                pipeline = Pipeline(steps=[
                    ('knn_imputer', knn_imputer),
                    # ('std_scaler',std_scaler)
                ])

            logging.info("Pipeline object created for numerical features")

            return pipeline
        except Exception as e:
            print(CustomException(e, sys))




    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Executes the data transformation process for training and testing datasets.

        This method performs missing value imputation and encoding on categorical and 
        numerical features, saves the transformation objects, and outputs transformed data.

        Returns:
            DataTransformationArtifact: An object containing file paths of the saved transformer 
            and transformed train/test datasets.
        """
        try:
            logging.info("Data Transformation initiated.")

            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            df_train = DataTransformation.read_data(file_path=train_file_path)
            df_test = DataTransformation.read_data(file_path=test_file_path)

            # This pipeline performs Missing Value Imputation followed by label encoding of categorical columns
            transformer = ColumnTransformer([
                ('cat_pipe', self.get_categorical_feature_pipeline(perform_scaling=False), CAT_COLS),
                ('num_pipe', self.get_numeric_feature_pipeline(perform_scaling=False), NUM_COLS+CALIBRATE_COL)
            ])
            logging.info("Transformer Object Created")

            # Seperating into input and target features
            X_train, X_test = df_train[CAT_COLS+NUM_COLS+CALIBRATE_COL], df_test[CAT_COLS+NUM_COLS+CALIBRATE_COL]
            y_train, y_test = df_train[TARGET_COL], df_test[TARGET_COL]

            
            # Performing Transformation
            transformer.fit(X_train)
            transformed_col_names = [col.split('__')[-1] for col in transformer.get_feature_names_out()]
            X_train_trans = pd.DataFrame(transformer.transform(X_train), columns=transformed_col_names)
            X_test_trans = pd.DataFrame(transformer.transform(X_test), columns=transformed_col_names)
            logging.info("Transformation Succesfully Executed")

            
            # Convert dataframes to numpy arrays
            arr_train = np.c_[np.array(X_train_trans), np.array(y_train)]
            arr_test = np.c_[np.array(X_test_trans), np.array(y_test)]

            # Save preprocessor object
            save_object(self.data_transformation_config.transformed_object_file_path, transformer)
            # save_object("final_model/preprocessor.pkl", transformer)

            # Save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_training_file_path, array=arr_train)
            save_numpy_array_data(self.data_transformation_config.transformed_testing_file_path, array=arr_test)


            data_transformation_artifact = DataTransformationArtifact(
                                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                                transformed_train_file_path=self.data_transformation_config.transformed_training_file_path, 
                                transformed_test_file_path=self.data_transformation_config.transformed_testing_file_path)
            logging.info("Data Transformation completed.")


            return data_transformation_artifact


            # # DEBUG:
            # print(X_train_trans.head(2))
            # print(X_train_trans.info())
            # print(X_test_trans.info())
            # for col in CAT_COLS:
            #     print(col)
            #     print('Before Transformation:', X_train[col].nunique())
            #     print('After Transformation:', X_train_trans[col].nunique())
            #     print('-'*50)

            # for col in CAT_COLS:
            #     print(col)
            #     print('Before Transformation:', X_test[col].nunique())
            #     print('After Transformation:', X_test_trans[col].nunique())
            #     print('Unknown Categories:', (X_test_trans[col]==-123).any())
            #     print('-'*50)


        except Exception as e:
            print(CustomException(e, sys))
