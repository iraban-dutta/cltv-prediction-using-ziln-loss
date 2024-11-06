import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logging.logger import logging 
from src.exception.exception import CustomException


NUM_COLS = ['start_month', 'start_day_isweekend', 'first_purchase_amount_log', 'first_purchase_txns_cnt', 'first_purchase_productsize']
CAT_COLS = ['first_purchase_chain', 'first_purchase_dept', 'first_purchase_category', 'first_purchase_brand']
CALIBRATE_COL = ['first_purchase_amount']
TARGET_COL = ['cltv']



class DataIngestion:
    def __init__(self, company_id:int, data_ingestion_config:DataIngestionConfig):
        try:
            self.company_id = company_id
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            print(CustomException(e, sys))

    def import_txn_table_as_df(self)->pd.DataFrame:
        '''
        Reads postgres table from DB and loads it into a DF
        '''
        try:
            logging.info("Started reading table from postgres DB.")
            engine = create_engine(self.data_ingestion_config.psql_engine_string)
            # sql_query = f'''select * from txns_{self.company_id} limit 1000;'''
            sql_query = f'''select * from txns_{self.company_id};'''
            df = pd.read_sql_query(sql=text(sql_query), con=engine)

            df.drop_duplicates(keep='first', inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.replace({'na':np.nan}, inplace=True)
            logging.info("Completed reading table from postgres DB.")


            return df
        except Exception as e:
            print(CustomException(e, sys))


    def process_df_txn_to_customer_lvl(self, df:pd.DataFrame)->pd.DataFrame:
        '''
        Aggregates the dataset from the txn level to the customer level
        Args:
            df: DF @ Transaction level,
        Returns:
            pd.DataFrame: DF @ Customer level with CLTV values,
        '''
        try:
            logging.info("Started aggregating table from txn level to customer level.")
            # Get the start date for each customer
            df['start_date']=df.groupby('id')['date'].transform('min')

            # First Purchase Values: Amount, Txn Counts 
            df_cust_first_purchase_val = df.query('date==start_date').groupby('id')[['purchaseamount']].agg(['sum', 'count']).reset_index()
            df_cust_first_purchase_val.columns = ['id', 'first_purchase_amount', 'first_purchase_txns_cnt']


            # First purchase Attributes: Attributes of the most costly txn in case of multiple txns
            df_cust_first_purchase_attr = (
                df.query('date == start_date')
                .sort_values('purchaseamount', ascending=False)
                .groupby('id')[['start_date', 'chain', 'dept', 'category', 'brand', 'productsize', 'productmeasure']]
                .first()
                .reset_index()
            )
            df_cust_first_purchase_attr.columns = ['id', 'start_date'] + ['first_purchase_'+col for col in df_cust_first_purchase_attr.columns[2:]]

            # Generate Cltv & Txn Counts (1 Year Window)
            mask_future1Y_from_activate = (df['date']>df['start_date']) & (df['date']<=df['start_date'] + np.timedelta64(365, 'D'))
            df_cust_ltv_1y = df[mask_future1Y_from_activate].groupby('id')['purchaseamount'].agg(['sum', 'count']).reset_index()
            df_cust_ltv_1y.columns = ['id', 'cltv', 'txns_cnt']

            # Merging
            df_cust_lvl = pd.merge(pd.merge(df_cust_first_purchase_attr, df_cust_first_purchase_val, on='id', how='left'), df_cust_ltv_1y, on='id', how='left')
            

            # Extracting some basic features
            df_cust_lvl['start_month'] = df_cust_lvl['start_date'].dt.month
            df_cust_lvl['start_day_isweekend'] = df_cust_lvl['start_date'].dt.day_of_week.apply(lambda x: True if x>4 else False).astype('int')
            df_cust_lvl['first_purchase_amount_log'] = np.log(df_cust_lvl['first_purchase_amount'])


            # Dropping some unnecessary columns
            df_cust_lvl.drop(['start_date', 'first_purchase_productmeasure', 'txns_cnt'], axis=1, inplace=True)


            # Fill missing values
            df_cust_lvl.fillna(0, inplace=True)

            # Fix column data types
            df_cust_lvl['first_purchase_chain'] = df_cust_lvl['first_purchase_chain'].astype('category')
            df_cust_lvl['first_purchase_dept'] = df_cust_lvl['first_purchase_dept'].astype('category')
            df_cust_lvl['first_purchase_category'] = df_cust_lvl['first_purchase_category'].astype('category')
            df_cust_lvl['first_purchase_brand'] = df_cust_lvl['first_purchase_brand'].astype('category')

            logging.info("Completed aggregating table from txn level to customer level.")

            return df_cust_lvl


        except Exception as e:
            print(CustomException(e, sys))


    def save_data_into_feature_store(self, df:pd.DataFrame)->None:
        '''
        Save data into the feature store
        Args:
            df: DF that needs to be saved,
        Returns:
            None,
        '''
        try:
            logging.info("Started saving raw_data into the feature store.")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Make dir
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            df.to_csv(path_or_buf=feature_store_file_path, index=False, header=True)
            logging.info("Completed saving raw_data into the feature store.")
        except Exception as e:
            print(CustomException(e, sys))


    def train_test_splitter(self, df:pd.DataFrame)->None:
        '''
        Perfroms train-test split
        Args:
            df: DF to perfrom train-test split on,
        Returns:
            None,
        '''
        try:
            # Performing train-test split
            logging.info("Started train-test split.")
            df_filtered = df[CAT_COLS+NUM_COLS+CALIBRATE_COL+TARGET_COL]

            df_train, df_test  = train_test_split(
                                    df_filtered, 
                                    test_size=self.data_ingestion_config.train_test_split_ratio, 
                                    random_state=42)
            

            logging.info("Completed train-test split.")


            logging.info("Started exporting train and test data.")
            # Make dir
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            # Saving df_train and df_test
            df_train.to_csv(path_or_buf=self.data_ingestion_config.training_file_path, index=False, header=True)
            df_test.to_csv(path_or_buf=self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Completed exporting train and test data.")

        except Exception as e:
            print(CustomException(e, sys))


    def initiate_data_ingestion(self)->DataIngestionArtifact:
        '''
        Run the data ingestion pipeline
        Args:
            None,
        Returns:
            DataIngestionArtifact: Object of the class DataIngestionArtifact storing the train and test file path,
        '''
        try:
            logging.info("Data Ingestion initiated.")

            df_txns_lvl=self.import_txn_table_as_df()
            df_cust_lvl=self.process_df_txn_to_customer_lvl(df=df_txns_lvl)
            self.save_data_into_feature_store(df=df_cust_lvl)
            self.train_test_splitter(df=df_cust_lvl)
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info("Data Ingestion completed.")
            
            return data_ingestion_artifact

            
        except Exception as e:
            print(CustomException(e, sys))


if __name__=='__main__':
    try:
        obj1 = TrainingPipelineConfig(company_id=103800030)
        obj2 = DataIngestionConfig(company_id=103800030, training_pipeline_config=obj1)
        obj3 = DataIngestion(company_id=103800030, data_ingestion_config=obj2)

        df_txn_test = obj3.import_txn_table_as_df()
        print(df_txn_test.shape)

        df_cust_test = obj3.process_df_txn_to_customer_lvl(df=df_txn_test)
        print(df_cust_test.shape)
        print(df_cust_test.info())
        print(df_cust_test.isna().sum())
        print(df_cust_test.duplicated().sum())


        obj3.save_data_into_feature_store(df=df_cust_test)
        obj3.train_test_splitter(df=df_cust_test)
    except Exception as e:
        print(CustomException(e, sys))