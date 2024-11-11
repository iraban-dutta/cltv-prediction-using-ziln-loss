import os
import sys
import numpy as np
import pandas as pd
from src.utils.main_utils.utils import load_object, save_object, load_numpy_array_data, write_yaml_file
from src.entity.config_entity import TrainingPipelineConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
import tensorflow as tf
import keras
from keras import layers, models, Model
from src.utils.ml_utils.model.utils import Embedding_Layer
from src.utils.ml_utils.model.ziln import ZeroInflatedLogNormalLoss, zero_inflated_lognormal_pred
from src.utils.ml_utils.metric.classification_metric import get_classification_metrics
from src.utils.ml_utils.metric.regression_metric import get_regression_metrics
from src.utils.ml_utils.model.estimator import CLTVModel
from src.logging.logger import logging 
from src.exception.exception import CustomException



NUM_COLS = ['start_month', 'start_day_isweekend', 'first_purchase_amount_log', 'first_purchase_txns_cnt', 'first_purchase_productsize']
CAT_COLS = ['first_purchase_chain', 'first_purchase_dept', 'first_purchase_category', 'first_purchase_brand']
CALIBRATE_COL = ['first_purchase_amount']
TARGET_COL = ['cltv']
LEARNING_RATE=0.0002
EPOCHS=5
BATCH_SIZE=1024



class ModelTrainer:
    def __init__(self, 
                 company_id:int, 
                 model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.company_id = company_id
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            print(CustomException(e, sys))


    def get_ann_model(self, ziln_loss:bool, X_dict:dict):
        try:
            
            # Define input layers
            embeddings_input = [layers.Input(shape=(1, ), name=col, dtype=np.int32) for col in CAT_COLS]
            numeric_input = layers.Input(shape=(len(NUM_COLS), ), name='numeric_feats', dtype=np.float32)

            # Pass through Embeddings model and get embeddings output
            # embeddings_output = [Embedding_Layer(vocab_len=np.unique(X_dict[col]).shape[0], scale_down_factor=0.25)(emb_inp) for col, emb_inp in dict(zip(CAT_COLS, embeddings_input)).items()]
            embeddings_output = [Embedding_Layer(vocab_len=np.max(X_dict[col])+1, scale_down_factor=0.25)(emb_inp) for col, emb_inp in dict(zip(CAT_COLS, embeddings_input)).items()]

            # Concatenation: Form Deep input
            deep_input = layers.concatenate(embeddings_output + [numeric_input])

            # Final Final Fully Connected network
            deep_model = models.Sequential([
                layers.Dense(units=64, activation='relu',),
                layers.Dense(units=32, activation='relu',),
            ])
            if ziln_loss:
                deep_model.add(layers.Dense(units=3, name='output'))
            else:
                deep_model.add(layers.Dense(units=1, name='output'))

            # Assemble model
            complete_model = Model(inputs=embeddings_input + [numeric_input], outputs=deep_model(deep_input))
            return complete_model

        except Exception as e:
            print(CustomException(e, sys))   


    def start_ann_model_fitting(self, model, X_train, X_test, y_train, y_test):
        try:
            callbacks = [
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=1e-6),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            ]

            fit_history = model.fit(
                x={
                    'first_purchase_chain': X_train['first_purchase_chain'],
                    'first_purchase_dept': X_train['first_purchase_dept'],
                    'first_purchase_category': X_train['first_purchase_category'],
                    'first_purchase_brand': X_train['first_purchase_brand'],
                    'numeric_feats': X_train['numeric_feats']
                },
                y=y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=2,
                callbacks=callbacks,
                validation_data=(X_test, y_test)
            ).history

            return fit_history, model

        except Exception as e:
            print(CustomException(e, sys))   



    def start_ann_model_evaluation(self, 
                                   model, 
                                   X:dict, 
                                   y:np.ndarray, 
                                   y0:np.ndarray, 
                                   loss:str, 
                                   save_fig:bool, 
                                   report_file_path:str,
                                   figure_cls_file_path:str,
                                   figure_reg_file_path:str)->list:
        try:

            eval_artifacts = []
            if loss=='ziln':
                logits_pred = model.predict(x=X, batch_size=BATCH_SIZE)
                cltv_pred_proba_tf, cltv_pred_tf = zero_inflated_lognormal_pred(logits_pred, clip_value_logn_sigma=1)
                y_pred_proba = np.array(cltv_pred_proba_tf).ravel()
                y_pred_binary = (y_pred_proba>0.5).astype('float32')
                y_pred = np.array(cltv_pred_tf).ravel()

                y_true_binary = (y>0).astype('float32')
                cls_metrics_artifact = get_classification_metrics(y_true=y_true_binary, 
                                                                  y_pred=y_pred_binary, 
                                                                  y_pred_proba=y_pred_proba, 
                                                                  save_fig=save_fig, 
                                                                  file_path=figure_cls_file_path)
                reg_metrics_artifact = get_regression_metrics(y_true=y, 
                                                              y_pred=y_pred, 
                                                              y0_true=y0, 
                                                              save_fig=save_fig, 
                                                              file_path=figure_reg_file_path)
                
                cls_metrics_dict = {k:float(np.round(v, 5)) for k,v in cls_metrics_artifact.__dict__.items() if 'file_path' not in k}
                reg_metrics_dict = {k:float(np.round(v, 5)) for k,v in reg_metrics_artifact.__dict__.items() if 'file_path' not in k}
                write_yaml_file(file_path=report_file_path, 
                                content={
                                    'classification_metrics':cls_metrics_dict, 
                                    'regression_metrics':reg_metrics_dict
                                })
                eval_artifacts.append(cls_metrics_artifact)
                eval_artifacts.append(reg_metrics_artifact)

            elif loss=='mse':
                y_pred = model.predict(x=X, batch_size=BATCH_SIZE).ravel()
                reg_metrics_artifact = get_regression_metrics(y_true=y, 
                                                              y_pred=y_pred, 
                                                              y0_true=y0, 
                                                              save_fig=save_fig, 
                                                              file_path=figure_reg_file_path)
                
                reg_metrics_dict = {k:float(np.round(v, 5)) for k,v in reg_metrics_artifact.__dict__.items() if 'file_path' not in k}
                write_yaml_file(file_path=report_file_path, 
                                content={
                                    'regression_metrics':reg_metrics_dict
                                })
                eval_artifacts.append(None)
                eval_artifacts.append(reg_metrics_artifact)

            elif loss=='bce':
                logits = model.predict(x=X, batch_size=BATCH_SIZE)
                y_pred_proba = tf.keras.backend.sigmoid(logits).numpy().ravel()
                y_pred_binary = (y_pred_proba>0.5).astype('float32')

                y_true_binary = (y>0).astype('float32')
                cls_metrics_artifact = get_classification_metrics(y_true=y_true_binary, 
                                                                  y_pred=y_pred_binary, 
                                                                  y_pred_proba=y_pred_proba, 
                                                                  save_fig=save_fig, 
                                                                  file_path=figure_cls_file_path)
                
                cls_metrics_dict = {k:float(np.round(v, 5)) for k,v in cls_metrics_artifact.__dict__.items() if 'file_path' not in k}
                write_yaml_file(file_path=report_file_path, 
                                content={
                                    'classification_metrics':cls_metrics_dict
                                })
                eval_artifacts.append(cls_metrics_artifact)
                eval_artifacts.append(None)
                
            else:
                raise Exception('Invalid Loss function!')
            
            return eval_artifacts


        except Exception as e:
            print(CustomException(e, sys))  



    def initiate_model_training(self, loss:str)->ModelTrainerArtifact:
        try:
            
            logging.info(f"Model Training & Evaluation initiated with {loss} loss.")

            # Load transformed data
            dict_train = load_object(self.data_transformation_artifact.transformed_train_file_path)
            dict_test = load_object(self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Transformed data loaded succesfully.")

            # Split into dict:X  & arrays:(y0, y)
            keys_to_exclude = ['initial_purchase_amount', 'cltv']
            X_train, X_test = (
                {k:v for k, v in dict_train.items() if k not in keys_to_exclude}, 
                {k:v for k, v in dict_test.items() if k not in keys_to_exclude}
            )
            y0_train, y0_test = (
                dict_train['initial_purchase_amount'], 
                dict_test['initial_purchase_amount']
            )
            y_train, y_test = (
                dict_train['cltv'], 
                dict_test['cltv']
            )

            # # DEBUG
            # for k, v in X_train.items():
            #     if k!='numeric_feats':
            #         print(k, v.shape, np.unique(v).shape, np.min(v), np.max(v))
            #         print(v[:5])
            #         print('-'*50)
            #     else:
            #         print(k, v.shape)
            #         print(v[:5])
            #         print('-'*50)
            # for k, v in X_test.items():
            #     if k!='numeric_feats':
            #         print(k, v.shape, np.unique(v).shape, np.min(v), np.max(v))
            #     else:
            #         print(k, v.shape)
            # print(y0_train.shape, y0_test.shape)
            # print(y_train.shape, y_test.shape)
            # print('-'*100)


            # Build and compile ANN model
            if loss=='ziln':
                ann_model = self.get_ann_model(ziln_loss=True, X_dict=X_train)
                ann_model.compile(
                    loss=ZeroInflatedLogNormalLoss(
                        weigh_high_cltv_samples=True,
                        cltv_upp_threshold=10000.0,
                        high_cltv_samples_weight=10.0,
                        clip_value_logn_sigma=1.0
                    ),      
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),     
                )
            elif loss=='mse':
                ann_model = self.get_ann_model(ziln_loss=False, X_dict=X_train)
                ann_model.compile(
                    loss=keras.losses.MeanSquaredError(),      
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),     
                )
            elif loss=='bce':
                ann_model = self.get_ann_model(ziln_loss=False, X_dict=X_train)
                ann_model.compile(
                    loss=keras.losses.BinaryCrossentropy(from_logits=True),      
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),     
                )
                y_train = (y_train>0).astype('float32')
                y_test = (y_test>0).astype('float32')
            else:
                raise Exception('Invalid Loss function!')
            logging.info("ANN-Model built & compiled succesfully.")

            # # DEBUG
            # print(ann_model.summary())
            # for item in ann_model.inputs:
            #     print(item)
            # for item in ann_model.outputs:
            #     print(item)
            # print('-'*100)



            # Fit ANN model
            fit_history, fitted_ann_model = self.start_ann_model_fitting(model=ann_model, 
                                                                         X_train=X_train, X_test=X_test, 
                                                                         y_train=y_train, y_test=y_test)
            logging.info("ANN-Model fitted succesfully.")


            # Make directory to save model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            cltv_model = CLTVModel(preprocessor=preprocessor, model=fitted_ann_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=cltv_model)
            logging.info(f"ANN-Model saved succesfully.")
            

            # Make directory to store evaluation metrics
            os.makedirs(os.path.dirname(self.model_trainer_config.figure_cls_metrics_file_path), exist_ok=True)

            # Model Evaluation: Train Data
            eval_train_artifacts = self.start_ann_model_evaluation(model=fitted_ann_model, 
                                                                   X=X_train, 
                                                                   y=y_train, 
                                                                   y0=y0_train, 
                                                                   loss=loss, 
                                                                   save_fig=True,
                                                                   report_file_path=self.model_trainer_config.report_metrics_file_path.format('train'),
                                                                   figure_cls_file_path=self.model_trainer_config.figure_cls_metrics_file_path.format('train'),
                                                                   figure_reg_file_path=self.model_trainer_config.figure_reg_metrics_file_path.format('train'))

            # Model Evaluation: Test Data
            eval_test_artifacts = self.start_ann_model_evaluation(model=fitted_ann_model, 
                                                                   X=X_test, 
                                                                   y=y_test, 
                                                                   y0=y0_test, 
                                                                   loss=loss, 
                                                                   save_fig=True,
                                                                   report_file_path=self.model_trainer_config.report_metrics_file_path.format('test'),
                                                                   figure_cls_file_path=self.model_trainer_config.figure_cls_metrics_file_path.format('test'),
                                                                   figure_reg_file_path=self.model_trainer_config.figure_reg_metrics_file_path.format('test'))
            logging.info(f"ANN-Model evaluated succesfully.")



            model_trainer_artifact = ModelTrainerArtifact(
                            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                            train_cls_metric_artifact=eval_train_artifacts[0],
                            test_cls_metric_artifact=eval_test_artifacts[0],
                            train_reg_metric_artifact=eval_train_artifacts[1],
                            test_reg_metric_artifact=eval_test_artifacts[1],
            )
            logging.info("Model Training & Evaluation completed.")

            return model_trainer_artifact

        except Exception as e:
            print(CustomException(e, sys))