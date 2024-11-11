from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str



@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str



@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str



@dataclass
class ClassificationMetricArtifact:
    accuracy:float
    roc_auc:float
    precision_0:float
    recall_0:float
    f1_score_0:float
    precision_1:float
    recall_1:float
    f1_score_1:float
    confusion_matrix_roc_curve_file_path:str




@dataclass
class RegressionMetricArtifact:
    spearman_rank_corr_coef:float
    normalized_gini_coef_baseline:float
    normalized_gini_coef_model:float
    mean_decile_mape:float
    lorenz_curve_decile_chart_file_path:str




@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    train_cls_metric_artifact:ClassificationMetricArtifact
    test_cls_metric_artifact:ClassificationMetricArtifact
    train_reg_metric_artifact:RegressionMetricArtifact
    test_reg_metric_artifact:RegressionMetricArtifact
