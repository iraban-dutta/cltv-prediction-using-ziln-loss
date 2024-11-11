import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.entity.artifact_entity import ClassificationMetricArtifact
from src.logging.logger import logging 
from src.exception.exception import CustomException




def get_classification_metrics(y_true:np.ndarray, 
                               y_pred:np.ndarray, 
                               y_pred_proba:np.ndarray, 
                               save_fig:bool,
                               file_path:str)->ClassificationMetricArtifact:
    try:

        # Classification Scores
        accuracy = accuracy_score(y_true, y_pred)   
        roc_auc = roc_auc_score(y_true, y_pred_proba) 
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        # Confusion Matrix and ROC Curve
        cls_fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axs[0])
        axs[0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
        RocCurveDisplay.from_predictions(y_true, y_pred_proba, pos_label=1, ax=axs[1])
        axs[1].set_title('ROC Curve', fontweight='bold', fontsize=12)
        if save_fig:
            cls_fig.savefig(file_path, dpi='figure')
        # plt.show()
        plt.close(cls_fig)

        classification_metric =  ClassificationMetricArtifact(
                    accuracy=accuracy,
                    roc_auc=roc_auc,
                    precision_0=precision[0],
                    recall_0=recall[0],
                    f1_score_0=f1[0],
                    precision_1=precision[1],
                    recall_1=recall[1],
                    f1_score_1=f1[1],
                    confusion_matrix_roc_curve_file_path=file_path
        )

        return classification_metric
    
    except Exception as e:
        print(CustomException(e,sys))


# if __name__=='__main__':
#     y_true = np.random.randint(low=0, high=2, size=1000)
#     y_pred = np.random.randint(low=0, high=2, size=1000)
#     y_pred_proba = np.random.uniform(low=0, high=1, size=1000)
#     cls_metric_artifact = get_classification_metrics(y_true, 
#                                                      y_pred, 
#                                                      y_pred_proba, 
#                                                      False, 
#                                                      None)
#     print(cls_metric_artifact)
