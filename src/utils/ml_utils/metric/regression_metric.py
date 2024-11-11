import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.entity.artifact_entity import RegressionMetricArtifact
from src.logging.logger import logging 
from src.exception.exception import CustomException




def get_series_cumulative_revenue(y_true:np.array, y_pred:np.array)->pd.Series:
    df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred}).sort_values(by='y_pred', ascending=False)
    ser_cumrev = df['y_true'].cumsum()/df['y_true'].sum()
    return ser_cumrev


def get_df_gain(y_true:np.array, y_pred:np.array, y0_true:np.array)->pd.DataFrame:

    # Gain Chart (Lorenz Curve)
    ground_truth = get_series_cumulative_revenue(y_true=y_true, y_pred=y_true)
    baseline = get_series_cumulative_revenue(y_true=y_true, y_pred=y0_true)
    model = get_series_cumulative_revenue(y_true=y_true, y_pred=y_pred)

    df_gain = pd.DataFrame({
        'GroundTruth': ground_truth.values,
        'Baseline': baseline.values,
        'Model': model.values
    })

    return df_gain



def decile_stats_aggregate_fn(df:pd.DataFrame):
    return pd.Series({
        'label_mean': np.mean(df['y_true']),
        'pred_mean': np.mean(df['y_pred']),
        'normalized_rmse': np.sqrt(mean_squared_error(df['y_true'], df['y_pred'])) / df['y_true'].mean(), 
        'normalized_mae': mean_absolute_error(df['y_true'], df['y_pred']) / df['y_true'].mean(),
    })


def get_decile_stats(y_true:np.array, y_pred:np.array)->pd.DataFrame:

    num_buckets = 10
    decile = pd.qcut(y_pred, q=num_buckets, labels=[f'{i}' for i in range(num_buckets)])

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'decile': decile,
    }).groupby('decile', observed=False).apply(decile_stats_aggregate_fn, include_groups=False)

    df['decile_mape'] = np.abs(df['pred_mean'] - df['label_mean']) / df['label_mean']

    return df




def get_regression_metrics(y_true:np.ndarray, 
                           y_pred:np.ndarray, 
                           y0_true:np.ndarray, 
                           save_fig:bool, 
                           file_path:str)->RegressionMetricArtifact:
    try:

        # Compute Normalized Gini Coeff
        df_gain = get_df_gain(y_true, y_pred, y0_true)
        gini_coeff = df_gain.apply(lambda x: 2*x.sum()/x.shape[0])
        norm_gini_coeff = gini_coeff/gini_coeff.iloc[0]


        # Decile Stats
        df_decile_stats = get_decile_stats(y_true, y_pred)



        # Gain Chart (Lorenz Curve) and Decile Chart with MAPE values  
        reg_fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        df_gain.index=(df_gain.index/df_gain.shape[0])*100
        df_gain.plot(ax=axs[0])
        axs[0].set_xlabel('Cumulative Customer Count Percentage')
        axs[0].set_ylabel('Cumulative CLTV (Normalized)')
        axs[0].set_title('Gain Chart (Lorenz Curve)', fontweight='bold', fontsize=12)
        df_decile_stats[['label_mean', 'pred_mean']].plot.bar(rot=0, ax=axs[1])
        axs[1].set_xlabel('Prediction decile bucket')
        axs[1].set_ylabel('Average bucket value (CLTV)')
        axs[1].legend(['GroundTruth', 'Model'], loc='upper left')
        axs[1].set_title('Decile Chart', fontweight='bold', fontsize=12)
        if save_fig:
            reg_fig.savefig(file_path, dpi='figure')
        # plt.show()
        plt.close(reg_fig)

        regression_metric =  RegressionMetricArtifact(
                    spearman_rank_corr_coef=spearmanr(y_true, y_pred, nan_policy='omit').statistic,
                    normalized_gini_coef_baseline=norm_gini_coeff.iloc[1],
                    normalized_gini_coef_model=norm_gini_coeff.iloc[2],
                    mean_decile_mape=df_decile_stats['decile_mape'].mean(),
                    lorenz_curve_decile_chart_file_path=file_path
        )

        return regression_metric
    
    except Exception as e:
        print(CustomException(e,sys))


# if __name__=='__main__':
#     y_true = np.random.randint(low=0, high=5000, size=1000)
#     y_pred = np.random.randint(low=0, high=5000, size=1000)
#     y0_true = np.random.uniform(low=0, high=5000, size=1000)
#     reg_metric_artifact = get_regression_metrics(y_true, 
#                                                  y_pred, 
#                                                  y0_true, 
#                                                  False, 
#                                                  None)
#     print(reg_metric_artifact)






