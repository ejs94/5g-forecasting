import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from darts import TimeSeries

def process_results_parquet(folder: str) -> pd.DataFrame:
    """
    Reads all Parquet files in a folder of results and concatenates them into a single DataFrame.

    Args:
        folder (str): Path to the folder containing the results in Parquet files.

    Returns:
        pd.DataFrame: DataFrame containing all the data from the results.
    """
    # List to store DataFrames
    dfs = []

    # Iterate over all files in the folder
    for file in os.listdir(folder):
        if file.endswith('.parquet'):
            # Create the full path of the file
            file_path = os.path.join(folder, file)
            # Read the Parquet file and add it to the DataFrame
            df = pd.read_parquet(file_path)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def aggregate_median_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para calcular a mediana de MAE, RMSE e MSE, agrupando por Model, target e Activity.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados com as colunas 'Model', 'target', 'Activity', 'MAE', 'RMSE', 'MSE'.

    Retorna:
    pd.DataFrame: DataFrame agregado contendo as medianas das métricas MAE, RMSE e MSE por Model, target e Activity.
    """
    
    # Verificar se as colunas necessárias estão presentes no DataFrame
    required_columns = ['Model', 'target', 'Activity', 'MAE', 'RMSE', 'MSE']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna {col} está ausente no DataFrame")

    # Agrupar por Model, target e Activity e calcular as medianas das métricas
    aggregated_df = df.groupby(['Model', 'target', 'Activity'], as_index=False).agg({
        'MAE': 'median',
        'RMSE': 'median',
        'MSE': 'median'
    })
    
    # Renomear as colunas agregadas para refletir que são medianas
    aggregated_df.rename(columns={
        'MAE': 'MAE_Median',
        'RMSE': 'RMSE_Median',
        'MSE': 'MSE_Median'
    }, inplace=True)
    
    return aggregated_df


def plot_bar_for_medians_by_target(df: pd.DataFrame) -> None:
    """
    Generates bar plots for the metrics MAE_Median, RMSE_Median, and MSE_Median, separated by each unique target in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame that contains the data for the bar plots.

    Returns:
    None: Displays the bar plots for MAE_Median, RMSE_Median, and MSE_Median for each unique target.
    """
    # Obtaining unique targets
    targets = df['target'].unique()
    
    # List of metrics to be plotted
    metrics = ['MAE_Median', 'RMSE_Median', 'MSE_Median']
    
    # Define the desired order of activities
    activity_order = ['static_down', 'static_strm', 'driving_down', 'driving_strm']
    
    # Iterate over each unique target
    for target in targets:
        # Filtering data for the specific target
        target_data = df[df['target'] == target]
        
        # Iterate over each metric
        for metric in metrics:
            # Plotting the bar plot for the specific target and metric
            plt.figure(figsize=(10, 6))
            sns.barplot(data=target_data, x='Activity', y=metric, hue='Model', order=activity_order)
            
            # Adjusting title and axis labels
            plt.title(f'Bar Plot of {metric} for target: {target}')
            plt.ylabel(metric)
            plt.xlabel('Activity')
            
            # Displaying the graph
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def plot_boxplots_for_metrics_by_target(data: pd.DataFrame) -> None:
    """
    Function to generate boxplots of MAE, RMSE, and MSE grouped by 'Model', 'target', and 'Activity'.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data with columns 'Model', 'target', 'Activity', 'MAE', 'RMSE', 'MSE'.

    Returns:
    None: Displays the boxplot graphs.
    """
    
    # Remove NaN and infinite values
    # data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # data.dropna(inplace=True)

    # Get unique targets
    targets = data['target'].unique()

    # Define the desired order of activities
    activity_order = ['static_down', 'static_strm', 'driving_down', 'driving_strm']

    # Loop through the targets (e.g., CQI, RSRP, RSRQ, RSSI, SNR)
    for target in targets:
        # Filter the data for the current target
        data_subset = data[data['target'] == target]

        # Create subplots for MAE, RMSE, and MSE
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        fig.suptitle(f'Boxplots for {target}', fontsize=16)

        # Plot MAE metrics
        sns.boxplot(data=data_subset, x='Activity', y='MAE', hue='Model', ax=axes[0], order=activity_order)
        axes[0].set_title(f'Boxplot of MAE for {target}')
        axes[0].set_ylabel('MAE')
        axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot RMSE metrics
        sns.boxplot(data=data_subset, x='Activity', y='RMSE', hue='Model', ax=axes[1], order=activity_order)
        axes[1].set_title(f'Boxplot of RMSE for {target}')
        axes[1].set_ylabel('RMSE')
        axes[1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot MSE metrics
        sns.boxplot(data=data_subset, x='Activity', y='MSE', hue='Model', ax=axes[2], order=activity_order)
        axes[2].set_title(f'Boxplot of MSE for {target}')
        axes[2].set_ylabel('MSE')
        axes[2].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout for titles and legends
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
