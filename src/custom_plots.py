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
        if file.endswith(".parquet"):
            # Create the full path of the file
            file_path = os.path.join(folder, file)
            # Read the Parquet file and add it to the DataFrame
            df = pd.read_parquet(file_path)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def aggregate_median_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para calcular a mediana de MAE, RMSE, MSE, NRMSE e NMSE, agrupando por Model, target e Activity.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados com as colunas 'Model', 'target', 'Activity', 'MAE', 'RMSE', 'MSE', 'NRMSE', 'NMSE'.

    Retorna:
    pd.DataFrame: DataFrame agregado contendo as medianas das métricas MAE, RMSE, MSE, NRMSE e NMSE por Model, target e Activity.
    """

    # Verificar se as colunas necessárias estão presentes no DataFrame
    required_columns = [
        "Model",
        "target",
        "Activity",
        "MAE",
        "RMSE",
        "MSE",
        "NRMSE",
        "NMSE",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna {col} está ausente no DataFrame")

    # Agrupar por Model, target e Activity e calcular as medianas das métricas
    aggregated_df = df.groupby(["Model", "target", "Activity"], as_index=False).agg(
        {
            "MAE": "median",
            "RMSE": "median",
            "MSE": "median",
            "NRMSE": "median",
            "NMSE": "median",
        }
    )

    # Renomear as colunas agregadas para refletir que são medianas
    aggregated_df.rename(
        columns={
            "MAE": "MAE_Median",
            "RMSE": "RMSE_Median",
            "MSE": "MSE_Median",
            "NRMSE": "NRMSE_Median",
            "NMSE": "NMSE_Median",
        },
        inplace=True,
    )

    return aggregated_df


def plot_bar_for_medians_by_target(df: pd.DataFrame) -> None:
    """
    Generates bar plots for the metrics MAE_Median, RMSE_Median, MSE_Median, NRMSE_Median, and NMSE_Median,
    separated by each unique target in the DataFrame, with subplots for static and driving activities.

    Parameters:
    df (pd.DataFrame): The DataFrame that contains the data for the bar plots.

    Returns:
    None: Displays the bar plots for each metric and target with static and driving activities in subplots.
    """
    # Obtaining unique targets
    targets = df["target"].unique()

    # List of metrics to be plotted
    metrics = ["MAE_Median", "RMSE_Median", "MSE_Median", "NRMSE_Median", "NMSE_Median"]

    # Iterate over each unique target
    for target in targets:
        # Filtering data for the specific target
        target_data = df[df["target"] == target]

        # Iterate over each metric
        for metric in metrics:
            # Creating a figure with two subplots (static and driving)
            fig, axes = plt.subplots(1, 2, sharex=False)

            # Filter and sort data for static activities
            static_data = target_data[target_data["Activity"].str.contains("static")]
            static_data_sorted = static_data.sort_values(by=metric, ascending=True)
            sns.barplot(
                data=static_data_sorted,
                x="Activity",
                y=metric,
                hue="Model",
                ax=axes[0],
                order=["static_down", "static_strm"],
            )
            axes[0].set_title(f"Static: {metric} ({target})")
            axes[0].set_ylabel(metric)
            axes[0].legend(title="Model", loc="upper left")

            # Filter and sort data for driving activities
            driving_data = target_data[target_data["Activity"].str.contains("driving")]
            driving_data_sorted = driving_data.sort_values(by=metric, ascending=True)
            sns.barplot(
                data=driving_data_sorted,
                x="Activity",
                y=metric,
                hue="Model",
                ax=axes[1],
                order=["driving_down", "driving_strm"],
            )
            axes[1].set_title(f"Driving: {metric} ({target})")
            axes[1].set_ylabel(metric)
            axes[1].legend(title="Model", loc="upper left")

            # Adjusting layout and displaying the plot
            plt.xlabel("Activity")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def plot_boxplots_for_metrics_by_target(
    data: pd.DataFrame, show_outliers: bool = True
) -> None:
    """
    Function to generate boxplots of MAE, RMSE, MSE, NRMSE, and NMSE grouped by 'Model', 'target', and 'Activity',
    with separate plots for static and driving activities displayed side-by-side.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data with columns 'Model', 'target', 'Activity', 'MAE', 'RMSE', 'MSE', 'NRMSE', 'NMSE'.
    show_outliers (bool): If True, outliers will be displayed. If False, they will be hidden. Default is True.

    Returns:
    None: Displays the boxplot graphs.
    """
    # Define the desired order of activities
    static_activities = ["static_down", "static_strm"]
    driving_activities = ["driving_down", "driving_strm"]

    # Get unique targets
    targets = data["target"].unique()

    # Loop through the targets (e.g., CQI, RSRP, RSRQ, RSSI, SNR)
    for target in targets:
        # Filter the data for the current target
        data_subset = data[data["target"] == target]

        # Create subplots for static and driving activities side-by-side
        fig, axes = plt.subplots(5, 2, figsize=(18, 30))
        fig.suptitle(f"Forecasting Metrics for {target}", fontsize=16)

        # List of metrics
        metrics = ["MAE", "RMSE", "MSE", "NRMSE", "NMSE"]

        # Plot each metric for static and driving activities
        for i, metric in enumerate(metrics):
            # Static activities
            sns.boxplot(
                data=data_subset[data_subset["Activity"].isin(static_activities)],
                x="Activity",
                y=metric,
                hue="Model",
                ax=axes[i, 0],
                order=static_activities,
                showfliers=show_outliers,
            )
            axes[i, 0].set_title(f"Static: {metric}")
            axes[i, 0].set_ylabel(metric)
            axes[i, 0].legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))

            # Driving activities
            sns.boxplot(
                data=data_subset[data_subset["Activity"].isin(driving_activities)],
                x="Activity",
                y=metric,
                hue="Model",
                ax=axes[i, 1],
                order=driving_activities,
                showfliers=show_outliers,
            )
            axes[i, 1].set_title(f"Driving: {metric}")
            axes[i, 1].legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
