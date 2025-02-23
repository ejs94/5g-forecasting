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
    Generates improved bar plots for the metrics MAE_Median, RMSE_Median, MSE_Median, NRMSE_Median, and NMSE_Median,
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

    full_model_order = [
        "Naive",
        "NaiveDrift",
        "NaiveMean",
        "NaiveMovingAverage",
        "LinearRegression",
        "ExponentialSmoothing",
        "ARIMA",
        "FFT",
        "Theta",
        "LightGBM",
        "LSTM",
        "NBEATS",
        "Prophet",
    ]

    # Desired label mapping
    label_mapping = {
        "Naive": "Naïve",
        "NaiveDrift": "Naïve Drift",
        "NaiveMean": "Naïve Mean",
        "NaiveMovingAverage": "Naïve Mov Avg",
        "LinearRegression": "Lin Regression",
        "ExponentialSmoothing": "ES",
        "ARIMA": "ARIMA",
        "FFT": "FFT",
        "Theta": "Theta",
        "LightGBM": "LightGBM",
        "LSTM": "LSTM",
        "NBEATS": "NBEATS",
        "Prophet": "Prophet",
    }

    # Determine models present in the DataFrame
    models_in_data = df["Model"].unique()
    model_order = [model for model in full_model_order if model in models_in_data]

    # Emit a warning for missing models
    missing_models = set(full_model_order) - set(models_in_data)
    if missing_models:
        print(
            f"[WARNING] The following models are missing in the DataFrame: {missing_models}"
        )

    # Define a custom color palette
    custom_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
    ]

    # Assign colors to models
    model_palette = dict(zip(model_order, custom_colors[: len(model_order)]))

    # Plotting the graphs
    for target in targets:
        target_data = df[df["target"] == target]

        for metric in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

            # Static activities
            static_data = target_data[target_data["Activity"].str.contains("static")]
            sns.barplot(
                data=static_data,
                x="Activity",
                y=metric,
                hue="Model",
                ax=axes[0],
                order=["static_down", "static_strm"],
                hue_order=model_order,
                palette=model_palette,
            )
            axes[0].set_title("Static")
            axes[0].set_ylabel(
                metric.replace("_", " ")
            )  # Replace _ with space for display
            axes[0].set_xticks(range(2))  # Define explicit tick positions
            axes[0].set_xticklabels(["Downloading", "Streaming"])
            axes[0].set_xlabel("")
            axes[0].legend().remove()

            # Driving activities
            driving_data = target_data[target_data["Activity"].str.contains("driving")]
            sns.barplot(
                data=driving_data,
                x="Activity",
                y=metric,
                hue="Model",
                ax=axes[1],
                order=["driving_down", "driving_strm"],
                hue_order=model_order,
                palette=model_palette,
            )
            axes[1].set_title("Driving")
            axes[1].set_xticks(range(2))  # Define explicit tick positions
            axes[1].set_xticklabels(["Downloading", "Streaming"])
            axes[1].set_xlabel("")
            axes[1].legend().remove()

            # Adjust layout and add a vertical legend with updated labels
            handles, labels = axes[0].get_legend_handles_labels()

            # Replace labels with the new mapped labels
            updated_labels = [label_mapping.get(label, label) for label in labels]

            fig.legend(
                handles,
                updated_labels,
                loc="center left",
                ncol=1,
                bbox_to_anchor=(0.9, 0.5),
                fontsize="small",
                title="Models",
            )

            plt.suptitle(
                f"Comparison of {metric.replace('_', ' ')} for {target}",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout(rect=[0, 0, 0.9, 0.92])
            plt.show()


def plot_boxplots_for_metrics_by_target(
    data: pd.DataFrame, show_outliers: bool = True
) -> None:
    """
    Generates boxplots for MAE, RMSE, MSE, NRMSE, and NMSE grouped by 'Model', 'target', and 'Activity',
    with separate plots for static and driving activities. Each metric gets its own figure.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data with columns 'Model', 'target', 'Activity', 'MAE', 'RMSE', 'MSE', 'NRMSE', 'NMSE'.
    show_outliers (bool): If True, outliers will be displayed. If False, they will be hidden. Default is True.

    Returns:
    None: Displays the boxplot graphs.
    """
    # Define a custom color palette
    custom_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
    ]

    # Desired label mapping
    label_mapping = {
        "Naive": "Naïve",
        "NaiveDrift": "Naïve Drift",
        "NaiveMean": "Naïve Mean",
        "NaiveMovingAverage": "Naïve Mov Avg",
        "LinearRegression": "Lin Regression",
        "ExponentialSmoothing": "ES",
        "ARIMA": "ARIMA",
        "FFT": "FFT",
        "Theta": "Theta",
        "LightGBM": "LightGBM",
        "LSTM": "LSTM",
        "NBEATS": "NBEATS",
        "Prophet": "Prophet",
    }

    # Define the desired order of models
    model_order = [
        "Naive",
        "NaiveDrift",
        "NaiveMean",
        "NaiveMovingAverage",  # Baseline benchmarks
        "LinearRegression",
        "ExponentialSmoothing",  # Standard benchmarks
        "ARIMA",
        "FFT",
        "Theta",  # Statistical methods
        "LightGBM",
        "LSTM",
        "NBEATS",
        "Prophet",  # Machine Learning models
    ]

    # Define the desired order of activities
    static_activities = ["static_down", "static_strm"]
    driving_activities = ["driving_down", "driving_strm"]

    # List of metrics
    metrics = ["MAE", "RMSE", "MSE", "NRMSE", "NMSE"]

    # Get unique targets
    targets = data["target"].unique()

    # Assign colors to models using the custom color palette
    model_palette = dict(zip(model_order, custom_colors[: len(model_order)]))

    # Loop through the targets (e.g., CQI, RSRP, RSRQ, RSSI, SNR)
    for target in targets:
        # Filter the data for the current target
        data_subset = data[data["target"] == target]

        for metric in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Static activities
            static_data = data_subset[data_subset["Activity"].isin(static_activities)]
            sns.boxplot(
                data=static_data,
                x="Activity",
                y=metric,
                hue="Model",
                hue_order=[
                    m for m in model_order if m in static_data["Model"].unique()
                ],
                ax=axes[0],
                order=static_activities,
                showfliers=show_outliers,
                palette=model_palette,
            )
            axes[0].set_title("Static")
            axes[0].set_ylabel(metric)
            axes[0].set_xticks(range(2))
            axes[0].set_xticklabels(["Downloading", "Streaming"])
            axes[0].set_xlabel("")
            axes[0].legend().remove()

            # Driving activities
            driving_data = data_subset[data_subset["Activity"].isin(driving_activities)]
            sns.boxplot(
                data=driving_data,
                x="Activity",
                y=metric,
                hue="Model",
                hue_order=[
                    m for m in model_order if m in driving_data["Model"].unique()
                ],
                ax=axes[1],
                order=driving_activities,
                showfliers=show_outliers,
                palette=model_palette,
            )
            axes[1].set_title("Driving")
            axes[1].set_xticks(range(2))
            axes[1].set_xticklabels(["Downloading", "Streaming"])
            axes[1].set_xlabel("")
            axes[1].legend().remove()

            # Ajusta espaçamento entre os gráficos
            plt.subplots_adjust(wspace=0.3)

            # Ajuste da legenda global
            handles, labels = axes[1].get_legend_handles_labels()
            # Replace labels with the new mapped labels
            updated_labels = [label_mapping.get(label, label) for label in labels]

            fig.legend(
                handles,
                updated_labels,
                loc="center left",
                ncol=1,
                bbox_to_anchor=(0.9, 0.5),
                fontsize="small",
                title="Models",
            )

            plt.suptitle(
                f"Comparison of {metric} for {target}", fontsize=16, fontweight="bold"
            )
            plt.tight_layout(rect=[0, 0, 0.9, 0.92])
            plt.show()
