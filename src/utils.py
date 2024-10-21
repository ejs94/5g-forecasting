import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import shortuuid
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

from metrics import collect_univariate_metrics, compare_series_metrics, calculate_grouped_statistics


def extract_5G_dataset(path: os.path) -> list[pd.DataFrame]:

    df_static = []
    df_driving = []

    files = glob.glob(f"{path}/**/*.csv", recursive=True)

    for file in files:
        file = os.path.normpath(file)
        df = pd.read_csv(file)
        folder_name, filename = os.path.split(file)

        df["Uid"] = shortuuid.uuid()[:8]

        streaming_services = ["Netflix", "Amazon_Prime"]
        if any(service in folder_name for service in streaming_services):
            df["User_Activity"] = "Streaming Video"

        if ("Download") in folder_name:
            df["User_Activity"] = "Downloading a File"

        if "Static" in folder_name:
            df["Mobility"] = "Static"
            df_static.append(df)

        if "Driving" in folder_name:
            df["Mobility"] = "Driving"
            df_driving.append(df)

    df_static = pd.concat(df_static, axis=0, ignore_index=True)
    df_driving = pd.concat(df_driving, axis=0, ignore_index=True)

    return [df_static, df_driving]


def preprocess_5G_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "Latitude",
        "Longitude",
        "Operatorname",
        "CellID",
        "PINGAVG",
        "PINGMIN",
        "PINGMAX",
        "PINGSTDEV",
        "PINGLOSS",
        "CELLHEX",
        "NODEHEX",
        "LACHEX",
        "RAWCELLID",
        "NRxRSRP",
        "NRxRSRQ",
        "Mobility",
    ]
    cleaned = df.drop(cols_to_drop, axis=1)

    # Convert unkown string to datetime64
    # add TZ +1000 for Dublin, Ireland UTC
    cleaned["Timestamp"] = (
        cleaned["Timestamp"]
        .apply(
            lambda row: row[:9].replace(".", "-")
            + row[9:].replace(".", ":").replace("_", " ")
        )
        .astype("datetime64[ns]")
    )

    # Rename '-' to NaN values
    cleaned[["RSRQ", "SNR", "CQI", "RSSI"]] = cleaned[
        ["RSRQ", "SNR", "CQI", "RSSI"]
    ].replace("-", np.nan)

    # Change objects columns to int64 dtype
    # cleaned[["RSRQ","SNR","CQI", "RSSI"]] = cleaned[["RSRQ","SNR","CQI", "RSSI"]].astype(float).astype('Int64')
    cleaned[["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]] = cleaned[
        ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
    ].astype(float)

    # Configurar a coluna de data/hora como índice
    cleaned = cleaned.set_index("Timestamp")

    cleaned_dfs = []

    for uid in cleaned.Uid.unique():
        df_uid = cleaned[cleaned.Uid == uid]
        df_uid = df_uid[~df_uid.index.duplicated(keep="first")]
        cleaned_dfs.append(df_uid)

    cleaned = pd.concat(cleaned_dfs).sort_index()

    return cleaned


def compact_5G_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.reset_index()
    compact_df = (
        df.groupby("Uid")[["Timestamp", "RSRP", "RSRQ", "SNR", "CQI", "RSSI"]]
        .agg(lambda x: list(x))
        .reset_index()
    )
    return compact_df


def separate_by_uid_and_frequency(
    df: pd.DataFrame, target_columns: list, frequency: str
) -> list:
    """
    Separa o DataFrame com base nas colunas específicas e no 'Uid', retornando uma lista de DataFrames filtrados.
    A frequência temporal é obrigatória para ajustar os dados.

    Args:
        df (pd.DataFrame): DataFrame original contendo a coluna 'Uid'.
        target_columns (list): Lista de colunas a serem extraídas para cada Uid.
        frequency (str): Frequência temporal para ajustar os dados (obrigatório).

    Returns:
        list: Lista de DataFrames, um para cada Uid, contendo as colunas especificadas.
    """
    separated_dfs = []

    # Obtém a lista única de Uids e ordena
    unique_uids = sorted(df["Uid"].unique())

    # Itera sobre cada Uid e filtra os dados
    for uid in unique_uids:
        filtered_df = df[df["Uid"] == uid][target_columns].asfreq(freq=frequency)
        separated_dfs.append(filtered_df)

    return separated_dfs


def preprocess_list_ts(
    list_ts: List[TimeSeries],
) -> Tuple[List[TimeSeries], List[Scaler]]:
    """
    Preprocessa uma lista de séries temporais aplicando preenchimento de valores faltantes e escalonamento,
    e retorna a lista de séries transformadas junto com os escalonadores usados para cada série.

    Args:
        list_ts (List[TimeSeries]): Lista de séries temporais a serem transformadas.

    Returns:
        Tuple[List[TimeSeries], List[Scaler]]: Lista de séries temporais transformadas e seus respectivos escalonadores.
    """

    list_transformed = []
    scalers = []

    filler = MissingValuesFiller()

    for ts in list_ts:
        # Escalonador independente para cada série temporal
        scaler = Scaler()
        pipe = Pipeline([filler, scaler])

        transformed = pipe.fit_transform(ts)
        list_transformed.append(transformed)
        scalers.append(scaler)

    return list_transformed, scalers


def convert_dfs_to_ts(
    list_df: List[pd.DataFrame], target_columns: List[str]
) -> List[TimeSeries]:
    """
    Converte uma lista de DataFrames em uma lista de séries temporais (`TimeSeries`).

    Esta função recebe uma lista de DataFrames (`list_df`) e converte cada DataFrame
    em uma série temporal (`TimeSeries`) utilizando as colunas especificadas em `target_columns`.

    Args:
        list_df (List[pd.DataFrame]): Lista de DataFrames a serem convertidos.
        target_columns (List[str]): Lista de nomes das colunas que contêm os valores das séries temporais.

    Returns:
        List[TimeSeries]: Lista de objetos `TimeSeries` resultantes da conversão.
    """

    for i, df in enumerate(list_df):
        # Converter o DataFrame para TimeSeries usando as colunas especificadas
        list_df[i] = TimeSeries.from_dataframe(df, value_cols=target_columns)

    return list_df


def training_model_for_activity(
    activity: str,
    model_name: str,
    model,
    list_series,
    target_columns,
    output_file: str,
    K: int,
    H: int
) -> bool:
    """
    Trains a univariate model for a specific activity and indicates whether the operation was successful.

    Args:
        activity (str): Name of the activity to be evaluated.
        model_name (str): Name of the model to be used.
        model: The forecasting model to be applied.
        list_series (list): List of time series for the activity.
        target_columns (list): KPI columns to be evaluated.
        output_file (str): Name of the output file (without extension).
        K (int): Number of subsets for cross-validation.
        H (int): Forecast horizon.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    
    print(f"---{model_name} Forecast---")
    
    try:
        # Collect univariate metrics
        result = collect_univariate_metrics(
            activity, list_series, target_columns, model_name, model, K, H
        )

        if result is None:
            print(f"Warning: Result for {activity} using {model_name} is None.")
            return False  # Indicate failure if result is None
        
        # Define the path to save the file, adding the .parquet extension
        output_path = os.path.join(os.curdir, "data", "results", f"{output_file}.parquet")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the DataFrame in Parquet format
        result_record = pd.DataFrame(result)
        result_record.to_parquet(output_path, compression="gzip")

        print(f"Saved in {output_path}")

        return True  # Indicate success

    except Exception as e:
        print(f"Error processing activity '{activity}' with model '{model_name}': {e}")
        return False  # Indicate failure in case of an exception