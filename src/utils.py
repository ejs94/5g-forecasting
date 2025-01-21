import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import shortuuid
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

from metrics import evaluate_global_model


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


def train_and_evaluate_global_model(
    activity: str,
    model_name: str,
    model,
    list_series,
    target_columns,
    H: int,
) -> bool:
    """
    Treina e avalia um modelo global (N-BEATS, LSTM) para uma atividade específica,
    coletando métricas e salvando os resultados em um arquivo Parquet.

    Args:
        activity (str): Nome da atividade a ser avaliada.
        model_name (str): Nome do modelo a ser usado.
        model: O modelo de previsão global (N-BEATS, LSTM) a ser utilizado.
        list_series (list): Lista de séries temporais para a atividade.
        target_columns (list): Colunas de KPIs a serem avaliadas.
        output_file (str): Nome do arquivo de saída (sem extensão).
        H (int): Horizonte de previsão.

    Returns:
        bool: True se a operação foi bem-sucedida, False caso contrário.
    """
    output_file = (f"mult_{model_name}_{activity}",)

    print(f"---{model_name} Forecast---")

    try:
        result_records = []
        total_series = len(list_series)

        # Itera sobre cada série temporal
        for i, series in enumerate(list_series):
            print(f"---> Processando série {i}/{total_series - 1}... <---")
            try:
                # Avalia o modelo com a série temporal para o KPI específico
                results = evaluate_global_model(model, series, H, model_name)
                results["Activity"] = activity
                result_records.append(results)
            except Exception as e:
                print(f"Erro ao processar a série {i}: {e}")
                continue

        # Converte a lista de resultados em um DataFrame e retorna
        result = pd.DataFrame(result_records)

        if result.empty:
            print(f"Warning: Result for {activity} using {model_name} is empty.")
            return False  # Indica falha se o resultado for vazio

        # Define o caminho para salvar o arquivo Parquet
        output_path = os.path.join(
            os.curdir, "data", "results", "multivariate", f"{output_file}.parquet"
        )

        # Cria o diretório, se necessário
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salva os resultados em formato Parquet
        result.to_parquet(output_path, compression="gzip")

        print(f"Results saved at {output_path}")

        return True  # Indica sucesso

    except Exception as e:
        print(f"Error processing activity '{activity}' with model '{model_name}': {e}")
        return False  # Indica falha em caso de exceção
