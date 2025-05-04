import json
import os
import pickle
import shutil
import numpy as np
import pandas as pd
from darts import TimeSeries
from utils import (
    convert_dfs_to_ts,
    extract_5G_dataset,
    preprocess_5G_dataframe,
    separate_by_uid_and_frequency,
)


def load_or_create_config(config_path):
    """
    Carrega a configuração do arquivo JSON ou cria uma configuração padrão.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Lendo configuração existente de: {config_path}")
    else:
        config = {
            "test_ratio": 0.1,
            "update_interval": 10,
            "target_columns": ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"],
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Configuração inicial salva em: {config_path}")
    return config


def save_dataframe_to_parquet(df, path):
    """
    Salva um DataFrame no formato Parquet com compressão gzip.
    """
    df.to_parquet(path, compression="gzip")
    print(f"Dados salvos em: {path}")


def group_metrics_by_uid(df, freq="s"):
    """
    Agrupa métricas por Uid e ajusta a frequência temporal.
    """
    df = df.asfreq(freq=freq).reset_index()
    metrics = ["Timestamp", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate", "Speed"]
    return df.groupby("Uid").agg({metric: list for metric in metrics}).reset_index()


def save_grouped_metrics_to_pickle(df, activity_name, output_dir, freq="s"):
    """
    Agrupa métricas por Uid e salva em formato pickle.
    """
    reduced_df = group_metrics_by_uid(df, freq)
    pickle_path = os.path.join(output_dir, f"{activity_name}_metrics.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(reduced_df, f)
    print(f"Arquivo pickle salvo em: {pickle_path}")


def process_reduced_metrics(
    list_static_strm, list_driving_strm, list_static_down, list_driving_down, output_dir
):
    """
    Processa e salva os datasets reduzidos.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_grouped_metrics_to_pickle(list_static_strm, "static_strm", output_dir)
    save_grouped_metrics_to_pickle(list_driving_strm, "driving_strm", output_dir)
    save_grouped_metrics_to_pickle(list_static_down, "static_down", output_dir)
    save_grouped_metrics_to_pickle(list_driving_down, "driving_down", output_dir)


def main():
    """
    Função principal para executar o processamento do dataset.
    """
    # config_path = os.path.join(os.curdir, "config.json")
    data_path = os.path.join(os.curdir, "data")
    original_path = os.path.join(os.curdir, "datasets", "5G-production-dataset")
    reduced_output_dir = os.path.join(data_path, "reduced_metrics_datasets")

    # config = load_or_create_config(config_path)

    print("--- Extraindo e Preprocessando o 5G Dataset ---")
    df_static, df_driving = extract_5G_dataset(original_path)
    df_static = preprocess_5G_dataframe(df_static)
    df_driving = preprocess_5G_dataframe(df_driving)

    save_dataframe_to_parquet(
        df_static, os.path.join(data_path, "5G_df_static.parquet")
    )
    save_dataframe_to_parquet(
        df_driving, os.path.join(data_path, "5G_df_driving.parquet")
    )

    df_static = pd.read_parquet(os.path.join(data_path, "5G_df_static.parquet"))
    df_driving = pd.read_parquet(os.path.join(data_path, "5G_df_driving.parquet"))

    print("--- Separando os conjuntos em Streaming vs. Downloading ---")
    list_static_strm = df_static.query("User_Activity == 'Streaming Video'")
    list_driving_strm = df_driving.query("User_Activity == 'Streaming Video'")
    list_static_down = df_static.query("User_Activity == 'Downloading a File'")
    list_driving_down = df_driving.query("User_Activity == 'Downloading a File'")

    print("--- Criando datasets com as métricas reduzidas ---")
    process_reduced_metrics(
        list_static_strm,
        list_driving_strm,
        list_static_down,
        list_driving_down,
        reduced_output_dir,
    )


if __name__ == "__main__":
    main()
