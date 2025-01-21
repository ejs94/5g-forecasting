import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
from darts import TimeSeries

# Importa funções do módulo utils
from utils import (
    convert_dfs_to_ts,
    extract_5G_dataset,
    preprocess_5G_dataframe,
    separate_by_uid_and_frequency,
)

# Caminhos
config_path = os.path.join(os.curdir, "config.json")
data_path = os.path.join(os.curdir, "data")
original_path = os.path.join(os.curdir, "datasets", "5G-production-dataset")

# Verifica se o arquivo de configuração já existe; caso contrário, cria um
if not os.path.exists(config_path):
    config = {
        "H": 10,  # Tamanho da janela de previsão
        "K": 50,  # Tamanho da janela de entrada
        "update_interval": 10,  # Atualização da janela deslizante
        "target_columns": ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuração inicial salva em: {config_path}")
else:
    print(f"Lendo configuração existente de: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

print("---Configuração Utilizada---")
print(config)

# Extração e pré-processamento do dataset 5G
print("---Extraindo o 5G Dataset---")
df_static, df_driving = extract_5G_dataset(original_path)

print("---Preprocessando o 5G Dataset---")
df_static = preprocess_5G_dataframe(df_static)
df_driving = preprocess_5G_dataframe(df_driving)

# Salvando os dados pré-processados
print("---Salvando o 5G Dataset Processado---")
os.makedirs(data_path, exist_ok=True)

df_static_save_path = os.path.join(data_path, "5G_df_static.parquet")
df_static.to_parquet(df_static_save_path, compression="gzip")
print(f"Dados estáticos salvos em: {df_static_save_path}")

df_driving_save_path = os.path.join(data_path, "5G_df_driving.parquet")
df_driving.to_parquet(df_driving_save_path, compression="gzip")
print(f"Dados em movimento salvos em: {df_driving_save_path}")

# Carregando os dados salvos
print("---Carregando os dados preprocessados---")
df_static = pd.read_parquet(df_static_save_path)
df_driving = pd.read_parquet(df_driving_save_path)

# Separando os conjuntos em Streaming e Downloading
print("---Separando os conjuntos em: Streaming vs. Downloading---")
list_static_strm = df_static.query("User_Activity == 'Streaming Video'")
list_driving_strm = df_driving.query("User_Activity == 'Streaming Video'")
list_static_down = df_static.query("User_Activity == 'Downloading a File'")
list_driving_down = df_driving.query("User_Activity == 'Downloading a File'")


# ---
# Gera Conjuntos de Dados com abordagem de redução em formato pandas
# Definir o caminho da pasta de saída comum para todas as métricas
reduced_output_dir = os.path.join(data_path, "reduced_metrics_datasets")

if os.path.exists(reduced_output_dir):
    shutil.rmtree(reduced_output_dir)
    print(f"Pasta {reduced_output_dir} deletada.")

os.makedirs(reduced_output_dir, exist_ok=True)


def group_metrics_by_uid(df: pd.DataFrame, freq: str = "s") -> pd.DataFrame:
    """
    Agrupa métricas em listas agrupadas por Uid, com suporte para ajuste de frequência temporal.

    Args:
        df (pd.DataFrame): DataFrame original contendo os dados.
        metrics (list): Lista de métricas (colunas) a serem agrupadas em listas.
        freq (str): Frequência para ajuste temporal com `asfreq`. Padrão é "s" (segundos).

    Returns:
        pd.DataFrame: DataFrame reduzido com as métricas agregadas em listas por Uid.
    """
    # Ajustar a frequência do índice temporal
    df = df.asfreq(freq=freq)

    # Resetar o índice para incluir o Timestamp como coluna
    df = df.reset_index()

    # Agrupar por Uid e agregar métricas em listas
    metrics_to_reduce = ["Timestamp", "RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
    reduced_df = (
        df.groupby("Uid")
        .agg({metric: list for metric in metrics_to_reduce})
        .reset_index()
    )

    return reduced_df


def save_grouped_metrics_to_pickle(
    df: pd.DataFrame, activity_name: str, freq: str = "s"
) -> None:
    """
    Agrupa as métricas por Uid e salva o resultado em arquivos pickle em uma pasta comum para todas as métricas.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        activity_name (str): Nome da atividade (Streaming ou Downloading).
        freq (str): Frequência para ajuste temporal com `asfreq`. Padrão é "s" (segundos).
    """
    # Ajustar a frequência do índice temporal
    df = df.asfreq(freq=freq)

    # Resetar o índice para incluir o Timestamp como coluna
    df = df.reset_index()

    # Agrupar por Uid e agregar métricas em listas
    metrics_to_reduce = ["Timestamp", "RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
    reduced_df = (
        df.groupby("Uid")
        .agg({metric: list for metric in metrics_to_reduce})
        .reset_index()
    )

    # Caminho do arquivo pickle para salvar o DataFrame
    pickle_file_path = os.path.join(reduced_output_dir, f"{activity_name}_metrics.pkl")

    # Salvar o DataFrame em um arquivo pickle
    with open(pickle_file_path, "wb") as f:
        pickle.dump(reduced_df, f)

    print(f"Arquivo pickle salvo em: {pickle_file_path}")


# Exemplo de como salvar os DataFrames de Streaming e Downloading
print("---Criando datasets com as métricas reduzidas---")
# Salvar os DataFrames de Streaming
save_grouped_metrics_to_pickle(list_static_strm, "static_strm")
save_grouped_metrics_to_pickle(list_driving_strm, "driving_strm")

# Salvar os DataFrames de Downloading
save_grouped_metrics_to_pickle(list_static_down, "static_down")
save_grouped_metrics_to_pickle(list_driving_down, "driving_down")


# ---
# Gera Conjunto de dados em janelas em formato timeseries
# Separando os conjuntos por UID único
print("---Separando os conjuntos por UID único---")
list_static_strm = separate_by_uid_and_frequency(
    list_static_strm, config["target_columns"], "s"
)
list_driving_strm = separate_by_uid_and_frequency(
    list_driving_strm, config["target_columns"], "s"
)
list_static_down = separate_by_uid_and_frequency(
    list_static_down, config["target_columns"], "s"
)
list_driving_down = separate_by_uid_and_frequency(
    list_driving_down, config["target_columns"], "s"
)

print("---Convertendo Dataframe para Timeseries (Darts)---")
list_static_strm = convert_dfs_to_ts(list_static_strm, config["target_columns"])
list_driving_strm = convert_dfs_to_ts(list_driving_strm, config["target_columns"])
list_static_down = convert_dfs_to_ts(list_static_down, config["target_columns"])
list_driving_down = convert_dfs_to_ts(list_driving_down, config["target_columns"])

# Mapeamento das listas para suas respectivas atividades
activities = {
    "static_strm": list_static_strm,
    "driving_strm": list_driving_strm,
    "static_down": list_static_down,
    "driving_down": list_driving_down,
}

# Criando datasets de Sliding Window para cada atividade
print("---Criando datasets de Sliding Window---")
sliding_window_path = os.path.join(data_path, "sliding_window_datasets")

# Deleta a pasta existente, se ela já estiver lá
if os.path.exists(sliding_window_path):
    shutil.rmtree(sliding_window_path)
    print(f"Pasta {sliding_window_path} deletada.")

# Recria a pasta
os.makedirs(sliding_window_path, exist_ok=True)
print(f"Pasta {sliding_window_path} recriada.")

for activity, series_list in activities.items():
    print(f"Processando {activity}...")
    sliding_data = []

    for ts in series_list:
        # Extrair os dados e o índice de tempo da série temporal transformada
        data = ts.values()
        times = ts.time_index

        num_windows = (len(data) - config["K"] - config["H"]) // config[
            "update_interval"
        ] + 1

        if num_windows <= 0:
            continue

        for i in range(num_windows):
            start_idx = i * config["update_interval"]
            train_end = start_idx + config["K"]
            test_end = train_end + config["H"]

            if test_end > len(data):
                break  # Para se não houver dados suficientes para a janela de teste

            # Criar as janelas de treino e teste com intervalo de 60 segundos
            ts_train = ts[start_idx:train_end]

            ts_test = ts[train_end:test_end]

            # Adiciona os dados com os nomes dos componentes
            sliding_data.append(
                {
                    "train": ts_train,
                    "test": ts_test,
                }
            )

    if sliding_data:
        # Salvar os objetos TimeSeries com Pickle
        pickle_file_path = os.path.join(
            sliding_window_path, f"{activity}_sliding_window.pkl"
        )
        with open(pickle_file_path, "wb") as f:
            pickle.dump(sliding_data, f)
        print(f"Dataset {activity} salvo em formato Pickle em: {pickle_file_path}")
    else:
        print(f"Nenhum dado gerado para {activity}.")

print("---Processamento de Sliding Window Finalizado---")
