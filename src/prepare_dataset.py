import os
import pickle
import time

import pandas as pd

from pipeline_5g.utils import (
    create_covariates_timeseries,
    create_target_timeseries,
    extract_5G_dataset,
    impute_timeseries_missing_values,
    preprocess_5G_dataframe,
)


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
    metrics = [
        "Timestamp",
        "RSRP",
        "RSRQ",
        "SNR",
        "CQI",
        "RSSI",
        "DL_bitrate",
        "UL_bitrate",
        "Speed",
    ]
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


def build_imputed_timeseries():
    """
    Carrega datasets reduzidos, concatena, converte para séries temporais,
    imputa valores ausentes e salva os resultados como arquivos .pkl.
    """
    data_dir = os.path.join(os.curdir, "data", "reduced_metrics_datasets")
    output_dir = os.path.join(os.curdir, "data", "processed_timeseries")
    os.makedirs(output_dir, exist_ok=True)

    file_names = {
        "static_down": "static_down_metrics.pkl",
        "static_strm": "static_strm_metrics.pkl",
        "driving_down": "driving_down_metrics.pkl",
        "driving_strm": "driving_strm_metrics.pkl",
    }

    # Carregamento dos arquivos .pkl
    print("[INFO] Lendo arquivos de métricas reduzidas...")
    dfs = []
    for label, filename in file_names.items():
        path = os.path.join(data_dir, filename)
        print(f"  - Carregando: {label} ({filename})")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
                df["source"] = label
                dfs.append(df)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar '{filename}': {e}")

    if not dfs:
        raise ValueError(
            "Nenhum DataFrame foi carregado. Verifique os arquivos de entrada."
        )

    # Concatenação dos dados
    df_total = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total de amostras após concatenação: {len(df_total)}")
    
    # Definições
    target_col = "DL_bitrate"
    covariates = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"]

    targets, covariates_ts = [], []

    print("[INFO] Gerando séries temporais e imputando valores ausentes...")
    for idx, row in df_total.iterrows():
        try:
            target_ts = create_target_timeseries(
                row, target_col, timestamp_col="Timestamp"
            )
            cov_ts = create_covariates_timeseries(
                row, covariates, timestamp_col="Timestamp"
            )

            target_ts = impute_timeseries_missing_values(target_ts)
            cov_ts = impute_timeseries_missing_values(cov_ts)

            targets.append(target_ts)
            covariates_ts.append(cov_ts)
        except Exception as e:
            print(f"[WARNING] Erro na linha {idx}: {e}")
            continue

    print(f"[INFO] Séries temporais processadas com sucesso: {len(targets)}")

    # Salvando resultados
    targets_path = os.path.join(output_dir, "processed_targets.pkl")
    covariates_path = os.path.join(output_dir, "processed_covariates.pkl")

    with open(targets_path, "wb") as f:
        pickle.dump(targets, f)
    print(f"[INFO] Targets salvos em: {targets_path}")

    with open(covariates_path, "wb") as f:
        pickle.dump(covariates_ts, f)
    print(f"[INFO] Covariáveis salvas em: {covariates_path}")


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

    build_imputed_timeseries()


if __name__ == "__main__":
    # Adicionar um timer para medir o tempo de execução total
    script_start_time = time.time()
    main()
    script_end_time = time.time()
    print(
        f"Tempo total de execução do script: {script_end_time - script_start_time:.2f} segundos."
    )
