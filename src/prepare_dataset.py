import os
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

def group_metrics_by_uid(df: pd.DataFrame, freq="s") -> pd.DataFrame:
    """
    Agrupa métricas por Uid e ajusta a frequência temporal,
    focando apenas em Uids que começam com 'trace_'.

    Args:
        df (pd.DataFrame): O DataFrame de entrada com colunas como 'Timestamp', 'Uid', e métricas.
                          O DataFrame deve ter 'Timestamp' como índice, ou ser capaz de ser convertido.
        freq (str): Frequência de reamostragem (ex: 's' para segundos, '1min' para 1 minuto).

    Returns:
        pd.DataFrame: Um DataFrame agrupado por Uid, com métricas como listas.
    """
    # Certifica-se de que o Timestamp é o índice para usar asfreq
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index('Timestamp')
        else:
            raise ValueError("O DataFrame deve ter um índice DatetimeIndex ou uma coluna 'Timestamp'.")

    # Filtra os Uids que começam com 'trace_'
    df_filtered_uids = df[df['Uid'].astype(str).str.startswith('trace_')]

    # Aplica asfreq e reseta o índice (agora o Timestamp será uma coluna novamente)
    df_resampled = df_filtered_uids.asfreq(freq=freq).reset_index()

    metrics = [
        "Timestamp", # Inclua o Timestamp aqui se quiser ele como uma lista de tempos
        "RSRP",
        "RSRQ",
        "SNR",
        "CQI",
        "RSSI",
        "DL_bitrate",
        "UL_bitrate",
        "Speed",
    ]

    # Garante que apenas as métricas existentes no DataFrame sejam usadas
    available_metrics = [m for m in metrics if m in df_resampled.columns]

    # Agrupa por Uid e agrega as métricas como listas
    grouped_df = df_resampled.groupby("Uid").agg({metric: list for metric in available_metrics}).reset_index()

    # Ordena o DataFrame pelo número extraído de 'Uid'
    grouped_df['sort_key'] = grouped_df['Uid'].str.extract('(\d+)').astype(int)
    grouped_df = grouped_df.sort_values(by='sort_key').drop(columns=['sort_key']).reset_index(drop=True)

    return grouped_df


def save_grouped_metrics_to_parquet(df, activity_name, output_dir, freq="s"):
    """
    Agrupa métricas por Uid e salva em formato parquet.
    """
    reduced_df = group_metrics_by_uid(df, freq)
    parquet_path = os.path.join(output_dir, f"{activity_name}_metrics.parquet")
    reduced_df.to_parquet(parquet_path)
    print(f"Arquivo parquet salvo em: {parquet_path}")


def process_reduced_metrics(
    list_static_strm, list_driving_strm, list_static_down, list_driving_down, output_dir
):
    """
    Processa e salva os datasets reduzidos.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_grouped_metrics_to_parquet(list_static_strm, "static_strm", output_dir)
    save_grouped_metrics_to_parquet(list_driving_strm, "driving_strm", output_dir)
    save_grouped_metrics_to_parquet(list_static_down, "static_down", output_dir)
    save_grouped_metrics_to_parquet(list_driving_down, "driving_down", output_dir)


def build_imputed_timeseries():
    """
    Carrega datasets reduzidos, concatena, converte para séries temporais,
    imputa valores ausentes e salva os resultados como arquivos .parquet.
    """
    data_dir = os.path.join(os.curdir, "data", "reduced_metrics_datasets")
    output_dir = os.path.join(os.curdir, "data", "processed_timeseries")
    os.makedirs(output_dir, exist_ok=True)

    file_names = {
        "static_down": "static_down_metrics.parquet",
        "static_strm": "static_strm_metrics.parquet",
        "driving_down": "driving_down_metrics.parquet",
        "driving_strm": "driving_strm_metrics.parquet",
    }

    # Carregamento dos arquivos .parquet
    print("[INFO] Lendo arquivos de métricas reduzidas...")
    dfs = []
    for label, filename in file_names.items():
        path = os.path.join(data_dir, filename)
        print(f"  - Carregando: {label} ({filename})")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        try:
            df = pd.read_parquet(path)
            df["source"] = label
            dfs.append(df)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar '{filename}': {e}")

    if not dfs:
        raise ValueError(
            "Nenhum DataFrame foi carregado. Verifique os arquivos de entrada."
        )

    # Concatenação dos dados
    output_df_total_path = os.path.join(output_dir, "df_total.parquet")
    df_total = pd.concat(dfs, ignore_index=True)
    df_total.to_parquet(output_df_total_path)
    print(f"[INFO] Total de amostras após concatenação: {len(df_total)}")
    
    # # Definições
    # target_col = ["DL_bitrate"]
    # covariates = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"]

    # traces_uid = []
    # targets_idx = []
    # targets_values = []
    # covariates_idx = []
    # covariates_values = []
    # print("[INFO] Gerando séries temporais...")
    # for idx, row in df_total.iterrows():
    #     try:
    #         # Recriar o DataFrame a partir das listas na linha
    #         ts_df_data = {col: row[col] for col in row.index if col not in ['Uid', 'source']}
          
    #         ts_df = pd.DataFrame(ts_df_data)

    #         target_ts = create_target_timeseries(
    #             ts_df, target_col, timestamp_col="Timestamp"
    #         )
    #         cov_ts = create_covariates_timeseries(
    #             ts_df, covariates, timestamp_col="Timestamp"
    #         )

    #         print(f"[INFO] Imputando valores ausentes em {row['Uid']}")

    #         target_ts = impute_timeseries_missing_values(target_ts)
    #         cov_ts = impute_timeseries_missing_values(cov_ts)

    #         traces_uid.append(row["Uid"])
    #         targets_idx.append(target_ts.time_index)
    #         targets_values.append(target_ts.values())
    #         covariates_idx.append(cov_ts.time_index)
    #         covariates_values.append(cov_ts.values())
    #     except Exception as e:
    #         print(f"[WARNING] Erro na linha {idx}: {e}")
    #         continue

    # print(f"[INFO] Séries temporais processadas com sucesso: {len(traces_uid)}")

    # # Salvando resultados

    # targets_df = pd.DataFrame({
    # "Uid": traces_uid,
    # "Targets_time_index": targets_idx,
    # "Targets": targets_values,
    # "Covariates_time_index": covariates_idx,
    # "Covariates": covariates_values
    # })

    # targets_df_path = os.path.join(output_dir, "processed_targets.parquet")
    # targets_df.to_parquet(targets_df_path)
    # print(f"[INFO] Targets salvos em: {targets_df_path}")

    target_col = ["DL_bitrate"]
    covariates = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"]

    normalized_traces = []  # iremos acumular dataframes já normalizados por Uid/Timestamp

    print("[INFO] Gerando séries temporais...")
    for idx, row in df_total.iterrows():
        try:
            # Recriar o DataFrame a partir das listas na linha
            ts_df_data = {col: row[col] for col in row.index if col not in ['Uid', 'source']}
            ts_df = pd.DataFrame(ts_df_data)

            # Cria séries de alvo e covariáveis (Darts TimeSeries, presumivelmente)
            target_ts = create_target_timeseries(ts_df, target_col, timestamp_col="Timestamp")
            cov_ts    = create_covariates_timeseries(ts_df, covariates, timestamp_col="Timestamp")

            print(f"[INFO] Imputando valores ausentes em {row['Uid']}")
            target_ts = impute_timeseries_missing_values(target_ts)
            cov_ts    = impute_timeseries_missing_values(cov_ts)

            # Converte para DataFrame do pandas
            # Darts fornece .pd_dataframe() com o time_index como index
            target_df = target_ts.pd_dataframe()      # colunas = target_col
            cov_df    = cov_ts.pd_dataframe()         # colunas = covariates

            df_join = target_df.join(cov_df, how="outer")

            # Garante nome do índice como 'Timestamp' e reseta para coluna
            df_join.index.name = "Timestamp"
            df_join = df_join.reset_index()

            # Anexa Uid
            df_join.insert(0, "Uid", row["Uid"])

            # (opcional) ordena temporalmente
            df_join = df_join.sort_values(["Uid", "Timestamp"])

            normalized_traces.append(df_join)

        except Exception as e:
            print(f"[WARNING] Erro na linha {idx}: {e}")
            continue

    print(f"[INFO] Séries temporais processadas com sucesso: {len(normalized_traces)}")

    # Salvando resultados em formato longo (1 linha por Uid-Timestamp)
    final_df = pd.concat(normalized_traces, ignore_index=True)

    # Garante dtypes adequados
    final_df["Timestamp"] = pd.to_datetime(final_df["Timestamp"], utc=False)
    # Salva em Parquet; particionar por Uid ajuda a ler por traço e reduz I/O
    output_path = os.path.join(output_dir,"processed_timeseries.parquet")
    final_df.to_parquet(output_path, engine="pyarrow")  # usar partition_cols=['Uid'] para múltiplos arquivos
    print(f"[INFO] Targets salvos em: {output_path}")

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
