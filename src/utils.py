import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

from metrics import evaluate_global_model


def compact_5G_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    compact_df = (
        df.groupby("Uid")[["Timestamp", "RSRP", "RSRQ", "SNR", "CQI", "RSSI"]]
        .agg(lambda x: list(x))
        .reset_index()
    )
    return compact_df


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
