import glob
import json
import os

import numpy as np
import pandas as pd
import shortuuid
from darts import TimeSeries


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

def save_historical_forecast_results(
    model_name: str,
    historical_preds_unscaled: list[TimeSeries],
    all_targets_cleaned: list[TimeSeries],
    fit_elapsed_time: float,
    hf_elapsed_time: float,
    results_path: str,
    mode: str = "covariate"
) -> str:
    """
    Salva os resultados das previsões históricas em formato Parquet.

    Parâmetros:
    ------------
    model_name : str
        Nome da classe do modelo.
    historical_preds_unscaled : List[TimeSeries]
        Lista de séries previstas (não normalizadas).
    all_targets_cleaned : List[TimeSeries]
        Lista de séries reais (não normalizadas).
    fit_elapsed_time : float
        Tempo total de treino do modelo (em segundos).
    hf_elapsed_time : float
        Tempo total gasto para gerar os forecasts históricos (em segundos).
    results_path : str
        Caminho onde os resultados serão salvos.
    mode : str, opcional
        Define o tipo de modelo: "covariate" (padrão) ou "univariate".
        Utilizado como prefixo no nome do arquivo salvo.

    Retorna:
    --------
    str
        Caminho completo para o arquivo Parquet salvo.
    """
    results_rows = []

    for i, preds in enumerate(historical_preds_unscaled):
        actuals = all_targets_cleaned[i]
        actuals_aligned = actuals.slice_intersect(preds)

        results_rows.append({
            "Model": model_name,
            "Series_id": i,
            "Fit_elapsed_time": fit_elapsed_time,
            "Historical_Forecast_elapsed_time": hf_elapsed_time,
            "Actuals_index": list(actuals_aligned.time_index),
            "Actuals_values": actuals_aligned.values().flatten().tolist(),
            "Preds_index": list(preds.time_index),
            "Preds_values": preds.values().flatten().tolist()
        })

    results_df = pd.DataFrame(results_rows)
    os.makedirs(results_path, exist_ok=True)

    prefix = f"{mode.lower()}_" if mode else ""
    results_file = os.path.join(results_path, f"{prefix}{model_name}_historical_forecast.parquet")
    results_df.to_parquet(results_file, compression="gzip")

    print(f"Resultados salvos em: {results_file}")
    return results_file
