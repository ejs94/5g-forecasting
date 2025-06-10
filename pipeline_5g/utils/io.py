import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shortuuid
from darts import TimeSeries


def extract_5G_dataset(path: Path) -> list[pd.DataFrame]:
    """
    Extrai e organiza datasets 5G a partir de arquivos CSV em um diretório.

    Classifica os dados em duas categorias de mobilidade: "Static" e "Driving",
    adiciona uma coluna de atividade do usuário baseada no nome do diretório
    e inclui um identificador único por amostra.

    Parâmetros:
    -----------
    path : Path
        Caminho base contendo os arquivos CSV.

    Retorna:
    --------
    list[pd.DataFrame]
        Lista contendo dois DataFrames: [dados_estáticos, dados_em_movimento].
    """
    df_static: list[pd.DataFrame] = []
    df_driving: list[pd.DataFrame] = []

    files = glob.glob(f"{path}/**/*.csv", recursive=True)

    for file in files:
        file = os.path.normpath(file)
        df = pd.read_csv(file)
        folder_name, filename = os.path.split(file)

        df["Uid"] = shortuuid.uuid()[:8]

        streaming_services = ["Netflix", "Amazon_Prime"]
        if any(service in folder_name for service in streaming_services):
            df["User_Activity"] = "Streaming Video"

        if "Download" in folder_name:
            df["User_Activity"] = "Downloading a File"

        if "Static" in folder_name:
            df["Mobility"] = "Static"
            df_static.append(df)

        if "Driving" in folder_name:
            df["Mobility"] = "Driving"
            df_driving.append(df)

    df_static_concat = pd.concat(df_static, axis=0, ignore_index=True)
    df_driving_concat = pd.concat(df_driving, axis=0, ignore_index=True)

    return [df_static_concat, df_driving_concat]


def load_or_create_config(config_path: str) -> dict[str, Any]:
    """
    Carrega configurações de um arquivo JSON ou cria configurações padrão.

    Parâmetros:
    -----------
    config_path : str
        Caminho do arquivo de configuração JSON.

    Retorna:
    --------
    dict[str, Any]
        Dicionário com os parâmetros de configuração.
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
    mode: str = ""
) -> str:
    """
    Salva os resultados das previsões históricas em formato Parquet.

    Args:
        model_name: Nome da classe do modelo.
        historical_preds_unscaled: Lista de séries previstas (não normalizadas).
        all_targets_cleaned: Lista de séries reais (não normalizadas).
        fit_elapsed_time: Tempo total de treino do modelo (em segundos).
        hf_elapsed_time: Tempo total gasto para gerar os forecasts históricos (em segundos).
        results_path: Caminho onde os resultados serão salvos.
        mode: Tipo de modelo: "covariate" (padrão) ou "univariate".

    Returns:
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
            "Actuals_index": actuals_aligned.time_index.astype(str).tolist(),
            "Actuals_values": actuals_aligned.values().flatten().tolist(),
            "Preds_index": preds.time_index.astype(str).tolist(),
            "Preds_values": preds.values().flatten().tolist()
        })

    results_df = pd.DataFrame(results_rows)
    os.makedirs(results_path, exist_ok=True)

    prefix = f"{mode.lower()}_" if mode else ""
    results_file = os.path.join(results_path, f"{prefix}{model_name}_historical_forecast.parquet")
    results_df.to_parquet(results_file, compression="gzip")

    print(f"Resultados salvos em: {results_file}")
    return results_file
