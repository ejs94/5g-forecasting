import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    AutoARIMA,
    ExponentialSmoothing,
    NaiveMean,
)
from darts.utils.model_selection import train_test_split
from darts.utils.utils import ModelMode, SeasonalityMode

from pipeline_5g.utils import get_torch_device_config, save_historical_forecast_results

# ======================= MAIN =======================

print("---Verificando disponibilidade de GPU/CPU---")
torch_kwargs = get_torch_device_config()

print("---Carregando os dados 5G Dataset---")

base_data_path = os.path.join(os.curdir, "data")
processed_timeseries_path = os.path.join(base_data_path, "processed_timeseries")

results_path = os.path.join(base_data_path, "results")
os.makedirs(results_path, exist_ok=True)

models_trained_path = os.path.join(results_path, "models_trained")
os.makedirs(models_trained_path, exist_ok=True)

try:
    with open(
        os.path.join(processed_timeseries_path, "processed_targets.pkl"), "rb"
    ) as f:
        all_targets_cleaned = pickle.load(f)
except FileNotFoundError:
    print("ERRO: Arquivos de dados processados não encontrados. Verifique os caminhos.")
    exit()

scaler_target = Scaler()
target_ts_scaled = scaler_target.fit_transform(all_targets_cleaned)

train_ts, _ = train_test_split(target_ts_scaled, test_size=0.2, axis=1)
predict_horizon = 10

# Lista para guardar resultados
all_preds = []
fit_times = []
hf_times = []


for i, series in enumerate(train_ts):
    print(f"\n--- Treinando modelo local para série {i} ---")
    #  --- Modelos Naives ---
    # model = NaiveMean()
    #  --- Modelos Simples ---
    # model = ExponentialSmoothing(
    #     trend=ModelMode.NONE,
    #     damped=False,
    #     seasonal=SeasonalityMode.NONE,
    #     seasonal_periods=None,
    #     random_state=0,
    # )
    # --- Modelos Avançados ---
    model = AutoARIMA(
        start_p=0,                  # Valor inicial para o parâmetro AR (p) na busca stepwise
        start_q=0,                  # Valor inicial para o parâmetro MA (q) na busca stepwise
        max_p=5,                    # Valor máximo para o parâmetro AR (p)
        max_q=5,                    # Valor máximo para o parâmetro MA (q)
        d=None,                     # Deixa o modelo escolher automaticamente o número de diferenciações (d)
        seasonal=False,             # Não considera componentes sazonais (SARIMA)
        stepwise=True,              # Usa algoritmo stepwise para acelerar a seleção de modelos
        suppress_warnings=True,     # Suprime avisos de convergência e outros durante a busca
        error_action='ignore',      # Ignora erros de modelos inválidos ao invés de lançar exceções
        n_jobs=-1,                  # Usa todos os núcleos disponíveis se `stepwise=False` e `parallel=True` (aqui, sem efeito)
        trace=True                  # Mostra o progresso da busca: exibe modelos testados e critérios de avaliação (AIC, etc.)
    )
    start_fit = time.time()
    model.fit(series)
    fit_elapsed = time.time() - start_fit

    print(f"--- Validando (historical_forecast) para série {i} ---")
    start_hf = time.time()
    preds = model.historical_forecasts(
        series=target_ts_scaled[i],
        start=0.8,
        forecast_horizon=predict_horizon,
        stride=1,
        retrain=True,
        verbose=True,
    )
    hf_elapsed = time.time() - start_hf

    all_preds.append(preds)
    fit_times.append(fit_elapsed)
    hf_times.append(hf_elapsed)

# Desscala todas as previsões
model_name = model.__class__.__name__
historical_preds_unscaled = scaler_target.inverse_transform(all_preds)

# Exporta resultados
save_historical_forecast_results(
    model_name=model_name,
    historical_preds_unscaled=historical_preds_unscaled,
    all_targets_cleaned=all_targets_cleaned,
    fit_elapsed_time=np.mean(fit_times),
    hf_elapsed_time=np.mean(hf_times),
    results_path=results_path,
    mode="local_univariate",
)
