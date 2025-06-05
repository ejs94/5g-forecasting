import os
import pickle
import time
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from darts.dataprocessing.transformers import Scaler

from darts.models import LinearRegressionModel
from darts.utils.model_selection import train_test_split

from pipeline_5g.train_and_validate import train_time_series_model, walk_forward_validation
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
    with open(os.path.join(processed_timeseries_path, "processed_targets.pkl"), "rb") as f:
        all_targets_cleaned = pickle.load(f)
except FileNotFoundError:
    print("ERRO: Arquivos de dados processados não encontrados. Verifique os caminhos.")
    exit()

scaler_target = Scaler()

target_ts_scaled = scaler_target.fit_transform(all_targets_cleaned)

train_ts, _ = train_test_split(target_ts_scaled, test_size=0.2, axis=1)

predict_horizon = 10

model = LinearRegressionModel()

# Treinamento sem covariáveis
model, fit_elapsed_time, model_name = train_time_series_model(
    model=model,
    train_series=train_ts,
    train_covariates=None,  # Explicitamente None para não usar covariáveis
    base_data_path=base_data_path
)

# Validação sem covariáveis (covariate_series = None)
historical_preds_scaled, hf_elapsed_time = walk_forward_validation(
    model=model,
    target_series=target_ts_scaled,
    covariate_series=None,
    forecast_horizon=predict_horizon
)

historical_preds_unscaled = scaler_target.inverse_transform(historical_preds_scaled)

# Exporta resultados com modo univariate (sem covariáveis)
save_historical_forecast_results(
    model_name=model_name,
    historical_preds_unscaled=historical_preds_unscaled,
    all_targets_cleaned=all_targets_cleaned,
    fit_elapsed_time=fit_elapsed_time,
    hf_elapsed_time=hf_elapsed_time,
    results_path=results_path,
    mode="univariate"
)

# TODO: Adicionar os métodos naive
