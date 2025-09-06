import os
import time

import numpy as np
import pandas as pd
from darts import TimeSeries
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

# base_data_path = os.path.join(os.curdir, "data")
# processed_timeseries_path = os.path.join(base_data_path, "processed_timeseries")

# results_path = os.path.join(base_data_path, "results")
# os.makedirs(results_path, exist_ok=True)

# models_trained_path = os.path.join(results_path, "models_trained")
# os.makedirs(models_trained_path, exist_ok=True)

# def df_to_series_list(df, id_col_name="id", time_col="Timestamp"):
#     """
#     Converts a DataFrame with a time column and an id column into a list of TimeSeries objects.
#     """
#     series_list = []
#     for i in sorted(df[id_col_name].unique()):
#         series_df = df[df[id_col_name] == i].drop(columns=[id_col_name])
#         series = TimeSeries.from_dataframe(series_df, time_col=time_col)
#         series_list.append(series)
#     return series_list

# try:
#     # Load target
#     targets_df = pd.read_parquet(os.path.join(processed_timeseries_path, "processed_targets.parquet"))
#     all_targets_cleaned = df_to_series_list(targets_df)

# except FileNotFoundError:
#     print(
#         "ERRO: Arquivos de dados processados (processed_targets.parquet) não encontrados. Verifique os caminhos."
#     )
#     exit()

base_data_path = os.path.join(os.curdir, "data")
processed_timeseries_path = os.path.join(base_data_path, "processed_timeseries")
results_path = os.path.join(base_data_path, "results")
os.makedirs(results_path, exist_ok=True)


parquet_path = os.path.join(processed_timeseries_path, "processed_timeseries.parquet")
if not os.path.exists(parquet_path):
    print(f"ERRO: Arquivo não encontrado: {parquet_path}")
    raise SystemExit(1)

df_long = pd.read_parquet(parquet_path)
df_long["Timestamp"] = pd.to_datetime(df_long["Timestamp"], utc=False)
df_long = df_long.sort_values(["Uid", "Timestamp"])

target_col = "DL_bitrate"

all_targets_cleaned, all_uids = [], []
for uid, g in df_long.groupby("Uid", sort=True):
    g_u = g[["Timestamp", target_col]].dropna()
    if g_u.empty:
        continue
    ts = TimeSeries.from_dataframe(g_u, time_col="Timestamp", value_cols=[target_col])
    if len(ts) < 5:  # filtro mínimo simples
        continue
    all_targets_cleaned.append(ts)
    all_uids.append(uid)

if not all_targets_cleaned:
    print("ERRO: Nenhuma série alvo construída a partir do parquet longo.")
    raise SystemExit(1)


scaler_target = Scaler()
target_ts_scaled = scaler_target.fit_transform(all_targets_cleaned)

train_ts, _ = train_test_split(target_ts_scaled, test_size=0.2, axis=1)

predict_horizon = 10

# Lista para guardar resultados
success_preds_scaled = []
success_targets_unscaled = []
success_uids = []
fit_times, hf_times = [], []

for i, series_tr in enumerate(train_ts):

    uid = all_uids[i]
    series_full = target_ts_scaled[i]          # série completa (p/ HF)
    series_full_unscaled = all_targets_cleaned[i]  # p/ salvar 'actuals'

    print(f"\n--- Treinando modelo local para série {uid} ---")
    #  --- Modelos Naives ---
    model = NaiveMean()
    #  --- Modelos Simples ---
    # model = ExponentialSmoothing(
    #     trend=ModelMode.NONE,
    #     damped=False,
    #     seasonal=SeasonalityMode.NONE,
    #     seasonal_periods=None,
    #     random_state=0,
    # )
    # --- Modelos Avançados ---
    # model = AutoARIMA(
    #     start_p=0,                  # Valor inicial para o parâmetro AR (p) na busca stepwise
    #     start_q=0,                  # Valor inicial para o parâmetro MA (q) na busca stepwise
    #     max_p=5,                    # Valor máximo para o parâmetro AR (p)
    #     max_q=5,                    # Valor máximo para o parâmetro MA (q)
    #     d=None,                     # Deixa o modelo escolher automaticamente o número de diferenciações (d)
    #     seasonal=False,             # Não considera componentes sazonais (SARIMA)
    #     stepwise=True,              # Usa algoritmo stepwise para acelerar a seleção de modelos
    #     # suppress_warnings=True,     # Suprime avisos de convergência e outros durante a busca
    #     # error_action='ignore',      # Ignora erros de modelos inválidos ao invés de lançar exceções
    #     # n_jobs=-1,                  # Usa todos os núcleos disponíveis se `stepwise=False` e `parallel=True` (aqui, sem efeito)
    #     trace=True                  # Mostra o progresso da busca: exibe modelos testados e critérios de avaliação (AIC, etc.)
    # )
    
    # fit
    try:
        # model = model_builder()
        t0 = time.time()
        model.fit(series_tr)
        fit_times.append(time.time() - t0)
    except Exception as e:
        print(f"[WARN] fit falhou em {uid}: {e}")
        continue

    print(f"--- Validando (historical_forecast) para série {uid} ---")
    # historical forecasts
    try:
        t0 = time.time()
        preds = model.historical_forecasts(
            series=series_full,
            start=0.8,                    # 80% do histórico como início das previsões
            forecast_horizon=predict_horizon,
            stride=1,
            retrain=True,
            verbose=False,
        )
        hf_times.append(time.time() - t0)
        if preds is None or len(preds) == 0:
            print(f"[WARN] HF vazia em {uid}")
            continue
    except Exception as e:
        print(f"[WARN] HF falhou em {uid}: {e}")
        continue

    # success_preds_scaled.append(preds)
    # fit_times.append(fit_elapsed)
    # hf_times.append(hf_elapsed)

    
    # acumula somente os casos bem-sucedidos
    success_preds_scaled.append(preds)
    success_targets_unscaled.append(series_full_unscaled)
    success_uids.append(uid)

# Desscala todas as previsões
model_name = model.__class__.__name__
historical_preds_unscaled = scaler_target.inverse_transform(success_preds_scaled)

# Exporta resultados
# save_historical_forecast_results(
#     model_name=model_name,
#     historical_preds_unscaled=historical_preds_unscaled,
#     all_targets_cleaned=all_targets_cleaned,
#     fit_elapsed_time=np.mean(fit_times),
#     hf_elapsed_time=np.mean(hf_times),
#     results_path=results_path,
#     mode="local_univariate",
# )

save_historical_forecast_results(
    model_name=model_name,
    historical_preds_unscaled=historical_preds_unscaled,
    all_targets_cleaned=success_targets_unscaled,   # apenas as que deram certo
    fit_elapsed_time=float(np.mean(fit_times)) if fit_times else 0.0,
    hf_elapsed_time=float(np.mean(hf_times)) if hf_times else 0.0,
    results_path=results_path,
    mode="local_univariate",
    series_ids=success_uids,                        # <- usa trace_id
)

print(f"[OK] {len(success_uids)}/{len(all_uids)} séries salvas com {model_name}.")