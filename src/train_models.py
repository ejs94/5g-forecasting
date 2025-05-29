import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mape, rmse
from darts.models import LinearRegressionModel, NBEATSModel
from darts.utils.model_selection import train_test_split

from pipeline_5g.utils.hardware import get_torch_device_config
from pipeline_5g.utils.timeseries import (
    create_covariates_timeseries,
    create_target_timeseries,
    impute_timeseries_missing_values,
)

# Detecta e configura o uso de GPU ou CPU
print("---Verificando disponibilidade de GPU/CPU---")
torch_kwargs = get_torch_device_config()

# Carregando os dados
print("---Carregando os dados 5G Dataset---")
data_directory = os.path.join(os.curdir, "data", "reduced_metrics_datasets")

file_dict = {
    "static_down_metrics": os.path.join(data_directory, "static_down_metrics.pkl"),
    "static_strm_metrics": os.path.join(data_directory, "static_strm_metrics.pkl"),
    "driving_down_metrics": os.path.join(data_directory, "driving_down_metrics.pkl"),
    "driving_strm_metrics": os.path.join(data_directory, "driving_strm_metrics.pkl"),
}

# Lê e concatena os DataFrames
print("---Lendo e concatenando os DataFrames---")
dfs = []
for name, path in file_dict.items():
    print(f"[INFO] Carregando arquivo: {name}")
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
            df["source"] = name  # opcional: adicionar coluna de origem
            dfs.append(df)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado {path}. Crie os dados de exemplo ou forneça o caminho correto.")
        # Para continuar, podemos adicionar um DataFrame vazio, mas isso pode causar outros problemas.
        # Por enquanto, vamos parar se os dados não puderem ser carregados.
        raise
    except Exception as e:
        print(f"ERRO ao carregar {path}: {e}")
        raise

if not dfs:
    raise ValueError("Nenhum DataFrame foi carregado. Verifique os caminhos dos arquivos e os dados.")


df_total = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Total de linhas após concatenação: {len(df_total)}")
print(f"[INFO] Colunas Selecionadas para Treinamento: {df_total.columns.values}")
print(
    "--- Convertendo linhas do DataFrame em TimeSeries e imputando valores ausentes ---"
)

target_col_name = "DL_bitrate"
covariate_cols_names = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"]

all_targets_raw = []
all_past_covariates_raw = []

for idx, row in df_total.iterrows():
    # Criação das séries temporais
    target_ts = create_target_timeseries(row, target_col_name, timestamp_col="Timestamp")
    past_cov_ts = create_covariates_timeseries(
        row, covariate_cols_names, timestamp_col="Timestamp"
    )

    # Imputação de valores ausentes
    target_ts = impute_timeseries_missing_values(target_ts)
    past_cov_ts = impute_timeseries_missing_values(past_cov_ts)

    all_targets_raw.append(target_ts)
    all_past_covariates_raw.append(past_cov_ts)


## MinMaxScaler
scaler_output = Scaler()
scaler_covariates = Scaler()

target_ts_scaled = scaler_output.fit_transform(all_targets_raw)
past_covariates_ts_scaled = scaler_covariates.fit_transform(
    all_past_covariates_raw
)

# Split Train/Test
train_ts, test_ts = train_test_split(target_ts_scaled, test_size=0.2, axis=1)
train_past_covariates_ts, test_past_covariates_ts = train_test_split(
    past_covariates_ts_scaled, test_size=0.2, axis=1
)

# Treiando modelos não covariados

# Treiando modelos covariados

# create a GFM model, train and predict

# Linear Regression

predict_horizon = 10
model = LinearRegressionModel(lags=10, lags_past_covariates=10)

# Início do treinamento INICIAL do modelo
print("\n--- Treinando modelo global com as porções iniciais de treino ---")
start_fit_time = time.time()
model.fit(train_ts, past_covariates=train_past_covariates_ts)
fit_elapsed_time = time.time() - start_fit_time
print(f"Modelo inicial treinado em {fit_elapsed_time:.2f}s")


# --- AVALIAÇÃO USANDO HISTORICAL FORECASTS ---
print("\n--- Realizando Historical Forecasts (Backtesting) ---")
start_hf_time = time.time()

historical_preds_scaled = model.historical_forecasts(
    series=target_ts_scaled,  # Lista de TODAS as séries alvo escaladas
    past_covariates=past_covariates_ts_scaled, # Lista de TODAS as covariáveis passadas escaladas
    start=0.8,              # Começa a prever após 80% de cada série
    forecast_horizon=predict_horizon,
    stride=1,               # Faz uma nova previsão a cada passo no período de teste
    retrain=False,          # USA O MODELO JÁ TREINADO GLOBALMENTE
    verbose=True,
    show_warnings=True
)

hf_elapsed_time = time.time() - start_hf_time
print(f"Historical forecasts gerados em {hf_elapsed_time:.2f}s")
historical_preds_unscaled = scaler_output.inverse_transform(historical_preds_scaled)


# Filtrar séries para as quais não foi possível gerar previsões (se houver Nones)
valid_actuals_for_metrics = []
valid_preds_for_metrics = []
for i in range(len(all_targets_raw)):
    if i < len(historical_preds_unscaled) and historical_preds_unscaled[i] is not None:
        # Verifica se a previsão não está vazia (pode acontecer se a fatia de teste for muito pequena)
        if len(historical_preds_unscaled[i]) > 0:
                valid_actuals_for_metrics.append(all_targets_raw[i])
                valid_preds_for_metrics.append(historical_preds_unscaled[i])
        else:
            print(f"INFO: Previsão histórica para série {i} está vazia. Pulando para métricas.")
    else:
        print(f"INFO: Nenhuma previsão histórica gerada para série {i} ou é None. Pulando para métricas.")


mae_overall = mae(valid_actuals_for_metrics, valid_preds_for_metrics)
rmse_overall = rmse(valid_actuals_for_metrics, valid_preds_for_metrics)

print(f"  Todos os MAE:  {mae_overall}")
print(f"  Todos os RMSE: {rmse_overall}")

print(f"  MAE Mean:  {np.mean(mae_overall)}")
print(f"  RMSE Mean: {np.mean(rmse_overall)}")


try:
    mape_overall = mape(valid_actuals_for_metrics, valid_preds_for_metrics)
    print(f"  MAPE Geral: {mape_overall}%")
except Exception as e:
    print("[Warning] Não foi possivel usar mape.\n", e)

# Testantando a primeira serie da lista
# y_pred = model.predict(
#     n=len(test_ts[0]),
#     series=train_ts[0],
#     past_covariates=past_covariates_ts_scaled[0],
# )

# elapsed_time = time.time() - start_time


# y_pred_unscaled = scaler_output.inverse_transform(y_pred)
# actual_unscaled = scaler_output.inverse_transform(test_ts[0])

# mae_score = mae(actual_unscaled, y_pred_unscaled)
# rmse_score = rmse(actual_unscaled, y_pred_unscaled)

# try:
#     mape_score = mape(actual_unscaled, y_pred_unscaled)
#     print("Previsão para a primeira série de teste (comparada com test_ts[0]):")
#     print(f"  MAE:  {mae_score:.4f}")
#     print(f"  RMSE: {rmse_score:.4f}")
#     print(f"  MAPE: {mape_score:.2f}%")
# except ZeroDivisionError:
# print("ERRO ao calcular MAPE: Divisão por zero encontrada nos valores reais.")
# print("Previsão para a primeira série de teste (comparada com test_ts[0]):")
# print(f"  MAE:  {mae_score:.4f}")
# print(f"  RMSE: {rmse_score:.4f}")
# print("  MAPE: Não pôde ser calculado devido a zeros nos valores reais.")

# NBEATS
# model = NBEATSModel(input_chunk_length=1, output_chunk_length=1)

# model.fit(train_ts, past_covariates=train_covariates_ts)

# pred = model.predict(
#     n=1, series=test_ts[:2], past_covariates=test_past_covariates_ts[:2]
# )

print("\n--- Gerando Gráficos Comparativos (Real vs. Previsto) com Métricas Individuais ---")

# Define quantas séries você quer plotar
num_series_to_plot = min(len(valid_actuals_for_metrics), 5) # Plota no máximo 5 séries

if num_series_to_plot == 0:
    print("Nenhuma série válida para plotar.")
else:
    for i in range(num_series_to_plot):
        actual_series = valid_actuals_for_metrics[i]
        predicted_series = valid_preds_for_metrics[i]
        
        # Calcula MAE e RMSE para a série ATUAL
        # As funções de métrica do Darts retornam um único valor se as entradas forem TimeSeries únicas
        current_mae = mae(actual_series, predicted_series)
        current_rmse = rmse(actual_series, predicted_series)
        # Tenta calcular MAPE para a série atual, se possível
        current_mape_str = ""
        try:
            current_mape = mape(actual_series, predicted_series)
            current_mape_str = f", MAPE: {current_mape:.2f}%"
        except Exception: # Captura ZeroDivisionError ou ValueError do Darts
            current_mape_str = ", MAPE: N/A"
            
        plt.figure(figsize=(14, 7)) # Aumentei um pouco a largura para a legenda
        
        actual_series.plot(label='Real (Completa)', lw=1.5)
        
        # Adiciona as métricas na legenda da série prevista
        predicted_series.plot(
            label=f'Previsto (MAE: {current_mae:.2f}, RMSE: {current_rmse:.2f}{current_mape_str})', 
            lw=1.5, 
            linestyle='--'
        )
        
        plt.title(f'Comparação Real vs. Previsto - Série {i+1}')
        plt.xlabel('Tempo')
        plt.ylabel(target_col_name)
        plt.legend(loc='best') # 'best' tenta encontrar a melhor localização para a legenda
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    if len(valid_actuals_for_metrics) > num_series_to_plot:
        print(f"\n[INFO] Mostrando gráficos para as primeiras {num_series_to_plot} de {len(valid_actuals_for_metrics)} séries válidas.")
        print("[INFO] Para plotar todas ou mais séries, ajuste 'num_series_to_plot' no código.")
