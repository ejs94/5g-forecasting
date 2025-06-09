import os
import pickle
import warnings

from darts.dataprocessing.transformers import Scaler
from darts.models import (
    BlockRNNModel,
    LightGBMModel,
    LinearRegressionModel,
    NBEATSModel,
    NHiTSModel,
    RandomForest,
    TFTModel,
    TransformerModel,
)
from darts.utils.model_selection import train_test_split
from tqdm.auto import tqdm

from pipeline_5g.train_and_validate import (
    train_time_series_model,
    walk_forward_validation,
)
from pipeline_5g.utils import get_torch_device_config, save_historical_forecast_results

warnings.filterwarnings("ignore")

# ======================= MAIN =======================

print("---Verificando disponibilidade de GPU/CPU---")
torch_kwargs = get_torch_device_config()

print("---Carregando os dados 5G Dataset---")

base_data_path = os.path.join(
    os.curdir, "data"
)  # Assumindo que 'data' está um nível acima do diretório do script
processed_timeseries_path = os.path.join(base_data_path, "processed_timeseries")

results_path = os.path.join(
    base_data_path, "results"
)  # Caminho para salvar os resultados
os.makedirs(results_path, exist_ok=True)

models_trained_path = os.path.join(
    base_data_path, "results", "models_trained"
)  # Caminho para salvar os resultados
os.makedirs(models_trained_path, exist_ok=True)

try:
    with open(
        os.path.join(processed_timeseries_path, "processed_targets.pkl"), "rb"
    ) as f:
        all_targets_cleaned = pickle.load(f)

    with open(
        os.path.join(processed_timeseries_path, "processed_covariates.pkl"), "rb"
    ) as f:
        all_past_covariates_cleaned = pickle.load(f)
except FileNotFoundError:
    print(
        "ERRO: Arquivos de dados processados (processed_targets.pkl ou processed_covariates.pkl) não encontrados. Verifique os caminhos."
    )
    exit()

## MinMaxScaler
scaler_output = Scaler()
scaler_covariates = Scaler()

target_ts_scaled = scaler_output.fit_transform(all_targets_cleaned)
past_covariates_ts_scaled = scaler_covariates.fit_transform(all_past_covariates_cleaned)

# Split Train/Test
train_ts, test_ts = train_test_split(target_ts_scaled, test_size=0.2, axis=1)
train_past_covariates_ts, test_past_covariates_ts = train_test_split(
    past_covariates_ts_scaled, test_size=0.2, axis=1
)

# Parametros comuns
predict_horizon = 10
input_chunk_length = 10
output_chunk_length = 10

# Treiando modelos covariados

# TODO: Adicionar os naive global

# --- Global Naive Baselines ---
# Covariates: RSRP, RSRQ, SNR, RSSI, Speed

# --- Linear Regression ---
# Covariates: RSRP, RSRQ, SNR, RSSI, Speed

# model = LinearRegressionModel(
#     lags=10,
#     lags_past_covariates=10,
#     output_chunk_length=output_chunk_length
# )

# model = RandomForest(
#     lags=10,
#     lags_past_covariates=10,
#     output_chunk_length=output_chunk_length,
#     n_estimators=200,
#     max_depth=15,
#     n_jobs=-1,
#     verbose=1,
# )

# model = LightGBMModel(
#     lags=10,
#     lags_past_covariates=10,
#     output_chunk_length=output_chunk_length,
#     n_jobs=-1,
#     verbose=1,
# )

# --- Deep Learning Based Models ---
# Covariates: RSRP, RSRQ, SNR, RSSI, Speed

model = BlockRNNModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    model="LSTM",
    hidden_dim=64,
    n_rnn_layers=2,
    hidden_fc_sizes=[64,32],
    dropout=0.2,
    activation="ReLU",
    batch_size=64,         
    n_epochs=100,                 
    **get_torch_device_config(),
)

# Treinamento com covariáveis
model, fit_elapsed_time, model_name = train_time_series_model(
    model=model,
    train_series=train_ts,
    train_covariates=train_past_covariates_ts,
    base_data_path=base_data_path,
)

# Validação com covariáveis
historical_preds_scaled, hf_elapsed_time = walk_forward_validation(
    model=model,
    target_series=target_ts_scaled,
    covariate_series=train_past_covariates_ts,
    forecast_horizon=predict_horizon,
)

historical_preds_unscaled = scaler_output.inverse_transform(historical_preds_scaled)

save_historical_forecast_results(
    model_name=model_name,
    historical_preds_unscaled=historical_preds_unscaled,
    all_targets_cleaned=all_targets_cleaned,
    fit_elapsed_time=fit_elapsed_time,
    hf_elapsed_time=hf_elapsed_time,
    results_path=results_path,
    mode="global_covariate",
)


# # Filtrar séries para as quais não foi possível gerar previsões (se houver Nones)
# valid_actuals_for_metrics = []
# valid_preds_for_metrics = []
# for i in range(len(all_targets_cleaned)):
#     if i < len(historical_preds_unscaled) and historical_preds_unscaled[i] is not None:
#         # Verifica se a previsão não está vazia (pode acontecer se a fatia de teste for muito pequena)
#         if len(historical_preds_unscaled[i]) > 0:
#                 valid_actuals_for_metrics.append(all_targets_cleaned[i])
#                 valid_preds_for_metrics.append(historical_preds_unscaled[i])
#         else:
#             print(f"INFO: Previsão histórica para série {i} está vazia. Pulando para métricas.")
#     else:
#         print(f"INFO: Nenhuma previsão histórica gerada para série {i} ou é None. Pulando para métricas.")

# mae_overall = mae(valid_actuals_for_metrics, valid_preds_for_metrics)
# rmse_overall = rmse(valid_actuals_for_metrics, valid_preds_for_metrics)

# print(f"  Todos os MAE:  {mae_overall}")
# print(f"  Todos os RMSE: {rmse_overall}")

# print(f"  MAE Mean:  {np.mean(mae_overall)}")
# print(f"  RMSE Mean: {np.mean(rmse_overall)}")

# try:
#     mape_overall = mape(valid_actuals_for_metrics, valid_preds_for_metrics)
#     print(f"  MAPE Geral: {mape_overall}%")
# except Exception as e:
#     print("[Warning] Não foi possivel usar mape.\n", e)

# print("\n--- Gerando Gráficos Comparativos (Real vs. Previsto) com Métricas Individuais ---")

# # Define quantas séries você quer plotar
# num_series_to_plot = min(len(valid_actuals_for_metrics), 5) # Plota no máximo 5 séries

# if num_series_to_plot == 0:
#     print("Nenhuma série válida para plotar.")
# else:
#     for i in range(num_series_to_plot):
#         actual_series = valid_actuals_for_metrics[i]
#         predicted_series = valid_preds_for_metrics[i]

#         # Calcula MAE e RMSE para a série ATUAL
#         # As funções de métrica do Darts retornam um único valor se as entradas forem TimeSeries únicas
#         current_mae = mae(actual_series, predicted_series)
#         current_rmse = rmse(actual_series, predicted_series)
#         # Tenta calcular MAPE para a série atual, se possível
#         current_mape_str = ""
#         try:
#             current_mape = mape(actual_series, predicted_series)
#             current_mape_str = f", MAPE: {current_mape:.2f}%"
#         except Exception: # Captura ZeroDivisionError ou ValueError do Darts
#             current_mape_str = ", MAPE: N/A"

#         plt.figure(figsize=(14, 7)) # Aumentei um pouco a largura para a legenda

#         actual_series.plot(label='Real (Completa)', lw=1.5)

#         # Adiciona as métricas na legenda da série prevista
#         predicted_series.plot(
#             label=f'Previsto (MAE: {current_mae:.2f}, RMSE: {current_rmse:.2f}{current_mape_str})',
#             lw=1.5,
#             linestyle='--'
#         )

#         plt.title(f'Comparação Real vs. Previsto - Série {i+1}')
#         plt.xlabel('Tempo')
#         plt.ylabel(target_col_name)
#         plt.legend(loc='best') # 'best' tenta encontrar a melhor localização para a legenda
#         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#         plt.tight_layout()
#         plt.show()

#     if len(valid_actuals_for_metrics) > num_series_to_plot:
#         print(f"\n[INFO] Mostrando gráficos para as primeiras {num_series_to_plot} de {len(valid_actuals_for_metrics)} séries válidas.")
#         print("[INFO] Para plotar todas ou mais séries, ajuste 'num_series_to_plot' no código.")
