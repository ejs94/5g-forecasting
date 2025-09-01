import os
import pickle
import warnings

from darts.dataprocessing.transformers import Scaler
from darts.models import (
    # Regression Models
    LinearRegressionModel,
    RandomForest,
    LightGBMModel,
    # PyTorch (Lightning)-based Models
    BlockRNNModel,
    NBEATSModel,
    # Not used in non covariates
    NHiTSModel,
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

base_data_path = os.path.join(os.curdir, "data")
processed_timeseries_path = os.path.join(base_data_path, "processed_timeseries")

results_path = os.path.join(base_data_path, "results")
os.makedirs(results_path, exist_ok=True)

models_trained_path = os.path.join(base_data_path, "results", "models_trained")
os.makedirs(models_trained_path, exist_ok=True)

try:
    # Load target
    with open(
        os.path.join(processed_timeseries_path, "processed_targets.pkl"), "rb"
    ) as f:
        all_targets_cleaned = pickle.load(f)

    # Load covariates
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
# train_ts, test_ts = train_test_split(target_ts_scaled, test_size=0.2, axis=1)

# train_past_covariates_ts, test_past_covariates_ts = train_test_split(
#     past_covariates_ts_scaled, test_size=0.2, axis=1
# )


train_ts, _ = train_test_split(target_ts_scaled, test_size=0.2, axis=1)

train_past_covariates_ts, _ = train_test_split(
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
#     lags_future_covariates=None,
#     output_chunk_length=output_chunk_length,
#     n_jobs=-1,
#     verbose=1,
# )

# --- Deep Learning Based Models ---
# Covariates: RSRP, RSRQ, SNR, RSSI, Speed

# model = BlockRNNModel(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
#     model="LSTM",
#     hidden_dim=64,
#     n_rnn_layers=2,
#     hidden_fc_sizes=[64,32],
#     dropout=0.2,
#     activation="ReLU",
#     batch_size=64,
#     n_epochs=100,
#     **get_torch_device_config(),
# )

# model = NBEATSModel(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
#     generic_architecture=True,
#     num_stacks=30,
#     num_blocks=1,
#     num_layers=4,
#     layer_widths=256,
#     expansion_coefficient_dim=5,
#     dropout=0.2,
#     batch_size=128,
#     n_epochs=100,
#     **torch_kwargs,
# )

model = TransformerModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    dropout=0.2,
    # batch_size=128,
    n_epochs=100,
    **torch_kwargs,
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
    covariate_series=past_covariates_ts_scaled,
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