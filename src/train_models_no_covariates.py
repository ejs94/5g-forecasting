import os
import pickle

from darts.dataprocessing.transformers import Scaler
from darts.models import (
    # Global Baselines
    GlobalNaiveAggregate,
    GlobalNaiveDrift,
    GlobalNaiveSeasonal,
    # Regression Models
    LinearRegressionModel,
    RandomForest,
    LightGBMModel,
    # PyTorch (Lightning)-based Models
    BlockRNNModel,
    NBEATSModel,
    TransformerModel,
)
from darts.utils.model_selection import train_test_split
from darts.utils.utils import ModelMode, SeasonalityMode

from pipeline_5g.train_and_validate import (
    train_time_series_model,
    walk_forward_validation,
)
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
input_chunk_length = 10
output_chunk_length = 10  # not defined for statistical models

# --- Exponential Smoothing ---

# Not Working: Train `series` must be a single `TimeSeries`.

# ValueError:
# model = NaiveMean()

#  ValueError: Train `series` must be a single `TimeSeries`.
# model = ExponentialSmoothing(
#     trend=ModelMode.NONE,
#     damped=False,
#     seasonal=SeasonalityMode.NONE,
#     seasonal_periods=None,
#     output_chunk_length=output_chunk_length,
# )


# ------------- Global Forecasting Models (GFMs) -------------
# Detail: Non Covariates

## Global Baseline Models
# - Working with issues: cant save model

# model = GlobalNaiveAggregate(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
#     agg_fn="mean",
# )

# model = GlobalNaiveDrift(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
# )

# model = GlobalNaiveSeasonal(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
# )

## Regression Models
# - Working Without problem

# model = LinearRegressionModel(
#     lags=10,
#     lags_past_covariates=None,
#     lags_future_covariates=None,
#     output_chunk_length=output_chunk_length,
# )

# model = RandomForest(
#     lags=10,
#     lags_past_covariates=None,
#     lags_future_covariates=None,
#     output_chunk_length=output_chunk_length,
#     n_estimators=500,
#     max_depth=None,
#     n_jobs=-1, # ativa o uso de todos os threads
# )

# model = RandomForest(
#     lags=10,
#     lags_past_covariates=None,
#     lags_future_covariates=None,
#     output_chunk_length=output_chunk_length,
#     n_estimators=500,
#     max_depth=None,
#     n_jobs=-1, # ativa o uso de todos os threads
# )

# model = LightGBMModel(
#     lags=10,
#     lags_past_covariates=None,
#     lags_future_covariates=None,
#     output_chunk_length=output_chunk_length,
#     n_jobs=-1, # ativa o uso de todos os threads
# )

# Deep Learning based Models

# model = BlockRNNModel(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
#     model="LSTM",
#     hidden_dim=64,
#     n_rnn_layers=3,
#     hidden_fc_sizes=[64, 32],
#     dropout=0.3,
#     n_epochs=100,
#     **torch_kwargs,
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

# Treinamento sem covariáveis
model, fit_elapsed_time, model_name = train_time_series_model(
    model=model,
    train_series=train_ts,
    train_covariates=None,  # Explicitamente None para não usar covariáveis
    base_data_path=base_data_path,
)

# Validação sem covariáveis (covariate_series = None)
historical_preds_scaled, hf_elapsed_time = walk_forward_validation(
    model=model,
    target_series=target_ts_scaled,
    covariate_series=None,
    forecast_horizon=predict_horizon,
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
    mode="global_no_covariate",
)

# TODO: Adicionar os métodos naive
