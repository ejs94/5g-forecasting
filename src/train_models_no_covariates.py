import os

import pandas as pd
from darts import TimeSeries

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

models_trained_path = os.path.join(base_data_path, "results", "models_trained")
os.makedirs(models_trained_path, exist_ok=True)

# Leitura do dataset longo
parquet_path = os.path.join(processed_timeseries_path, "processed_timeseries.parquet")
if not os.path.exists(parquet_path):
    print(f"ERRO: Arquivo não encontrado: {parquet_path}")
    exit(1)


df_long = pd.read_parquet(parquet_path)

# Garante dtypes e ordenação
df_long["Timestamp"] = pd.to_datetime(df_long["Timestamp"], utc=False)
df_long = df_long.sort_values(["Uid", "Timestamp"])

target_col = "DL_bitrate"
cov_cols = [c for c in ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"] if c in df_long.columns]

all_targets_cleaned = []
all_past_covariates_cleaned = []
all_uids = []

for uid, g in df_long.groupby("Uid", sort=True):
    # mantém apenas linhas com alvo disponível
    g_t = g.dropna(subset=[target_col]).copy()
    if g_t.empty:
        continue

    # cria séries Darts
    tgt_ts = TimeSeries.from_dataframe(g_t, time_col="Timestamp", value_cols=[target_col])

    if cov_cols:
        g_c = g.dropna(subset=cov_cols).copy()
        if g_c.empty:
            # sem covariadas válidas: pode pular ou criar uma série vazia
            continue
        cov_ts = TimeSeries.from_dataframe(g_c, time_col="Timestamp", value_cols=cov_cols)

        # alinha por interseção de índices (importante para Darts)
        tgt_ts = tgt_ts.slice_intersect(cov_ts)
        cov_ts = cov_ts.slice_intersect(tgt_ts)

        # filtra casos degenerados
        if len(tgt_ts) == 0 or len(cov_ts) == 0:
            continue

        all_targets_cleaned.append(tgt_ts)
        all_past_covariates_cleaned.append(cov_ts)
        all_uids.append(uid)
    else:
        # Sem covariadas no dataset; ainda adiciona targets
        if len(tgt_ts) > 0:
            all_targets_cleaned.append(tgt_ts)

if not all_targets_cleaned:
    print("ERRO: Não foi possível construir séries alvo a partir do parquet.")
    exit(1)

scaler_output = Scaler()

target_ts_scaled = scaler_output.fit_transform(all_targets_cleaned)

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

model = LinearRegressionModel(
    lags=10,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length=output_chunk_length,
)

model = RandomForest(
    lags=10,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length=output_chunk_length,
    # n_estimators=200,
    # max_depth=12,
    # min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)


model = LightGBMModel(
    lags=10,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length=output_chunk_length,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

# Deep Learning based Models

model = BlockRNNModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    # model="LSTM",
    # hidden_dim=64,
    # n_rnn_layers=2,
    # hidden_fc_sizes=[64,32],
    # dropout=0.2,
    # activation="ReLU",
    batch_size=32,
    n_epochs=100,
    random_state=42,
    **get_torch_device_config(),
)

model = NBEATSModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    # generic_architecture=True,
    # num_stacks=30,
    # num_blocks=1,
    # num_layers=4,
    # layer_widths=256,
    # expansion_coefficient_dim=5,
    # dropout=0.2,
    batch_size=32,
    n_epochs=100,
    random_state=42,
    **torch_kwargs,
)

model = TransformerModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    # dropout=0.2,
    # batch_size=128,
    batch_size=32,
    n_epochs=100,
    random_state=42,
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

historical_preds_unscaled = scaler_output.inverse_transform(historical_preds_scaled)

# Exporta resultados com modo univariate (sem covariáveis)
save_historical_forecast_results(
    model_name=model_name,
    historical_preds_unscaled=historical_preds_unscaled,
    all_targets_cleaned=all_targets_cleaned,
    fit_elapsed_time=fit_elapsed_time,
    hf_elapsed_time=hf_elapsed_time,
    results_path=results_path,
    mode="global_no_covariate",
    series_ids=all_uids,
)

# TODO: Adicionar os métodos naive
