import os
import time
from typing import List, Optional, Tuple

import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel


def train_time_series_model(
    model: ForecastingModel,
    train_series: List[TimeSeries],
    base_data_path: str,
    train_covariates: Optional[List[TimeSeries]] = None,
) -> Tuple[ForecastingModel, float, str]:
    """
    Treina um modelo de séries temporais com ou sem covariáveis passadas.

    Parâmetros:
    -----------
    model : ForecastingModel
        Modelo da biblioteca Darts que será treinado.
    train_series : List[TimeSeries]
        Lista de séries alvo para treino.
    base_data_path : str
        Caminho base onde os modelos treinados serão salvos.
    train_covariates : Optional[List[TimeSeries]], default=None
        Lista de covariáveis passadas para treino. Pode ser None para modelos não covariados.

    Retorna:
    --------
    model : ForecastingModel
        Modelo treinado.
    fit_elapsed_time : float
        Tempo de treino em segundos.
    model_name : str
        Nome da classe do modelo.
    """
    models_trained_path = os.path.join(base_data_path, "results", "models_trained")
    os.makedirs(models_trained_path, exist_ok=True)

    model_name = model.__class__.__name__
    model_output = os.path.join(models_trained_path, f"{model_name}_model.pth")

    print(f"\n--- Treinando modelo {model_name} ---")
    start_fit_time = time.time()

    if train_covariates is not None:
        model.fit(train_series, past_covariates=train_covariates)
    else:
        model.fit(train_series)

    fit_elapsed_time = time.time() - start_fit_time
    print(f"Modelo {model_name} treinado em {fit_elapsed_time:.2f}s")

    try:
        model.save(model_output)
        print(f"Modelo salvo em: {model_output}")
    except AttributeError as e:
        print(f"[AVISO] Falha ao salvar o modelo {model_name}: {e}")
    except Exception as e:
        print(f"[ERRO] Erro inesperado ao salvar o modelo {model_name}: {e}")

    return model, fit_elapsed_time, model_name

def walk_forward_validation(
    model: ForecastingModel,
    target_series: List[TimeSeries],
    covariate_series: Optional[List[TimeSeries]],
    forecast_horizon: int,
) -> Tuple[List[TimeSeries], float]:
    """
    Realiza a validação walk-forward com forecast histórico (backtesting).

    Parâmetros:
    -----------
    model : ForecastingModel
        Modelo treinado.
    target_series : List[TimeSeries]
        Séries alvo para previsão.
    covariate_series : Optional[List[TimeSeries]]
        Covariáveis passadas. Pode ser None para modelos não covariados.
    forecast_horizon : int
        Horizonte de previsão.

    Retorna:
    --------
    historical_preds_scaled : List[TimeSeries]
        Previsões escaladas geradas pelo modelo.
    hf_elapsed_time : float
        Tempo de execução do forecast histórico.
    """

    print(f"\n--- Realizando Historical Forecasts para {model.__class__.__name__} ---")
    start_hf_time = time.time()

    historical_preds_scaled = []

#     if covariate_series is not None:
#         historical_preds_scaled = model.historical_forecasts(
#             series=target_series,
#             past_covariates=covariate_series,
#             start=0.8,
#             forecast_horizon=forecast_horizon,
#             stride=1,
#             retrain=False,
#             verbose=True,
#             show_warnings=True,
#         )
    if covariate_series is not None:
        for target, cov in zip(target_series, covariate_series):
            preds = model.historical_forecasts(
                series=target,
                past_covariates=cov,
                start=0.8,
                forecast_horizon=forecast_horizon,
                stride=1,
                retrain=False,
                verbose=True,
                show_warnings=True,
            )
            historical_preds_scaled.append(preds)
    else:
        for target in target_series:
            preds = model.historical_forecasts(
                series=target,
                start=0.8,
                forecast_horizon=forecast_horizon,
                stride=1,
                retrain=False,
                verbose=True,
                show_warnings=True,
            )
            historical_preds_scaled.append(preds)

    hf_elapsed_time = time.time() - start_hf_time
    print(f"Forecasts concluídos em {hf_elapsed_time:.2f}s")

    if not historical_preds_scaled:
        raise ValueError(
            f"Nenhuma previsão histórica foi gerada por {model.__class__.__name__}."
        )

    return historical_preds_scaled, hf_elapsed_time
