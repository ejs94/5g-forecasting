import time
from typing import Any, Dict, List, Optional, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

# from darts.models import ForecastingModel
from darts.metrics import mae, rmse, mse
from tqdm.auto import tqdm


def sliding_window_cross_validate_and_evaluate(
    model,
    ts: TimeSeries,
    K: int,
    H: int,
    update_interval: int = 60,  # Atualização a cada 60 segundos
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Avalia um modelo de previsão utilizando uma janela deslizante em uma escala de segundos,
    com atualizações da janela a cada 60 segundos. Remove valores ausentes e reverte as predições
    para a escala original após o pós-processamento.

    :param model: Modelo de previsão a ser avaliado.
    :param ts: Série temporal completa para avaliação.
    :param K: Tamanho da janela de entrada (em segundos).
    :param H: Tamanho da janela de previsão (em segundos).
    :param update_interval: Intervalo de atualização da janela em segundos (default: 60).
    :param model_name: Nome do modelo (opcional). Se None, usa o nome da classe do modelo.

    :return: Dicionário contendo o nome do modelo, métricas de avaliação médias e tempo de execução.
    """

    if model_name is None:
        model_name = type(model).__name__

    # Aplica o pré-processamento com filler e scaler
    filler = MissingValuesFiller()  # Preenche valores ausentes
    scaler = Scaler()  # Escalona os dados
    pipe = Pipeline([filler, scaler])

    # Transformar a série temporal completa
    ts_transformed = pipe.fit_transform(ts)

    # Extrair os dados e o índice de tempo da série temporal transformada
    data = ts_transformed.values()
    times = ts_transformed.time_index

    # Armazenar resultados
    actuals = []
    preds = []

    # Número total de possíveis janelas deslizantes com atualização de 60 segundos
    total_windows = (len(data) - K - H) // update_interval + 1

    if total_windows <= 0:
        raise ValueError(
            "O tamanho da série temporal é insuficiente para realizar a validação cruzada com as janelas deslizantes fornecidas."
        )

    start_time = time.time()

    for i in tqdm(range(total_windows), desc="Processing Windows"):
        train_start = i * update_interval
        train_end = train_start + K
        test_start = train_end
        test_end = test_start + H

        if test_end > len(data):
            break  # Para se não houver dados suficientes para a janela de teste

        # Criar as janelas de treino e teste com intervalo de 60 segundos
        ts_train = TimeSeries.from_times_and_values(
            times[train_start:train_end], data[train_start:train_end]
        )
        ts_test = TimeSeries.from_times_and_values(
            times[test_start:test_end], data[test_start:test_end]
        )

        try:
            # Treinar o modelo com a série temporal transformada
            model.fit(ts_train)

            # Fazer previsões na escala transformada
            y_pred = model.predict(len(ts_test))

            # Se as predições não forem nulas e houver valores suficientes
            if y_pred is not None and len(y_pred) > 0:
                actuals.append(ts_test)
                preds.append(y_pred)

        except Exception as e:
            print(f"Erro ao processar a janela {i}: {e}")
            continue

    elapsed_time = time.time() - start_time

    # Verifica se há dados para concatenar
    if not actuals or not preds:
        raise ValueError(
            "Nenhuma previsão ou valor real foi gerado. Verifique o modelo e os dados."
        )

    try:
        # Concatena as séries temporais e converte para arrays
        actuals_concat = concatenate(actuals, ignore_time_axis=True)
        preds_concat = concatenate(preds, ignore_time_axis=True)

        # Reverter as predições para a escala original usando o inverso do escalonador
        actuals_series = scaler.inverse_transform(actuals_concat).all_values().flatten()
        preds_series = scaler.inverse_transform(preds_concat).all_values().flatten()

        time_indexs = actuals_concat.time_index.values

        results = {
            "Time_Index": time_indexs,
            "Model": model_name,
            "Actuals": actuals_series,
            "Preds": preds_series,
            "ElapsedTime": elapsed_time,
        }
    except Exception as e:
        print(f"Erro ao concatenar séries temporais ou converter para arrays: {e}")
        raise

    return results


def compare_series_metrics(results: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Calcula as métricas MAE, RMSE, MSE, NRMSE e NMSE para cada linha do DataFrame `results`
    comparando as séries temporais reais com as preditas, utilizando `n_jobs` para paralelização.

    Args:
        results (pd.DataFrame): DataFrame contendo as colunas "Actuals_index", "Actuals_values",
                                "Preds_index" e "Preds_values".
        n_jobs (int): Número de processos paralelos para o cálculo. Default é -1 (usa todos os processadores disponíveis).

    Returns:
        pd.DataFrame: DataFrame com as métricas calculadas adicionadas como novas colunas ("MAE", "RMSE", "MSE", "NRMSE", "NMSE").
    """
    if results is None:
        raise ValueError("O parâmetro 'results' não pode ser None.")
    if not isinstance(results, pd.DataFrame):
        raise TypeError("O parâmetro 'results' deve ser um DataFrame.")

    def validate_row(row):
        """
        Verifica se as colunas de índices e valores têm o mesmo comprimento e se não há valores NaN.

        Args:
            row (pd.Series): Linha do DataFrame a ser validada.

        Returns:
            bool: True se a linha for válida, False caso contrário.
        """
        pairs_to_check = [
            ("Train_index", "Train_values"),
            ("Actuals_index", "Actuals_values"),
            ("Preds_index", "Preds_values"),
        ]

        for index_col, value_col in pairs_to_check:
            if len(row[index_col]) != len(row[value_col]):
                return False
            if pd.isnull(row[index_col]).any() or pd.isnull(row[value_col]).any():
                return False

        return True

    def calculate_metrics(row):
        """Calcula todas as métricas para uma linha do DataFrame."""
        actual_idx = pd.DatetimeIndex(row["Actuals_index"])
        preds_idx = pd.DatetimeIndex(row["Preds_index"])

        # Criação das séries temporais
        actual_ts = TimeSeries.from_times_and_values(actual_idx, row["Actuals_values"])
        preds_ts = TimeSeries.from_times_and_values(preds_idx, row["Preds_values"])

        # Cálculo das métricas
        mae_val = mae(
            actual_series=actual_ts, pred_series=preds_ts, intersect=True, n_jobs=n_jobs
        )
        rmse_val = rmse(
            actual_series=actual_ts, pred_series=preds_ts, intersect=True, n_jobs=n_jobs
        )
        mse_val = mse(
            actual_series=actual_ts, pred_series=preds_ts, intersect=True, n_jobs=n_jobs
        )

        # Normalizações para NRMSE e NMSE
        actual_range = row["Actuals_values"].max() - row["Actuals_values"].min()
        actual_variance = row["Actuals_values"].var()
        nrmse_val = rmse_val / actual_range if actual_range != 0 else float("nan")
        nmse_val = mse_val / actual_variance if actual_variance != 0 else float("nan")

        return {
            "MAE": mae_val,
            "RMSE": rmse_val,
            "MSE": mse_val,
            "NRMSE": nrmse_val,
            "NMSE": nmse_val,
        }

    # Filtrar linhas inválidas
    valid_results = results[results.apply(validate_row, axis=1)].copy()

    # Calcula métricas para todas as linhas válidas
    metrics = valid_results.apply(calculate_metrics, axis=1, result_type="expand")

    # Combina métricas ao DataFrame original
    valid_results = pd.concat([valid_results.reset_index(drop=True), metrics], axis=1)

    return valid_results


def calculate_grouped_statistics(df):
    """
    Calcula os quartis, média, mediana, mínimo e máximo de MAE, RMSE e MSE agrupados por 'model' e 'target'.

    :param df: DataFrame contendo as colunas 'MAE', 'RMSE', 'MSE', 'model' e 'target'.
    :return: Um DataFrame com as estatísticas agrupadas por 'model' e 'target'.
    """
    grouped_stats = df.groupby(["Model", "target"]).agg(
        MAE_Min=("MAE", "min"),
        MAE_1Q=("MAE", lambda x: x.quantile(0.25)),
        MAE_Median=("MAE", "median"),
        MAE_3Q=("MAE", lambda x: x.quantile(0.75)),
        MAE_Max=("MAE", "max"),
        MAE_Mean=("MAE", "mean"),
        RMSE_Min=("RMSE", "min"),
        RMSE_1Q=("RMSE", lambda x: x.quantile(0.25)),
        RMSE_Median=("RMSE", "median"),
        RMSE_3Q=("RMSE", lambda x: x.quantile(0.75)),
        RMSE_Max=("RMSE", "max"),
        RMSE_Mean=("RMSE", "mean"),
        MSE_Min=("MSE", "min"),
        MSE_1Q=("MSE", lambda x: x.quantile(0.25)),
        MSE_Median=("MSE", "median"),
        MSE_3Q=("MSE", lambda x: x.quantile(0.75)),
        MSE_Max=("MSE", "max"),
        MSE_Mean=("MSE", "mean"),
    )
    return grouped_stats