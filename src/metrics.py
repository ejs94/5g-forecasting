import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate

# from darts.models import ForecastingModel
from darts.metrics import mae, mse, rmse
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm


def eval_model(
    model,
    ts_train: TimeSeries,
    ts_test: TimeSeries,
    model_name: Optional[str] = None,
) -> Tuple[TimeSeries, Dict[str, Any]]:
    """
    Evaluates a forecasting model with specific metrics.

    :param model: Forecasting model to be evaluated.
    :param ts_train: Training time series.
    :param ts_test: Testing time series.
    :param name: Name of the model (optional). If None, uses the class name of the model.

    :return: Model predictions and a dictionary with the algorithm name and evaluation metrics (MAE, MSE, MASE).
    """

    if model_name is None:
        model_name = type(model).__name__

    model.fit(ts_train)

    y_pred = model.predict(len(ts_test))

    return y_pred, {
        "Algorithm": model_name,
        "MAE": mae(actual_series=ts_test, pred_series=y_pred),
        "MSE": mse(actual_series=ts_test, pred_series=y_pred),
        "MASE": mase(actual_series=ts_test, pred_series=y_pred, insample=ts_train),
    }


def eval_forecasts(
    pred_series: List[TimeSeries],
    test_series: List[TimeSeries],
    insample_series: List[TimeSeries],
) -> Dict[str, List[float]]:
    """
    Avalia previsões de séries temporais calculando MAE, MSE e MASE e exibe subgráficos com os histogramas dessas métricas.

    :param pred_series: Lista de séries temporais previstas.
    :param test_series: Lista de séries temporais de teste.
    :param insample_series: Lista de séries temporais de treinamento usadas para calcular o MASE.

    :return: Dicionário com listas de valores para MAE, MSE e MASE para cada série temporal.
    """

    print("Computing metrics...")

    # Calcular métricas
    maes = [
        mae(actual_series=test, pred_series=pred)
        for test, pred in zip(test_series, pred_series)
    ]
    mses = [
        mse(actual_series=test, pred_series=pred)
        for test, pred in zip(test_series, pred_series)
    ]
    mases = [
        mase(actual_series=test, pred_series=pred, insample=insample)
        for test, pred, insample in zip(test_series, pred_series, insample_series)
    ]

    # Criar subgráficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histograma de MAE
    axes[0].hist(maes, bins=50)
    axes[0].set_title(f"Median MAE: {np.median(maes):.3f}")
    axes[0].set_xlabel("MAE")
    axes[0].set_ylabel("Count")

    # Histograma de MSE
    axes[1].hist(mses, bins=50)
    axes[1].set_title(f"Median MSE: {np.median(mses):.3f}")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Count")

    # Histograma de MASE
    axes[2].hist(mases, bins=50)
    axes[2].set_title(f"Median MASE: {np.median(mases):.3f}")
    axes[2].set_xlabel("MASE")
    axes[2].set_ylabel("Count")

    # Ajustar layout e exibir
    plt.tight_layout()
    plt.show()
    plt.close()

    return {"MAE": np.mean(maes), "MSE": np.mean(mses), "MASE": np.mean(mases)}


def sliding_window_cross_validate_and_evaluate(
    model,
    ts: TimeSeries,
    K: int,
    H: int,
    model_name: Optional[str] = None,
) -> Tuple[TimeSeries, Dict[str, Any]]:
    """
    Avalia um modelo de previsão utilizando uma janela deslizante.

    :param model: Modelo de previsão a ser avaliado.
    :param ts: Série temporal completa para avaliação.
    :param K: Tamanho da janela de entrada.
    :param H: Tamanho da janela de previsão.
    :param model_name: Nome do modelo (opcional). Se None, usa o nome da classe do modelo.

    :return: Tupla contendo as previsões da última divisão e um dicionário com o nome do modelo e as métricas de avaliação médias.
    """

    if model_name is None:
        model_name = type(model).__name__

    # Extrair os dados e o índice de tempo da série temporal
    data = ts.values()
    times = ts.time_index

    # Armazenar resultados
    actuals = []
    preds = []

    # Número total de possíveis janelas
    total_windows = len(data) - K - H + 1

    start_time = time.time()
    for i in tqdm(range(total_windows), desc="Processing Windows"):
        train_start = i
        train_end = train_start + K
        test_start = train_end
        test_end = test_start + H

        # Criar as janelas de treino e teste
        ts_train = TimeSeries.from_times_and_values(
            times[train_start:train_end], data[train_start:train_end]
        )
        ts_test = TimeSeries.from_times_and_values(
            times[test_start:test_end], data[test_start:test_end]
        )

        try:
            # Treinar o modelo
            model.fit(ts_train)

            # Fazer previsões
            y_pred = model.predict(len(ts_test))

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
        # Converte séries temporais para pd.Series antes de retornar
        actuals_series = concatenate(actuals).pd_series()
        preds_series = concatenate(preds).pd_series()

        results = {
            "Model": model_name,
            "Actuals": actuals_series,
            "Preds": preds_series,
            "ElapsedTime": elapsed_time,
        }
    except Exception as e:
        print(f"Erro ao concatenar séries temporais ou converter para pd.Series: {e}")
        raise

    return results


def compare_series_metrics(results):
    # Função para calcular a métrica MAE para cada linha
    def calculate_mae(row):
        actual_ts = TimeSeries.from_series(row["Actuals"])
        pred_ts = TimeSeries.from_series(row["Preds"])

        return mae(actual_series=actual_ts, pred_series=pred_ts)

    def calculate_rmse(row):
        actual_ts = TimeSeries.from_series(row["Actuals"])
        pred_ts = TimeSeries.from_series(row["Preds"])

        return rmse(actual_series=actual_ts, pred_series=pred_ts)

    def calculate_mse(row):
        actual_ts = TimeSeries.from_series(row["Actuals"])
        pred_ts = TimeSeries.from_series(row["Preds"])

        return mse(actual_series=actual_ts, pred_series=pred_ts)

    # Aplicando a função calculate_mae a cada linha do DataFrame
    results["MAE"] = results.apply(calculate_mae, axis=1)
    results["RMSE"] = results.apply(calculate_rmse, axis=1)
    results["MSE"] = results.apply(calculate_mse, axis=1)
    return results


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
