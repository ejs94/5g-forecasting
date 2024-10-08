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
from darts.metrics import mae, mse, rmse
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
        # "MASE": mase(actual_series=ts_test, pred_series=y_pred, insample=ts_train),
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
    # mases = [
    #     mase(actual_series=test, pred_series=pred, insample=insample)
    #     for test, pred, insample in zip(test_series, pred_series, insample_series)
    # ]

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
    # axes[2].hist(mases, bins=50)
    # axes[2].set_title(f"Median MASE: {np.median(mases):.3f}")
    # axes[2].set_xlabel("MASE")
    # axes[2].set_ylabel("Count")

    # Ajustar layout e exibir
    plt.tight_layout()
    plt.show()
    plt.close()

    return {
        "MAE": np.mean(maes),
        "MSE": np.mean(mses),
        # "MASE": np.mean(mases)
    }


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


def collect_univariate_metrics(list_series, target_columns, model_name, model, output_file, K=5, H=1):
    """
    Coleta e salva métricas univariadas para uma lista de séries temporais utilizando
    um modelo de previsão. Os resultados são salvos em formato Parquet.

    Args:
        list_series (list): Lista de séries temporais, cada uma contendo colunas de KPIs.
        target_columns (list): Lista de colunas (KPIs) que serão avaliadas nas séries.
        model_name (str): Nome do modelo de previsão.
        model: Modelo de previsão a ser utilizado.
        output_file (str): Nome do arquivo de saída (formato .parquet).
        K (int): Número de subconjuntos para validação cruzada. Padrão: 5.
        H (int): Horizonte de previsão. Padrão: 1.

    Returns:
        pd.DataFrame: DataFrame contendo os resultados das métricas para cada série e KPI.
    """

    # Define o caminho para salvar o arquivo, adicionando a extensão .parquet
    output_path = os.path.join(os.curdir, "data", "results", f"{output_file}.parquet")
    
    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_records = []

    # Itera sobre cada série temporal
    for i, series in enumerate(list_series):
        for kpi in target_columns:
            try:
                # Avalia e coleta os resultados para o KPI
                results = sliding_window_cross_validate_and_evaluate(
                    model, series[kpi], K, H, 60, model_name
                )
                results["target"] = kpi
                result_records.append(results)
            except Exception as e:
                print(f"Erro ao processar a série {i} com {kpi}: {e}")
                continue
    
    result_record = pd.DataFrame(result_records)
    
    # Salva o DataFrame em formato Parquet
    result_record.to_parquet(output_path, compression="gzip")

    print(f"Saved in {output_path}")
    return result_record


def compare_series_metrics(
    results: pd.DataFrame, default_freq: str = "S"
) -> pd.DataFrame:
    """
    Calcula as métricas MAE, RMSE e MSE para cada linha do DataFrame `results`
    comparando as séries temporais reais com as preditas. Se a frequência do 'Time_Index'
    estiver ausente, usa um valor padrão especificado.

    Args:
        results (pd.DataFrame): DataFrame contendo as colunas "Time_Index", "Actuals" e "Preds",
        que representam as séries temporais reais e preditas.
        default_freq (str): Frequência padrão em segundos a ser utilizada quando a frequência do 'Time_Index'
                            não puder ser inferida (padrão: "S" - segundos).

    Returns:
        pd.DataFrame: DataFrame com as métricas calculadas adicionadas como novas
        colunas ("MAE", "RMSE", "MSE").
    """
    # Verifica se 'results' é None ou não é um DataFrame
    if results is None:
        raise ValueError("O parâmetro 'results' não pode ser None.")
    if not isinstance(results, pd.DataFrame):
        raise TypeError("O parâmetro 'results' deve ser um DataFrame.")


    def validate_row(row):
        """
        Verifica se as colunas 'Time_Index', 'Actuals' e 'Preds' têm o mesmo comprimento
        e se não há valores NaN nos dados.
        """
        # Verifica se os comprimentos são iguais
        if len(row["Time_Index"]) != len(row["Actuals"]) or len(
            row["Time_Index"]
        ) != len(row["Preds"]):
            return False

        # Verifica se há NaN nos índices de tempo ou valores reais/preditos
        if (
            pd.isnull(row["Time_Index"]).any()
            or pd.isnull(row["Actuals"]).any()
            or pd.isnull(row["Preds"]).any()
        ):
            return False

        return True

    def get_frequency_or_default(time_idx, default_freq):
        """
        Tenta inferir a frequência de 'Time_Index'. Se não conseguir, retorna a frequência padrão.
        """
        if len(time_idx) < 3:
            # Se houver menos de 3 datas, usa a frequência padrão
            return default_freq
        inferred_freq = pd.infer_freq(time_idx)
        return inferred_freq if inferred_freq else default_freq

    def calculate_mae(row):
        time_idx = pd.DatetimeIndex(row["Time_Index"])
        freq = get_frequency_or_default(time_idx, default_freq)
        actual_ts = TimeSeries.from_times_and_values(
            time_idx, row["Actuals"], freq=freq
        )
        pred_ts = TimeSeries.from_times_and_values(time_idx, row["Preds"], freq=freq)
        return mae(actual_series=actual_ts, pred_series=pred_ts)

    def calculate_rmse(row):
        time_idx = pd.DatetimeIndex(row["Time_Index"])
        freq = get_frequency_or_default(time_idx, default_freq)
        actual_ts = TimeSeries.from_times_and_values(
            time_idx, row["Actuals"], freq=freq
        )
        pred_ts = TimeSeries.from_times_and_values(time_idx, row["Preds"], freq=freq)
        return rmse(actual_series=actual_ts, pred_series=pred_ts)

    def calculate_mse(row):
        time_idx = pd.DatetimeIndex(row["Time_Index"])
        freq = get_frequency_or_default(time_idx, default_freq)
        actual_ts = TimeSeries.from_times_and_values(
            time_idx, row["Actuals"], freq=freq
        )
        pred_ts = TimeSeries.from_times_and_values(time_idx, row["Preds"], freq=freq)
        return mse(actual_series=actual_ts, pred_series=pred_ts)

    # Filtrar linhas inválidas (com valores NaN ou desajustes de tamanho)
    valid_results = results[results.apply(validate_row, axis=1)].copy()

    # Aplicando as funções de métrica a cada linha válida do DataFrame
    valid_results["MAE"] = valid_results.apply(calculate_mae, axis=1)
    valid_results["RMSE"] = valid_results.apply(calculate_rmse, axis=1)
    valid_results["MSE"] = valid_results.apply(calculate_mse, axis=1)

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
