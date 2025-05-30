import argparse
import gc
import json
import multiprocessing
import os
import pickle
import time
import warnings

import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    ARIMA,
    FFT,
    ExponentialSmoothing,
    LightGBMModel,
    LinearRegressionModel,
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    NBEATSModel,
    Prophet,
    RNNModel,
    Theta,
)
from darts.utils.missing_values import (
    extract_subseries,
    fill_missing_values,
    missing_values_ratio,
)
from darts.utils.model_selection import train_test_split
from darts.utils.utils import SeasonalityMode
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

print("---Verificando se há GPU---")
# Verifica se a GPU está disponível
if torch.cuda.is_available():
    # Número total de threads disponíveis
    # Configura o número de threads
    num_threads = multiprocessing.cpu_count()
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print(f"Configurando PyTorch para usar {num_threads} threads.")

    print("A GPU está disponível.")

    def generate_torch_kwargs():
        # run torch models on CPU, and disable progress bars for all model stages except training.
        return {"pl_trainer_kwargs": {"accelerator": "gpu", "devices": [0]}}
else:
    print("A GPU NÃO está disponível. Rodando na CPU.")

    def generate_torch_kwargs():
        # run torch models on CPU, and disable progress bars for all model stages except training.
        return {
            "pl_trainer_kwargs": {
                "accelerator": "cpu",
            }
        }


print("---Verificando a Configuração---")
config_path = os.path.join(os.curdir, "config.json")
# Verifica se o arquivo de configuração já existe; caso contrário, cria um
if not os.path.exists(config_path):
    config = {
        "H": 5,
        "K": 25,
        "test_ratio": 0.1,
        "update_interval": 10,
        "target_columns": ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuração inicial salva em: {config_path}")
else:
    print(f"Lendo configuração existente de: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

print("---Configuração Utilizada---")
print(config)

print("---Carregando os dados preprocessados---")
reduced_metrics_path = os.path.join(os.curdir, "data", "reduced_metrics_datasets")
activities = ["static_down", "static_strm", "driving_down", "driving_strm"]

# Carregar os dados preprocessados para cada atividade
time_series_dict = {}  # Dicionário para armazenar as séries temporais por atividade

for activity in activities:
    pickle_file_path = os.path.join(reduced_metrics_path, f"{activity}_metrics.pkl")
    if os.path.exists(pickle_file_path):
        print(f"Carregando os dados pré-processados para {activity}...")
        with open(pickle_file_path, "rb") as f:
            time_series_list = pickle.load(f)
            time_series_dict[activity] = time_series_list
        print(f"Dados de {activity} carregados com sucesso.")
    else:
        print(f"Arquivo Pickle para {activity} não encontrado.")


print("---Configurando os modelos Baselines---")
baseline_models = {
    "Naive": NaiveSeasonal(K=1),
    "NaiveDrift": NaiveDrift(),
    "NaiveMovingAverage": NaiveMovingAverage(input_chunk_length=config["K"]),
    "NaiveMean": NaiveMean(),
    "ExponentialSmoothing": ExponentialSmoothing(seasonal=None),
    "LinearRegression": LinearRegressionModel(lags=config["K"]),
    "ARIMA": ARIMA(
        p=1,  # Ordem do autoregressivo (AR) - número de lags
        d=1,  # Ordem de diferenciação (I) - número de diferenciações
        q=1,  # Ordem do modelo de média móvel (MA) - tamanho da janela
        seasonal_order=(
            0,
            0,
            0,
            0,
        ),  # Não sazonal, sem componentes sazonais (P, D, Q, s)
        trend="n",  # Sem tendência determinística (sem tendência)
        random_state=42,  # Para garantir a reprodutibilidade (opcional)
        add_encoders=None,  # Não adicionar codificadores (opcional)
    ),
    "Theta": Theta(theta=1.0, season_mode=SeasonalityMode.NONE),
    "FFT": FFT(),
    "Prophet": Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_changepoints=50,
        changepoint_range=0.9,
        changepoint_prior_scale=0.1,
        interval_width=0.95,
    ),
}

print("---Configurando os modelos Machine Learning---")
dl_models = {
    "LSTM": RNNModel(
        model="LSTM",
        input_chunk_length=config["K"],
        output_chunk_length=config["H"],
        training_length=config["K"],
        hidden_dim=50,
        n_rnn_layers=4,
        dropout=0.0,
        batch_size=64,
        n_epochs=100,
        **generate_torch_kwargs(),
    ),
    "LightGBM": LightGBMModel(
        lags=config["K"],
        output_chunk_length=config["H"],
        **generate_torch_kwargs(),
    ),
    "NBEATS": NBEATSModel(
        input_chunk_length=config["K"],
        output_chunk_length=config["H"],
        generic_architecture=True,
        num_stacks=10,
        num_blocks=1,
        num_layers=4,
        layer_widths=512,
        batch_size=64,
        n_epochs=100,
        **generate_torch_kwargs(),
    ),
}


# Algumas funções


# Função para processar séries temporais
def train_process_timeseries(row, column_name, horizon=10):
    """
    Processa uma série temporal individual de acordo com critérios de valores ausentes
    e divide a série em sub-séries contínuas apenas se os valores consecutivos de NaN
    forem maiores que um horizonte especificado.

    Args:
        row (pd.Series): Linha do DataFrame contendo os dados.
        column_name (str): Nome da coluna da métrica.
        horizon (int): Número mínimo de NaN consecutivos para definir uma lacuna significativa.

    Returns:
        list[TimeSeries] | None: Lista de sub-séries ou None, se a série for descartada.
    """
    # Criar TimeSeries
    datetime_index = pd.to_datetime(row["Timestamp"])
    ts = TimeSeries.from_times_and_values(datetime_index, row[column_name], freq="s")

    # Calcular o ratio de valores ausentes
    ratio_missing = missing_values_ratio(ts)
    tqdm.write(f"[INFO] Missing ratio for {column_name}: {ratio_missing * 100:.2f}%")

    if ratio_missing > 0.4:  # Se mais de 40% dos valores estão ausentes, descartar
        tqdm.write(
            f"[INFO] Descartando a série {column_name} para o Uid {row['Uid']} (NaN > 40%)"
        )
        return None

    # Preencher valores ausentes com backward fill
    ts = fill_missing_values(ts)

    # Dividir em sub-séries apenas se lacunas consecutivas forem maiores que o horizonte
    subseries = extract_subseries(ts, min_gap_size=horizon, mode="any")

    # # Filtrar séries muito curtas
    # subseries_validas = [s for s in subseries if len(s) > 100]

    if not subseries:
        tqdm.write(
            f"[WARNING] Nenhuma sub-série válida encontrada para {column_name} (Uid {row['Uid']})"
        )
        return None

    return subseries


def save_results(
    result_records,
    no_window_results_path,
    model_name,
    activity,
    column_name,
    processed_count,
    discarded_count,
):
    """
    Save results to a Parquet file and log the number of processed and discarded series.

    Args:
        result_records (list[pd.DataFrame]): List of result records to save.
        no_window_results_path (str): Path to save the results.
        model_name (str): Name of the model.
        activity (str): Name of the activity.
        column_name (str): Name of the target column.
        processed_count (int): Number of series processed.
        discarded_count (int): Number of series discarded.
    """
    if result_records:
        result_record = pd.concat(result_records)
        output_file = os.path.join(
            no_window_results_path, f"{model_name}_{activity}_{column_name}.parquet"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_record.to_parquet(output_file, compression="gzip")
        tqdm.write(f"[SUCCESS] Results for {activity} saved at: {output_file}")
    else:
        tqdm.write(
            f"[WARNING] No results for activity '{activity}' with model '{model_name}'"
        )

    # Log the number of processed and discarded series
    tqdm.write(
        f"[INFO] Activity: {activity}, Model: {model_name}, Column: {column_name}\n"
        f"       Processed series: {processed_count}, Discarded series: {discarded_count}"
    )


def train_and_evaluate_models(models, time_series_dict, config, output_path):
    test_ratio = config["test_ratio"]
    scaler = Scaler()
    pipe = Pipeline([scaler])

    for model_name, model in tqdm(models.items(), desc="Training Models"):
        for activity, pd_series in tqdm(
            time_series_dict.items(), desc=f"Activity ({model_name})", leave=False
        ):
            for column_name in config["target_columns"]:
                evaluation_results = []
                processed_count = 0  # Contador de séries processadas
                discarded_count = 0  # Contador de séries descartadas

                processed_series = []
                for idx, row in pd_series.iterrows():
                    subseries = train_process_timeseries(
                        row, column_name, horizon=config["H"]
                    )
                    if subseries:
                        processed_series.extend(subseries)
                        processed_count += 1
                    else:
                        discarded_count += 1

                if not processed_series:
                    tqdm.write(
                        f"[WARNING] No valid series for {activity}, {column_name}, {model_name}"
                    )
                    continue

                for series in tqdm(
                    processed_series,
                    desc=f"Training Series {idx}: {column_name} ({model_name})",
                    leave=False,
                ):
                    start_time = time.time()

                    train_data, test_horizon = train_test_split(
                        series, test_size=test_ratio, axis=1
                    )

                    # Correção para modelos que precisam de um min de pontos em seu treino.
                    if len(train_data) <= 30:
                        continue

                    ts_transformed = pipe.fit_transform(train_data)

                    model.fit(ts_transformed)
                    y_pred = model.predict(len(test_horizon))

                    elapsed_time = time.time() - start_time

                    model_dir = os.path.join(
                        output_path, "models", activity, column_name
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    model.save(os.path.join(model_dir, f"lst_{model_name}_model.pkl"))

                    evaluation_results.append(
                        pd.DataFrame(
                            {
                                "target": column_name,
                                "Activity": activity,
                                "Model": model_name,
                                "Elapsed_time": [elapsed_time],
                                "Train_index": [
                                    pipe.inverse_transform(ts_transformed, partial=True)
                                    .pd_dataframe()
                                    .index.to_numpy()
                                ],
                                "Train_values": [
                                    pipe.inverse_transform(ts_transformed, partial=True)
                                    .pd_dataframe()
                                    .values.flatten()
                                ],
                                "Actuals_index": [
                                    test_horizon.pd_dataframe().index.to_numpy()
                                ],
                                "Actuals_values": [
                                    test_horizon.pd_dataframe().values.flatten()
                                ],
                                "Preds_index": [
                                    pipe.inverse_transform(y_pred, partial=True)
                                    .pd_dataframe()
                                    .index.to_numpy()
                                ],
                                "Preds_values": [
                                    pipe.inverse_transform(y_pred, partial=True)
                                    .pd_dataframe()
                                    .values.flatten()
                                ],
                            }
                        )
                    )

                save_results(
                    evaluation_results,
                    output_path,
                    model_name,
                    activity,
                    column_name,
                    processed_count,
                    discarded_count,
                )

                del evaluation_results
                gc.collect()

    tqdm.write("---Training complete---")


if __name__ == "__main__":
    # Configuração do argparse
    parser = argparse.ArgumentParser(
        description="Treinamento de modelos para séries temporais."
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Nome da coluna a ser treinada. Se não for especificado, todas as colunas serão usadas.",
    )

    # Argumento para selecionar os tipos de modelos a serem treinados
    parser.add_argument(
        "--models",
        type=str,
        choices=["baseline", "deep_learning"],
        nargs="+",  # Permite múltiplas opções
        required=True,
        help="Tipos de modelos a serem treinados. Pode ser 'baseline', 'deep_learning', ou ambos.",
    )

    args = parser.parse_args()

    # Atualiza a configuração com a coluna específica, se fornecida
    if args.column:
        if args.column in config["target_columns"]:
            config["target_columns"] = [args.column]
        else:
            print(
                f"[ERRO] A coluna '{args.column}' não está na lista de colunas disponíveis: {config['target_columns']}"
            )
            exit(1)

    print("---Configuração Atualizada---")
    print(config)

    no_window_results_path = os.path.join(os.curdir, "data", "results", "no_window")
    os.makedirs(no_window_results_path, exist_ok=True)

    # Seleciona e treina os modelos com base na escolha do usuário
    if "baseline" in args.models:
        print("[INFO] Iniciando o treinamento dos modelos de baseline...")
        train_and_evaluate_models(
            baseline_models, time_series_dict, config, no_window_results_path
        )

    if "deep_learning" in args.models:
        print("[INFO] Iniciando o treinamento dos modelos de deep learning...")
        train_and_evaluate_models(
            dl_models, time_series_dict, config, no_window_results_path
        )

    print("[INFO] Treinamento concluído.")
