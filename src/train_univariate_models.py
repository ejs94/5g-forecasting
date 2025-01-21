import json
import os
import pickle
import time
import warnings
import traceback
import gc

import pandas as pd
import torch
from darts import TimeSeries, concatenate
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.models import (
    FFT,
    AutoARIMA,
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
from darts.utils.statistics import (
    check_seasonality,
    stationarity_test_adf,
)
from darts.utils.utils import SeasonalityMode
from tqdm.auto import tqdm

from utils import (
    convert_dfs_to_ts,
    preprocess_list_ts,
    separate_by_uid_and_frequency,
)

warnings.filterwarnings("ignore")

print("---Verificando se há GPU---")
# Verifica se a GPU está disponível
if torch.cuda.is_available():
    print("A GPU está disponível.")
else:
    print("A GPU NÃO está disponível. Rodando na CPU.")

# Caminhos
config_path = os.path.join(os.curdir, "config.json")
data_path = os.path.join(os.curdir, "data")
sliding_window_path = os.path.join(data_path, "sliding_window_datasets")

print("---Verificando a Configuração---")
# Verifica se o arquivo de configuração já existe; caso contrário, cria um
if not os.path.exists(config_path):
    config = {
        "H": 10,  # Tamanho da janela de previsão
        "K": 50,  # Tamanho da janela de entrada
        "update_interval": 10,  # Atualização da janela deslizante
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
activities = ["static_down", "static_strm", "driving_down", "driving_strm"]

# Carregar os dados preprocessados para cada atividade
time_series_dict = {}  # Dicionário para armazenar as séries temporais por atividade

for activity in activities:
    pickle_file_path = os.path.join(
        sliding_window_path, f"{activity}_sliding_window.pkl"
    )
    if os.path.exists(pickle_file_path):
        print(f"Carregando os dados pré-processados para {activity}...")
        with open(pickle_file_path, "rb") as f:
            time_series_list = pickle.load(f)
            time_series_dict[activity] = time_series_list
        print(f"Dados de {activity} carregados com sucesso.")
    else:
        print(f"Arquivo Pickle para {activity} não encontrado.")

print("---Configurando os modelos Baselines---")
# Todos os modelos configurados neste script serão tratados como univariados.
# Isso significa que cada métrica alvo será prevista individualmente, sem considerar dependências entre diferentes métricas.
baseline_models = {
    "Naive": NaiveSeasonal(K=1),
    "NaiveDrift": NaiveDrift(),
    "NaiveMovingAverage": NaiveMovingAverage(input_chunk_length=config["K"]),
    "NaiveMean": NaiveMean(),
    "ExponentialSmoothing": ExponentialSmoothing(seasonal=None),
    "LinearRegression": LinearRegressionModel(lags=1),
    "AutoARIMA": AutoARIMA(
        start_p=1,  # Ordem inicial para o componente AR (Auto-Regressivo)
        start_q=1,  # Ordem inicial para o componente MA (Média Móvel)
        d=None,  # Deixa o AutoARIMA determinar automaticamente o grau de diferenciação
        seasonal=False,  # Não considera a sazonalidade nos dados
        start_P=0,  # Ordem inicial para o componente sazonal AR (não utilizado, pois seasonal=False)
        start_Q=0,  # Ordem inicial para o componente sazonal MA (não utilizado, pois seasonal=False)
        max_order=10,  # Limita a soma de (p + q + P + Q) para reduzir complexidade
        stepwise=True,  # Usa abordagem de busca em passos para otimização (mais rápido)
        error_action="ignore",  # Ignora erros ao tentar ajustar certos modelos
        suppress_warnings=True,  # Suprime os warnings durante o ajuste
        test="adf",  # Teste de estacionaridade a ser usado (ADF é uma boa escolha padrão)
        seasonal_test="ocsb",  # Teste de sazonalidade (não será usado, pois seasonal=False)
        max_p=3,  # Limite superior para a ordem AR
        max_q=3,  # Limite superior para a ordem MA
        max_P=0,  # Não há componente sazonal AR (max_P=0)
        max_Q=0,  # Não há componente sazonal MA (max_Q=0)
        max_d=2,  # Máximo grau de diferenciação
        max_D=0,  # Não há sazonalidade, então max_D=0
        trace=True,  # Mostra o progresso do ajuste para depuração
        n_jobs=-1,  # Permite paralelismo para acelerar o ajuste (usa todos os núcleos disponíveis)
        random_state=42,  # Para reprodutibilidade
    ),
    "Theta": Theta(theta=1.0),
    "FFT": FFT(),
    "Prophet": Prophet(),
}

print("---Configurando os modelos Machine Learning---")
# Todos os modelos configurados neste script serão tratados como univariados.
# Isso significa que cada métrica alvo será prevista individualmente, sem considerar dependências entre diferentes métricas.
dl_models = {
    # "NBEATS": NBEATSModel(
    #     input_chunk_length=config["K"],
    #     output_chunk_length=config["H"],
    #     generic_architecture=True,
    #     num_stacks=10,
    #     num_blocks=1,
    #     num_layers=4,
    #     layer_widths=512,
    #     n_epochs=100,
    #     nr_epochs_val_period=1,
    #     batch_size=64,  # Considerando ajustar para um valor menor, como 64
    #     random_state=None,
    # ),
    "LSTM": RNNModel(
        model="LSTM",
        input_chunk_length=3 * config["H"],
        output_chunk_length=config["H"],
        training_length=3 * config["H"],
        hidden_dim=50,
        n_rnn_layers=4,
        dropout=0.0,
        batch_size=64,
        n_epochs=100,
    ),
    # "LightGBM": LightGBMModel(
    #     lags=config["K"],
    #     output_chunk_length=config["H"],
    # ),
}


def train_model_in_dict(models_dict, result_path):
    filler = MissingValuesFiller(
        fill="auto"  # or 0.0
    )  # Justification for fill=0.0: Interpolation fails for multi-series with entirely NaN columns, as there's no value to interpolate from. Using fill=0.0 ensures consistent handling by replacing NaNs with zeros.
    scaler = Scaler()  # Escalona os dados
    pipe = Pipeline([filler, scaler])

    if not models_dict:
        print("Nenhum modelo configurado. Pulando o treinamento.")
    else:
        os.makedirs(baseline_results_path, exist_ok=True)
        start_time = time.time()

    for model_name, model in tqdm(
        models_dict.items(), desc="Treinando Baseline Models"
    ):
        result_records = []
        for activity, list_series in tqdm(
            time_series_dict.items(),
            desc=f"Treinando atividade ({model_name})",
            leave=False,
        ):
            tqdm.write(f"[INFO] modelo: {model_name} -- atividade: {activity}")

            for series in tqdm(
                list_series, desc=f"Treinando séries ({activity})", leave=False
            ):
                ts_train = series["train"]
                ts_test = series["test"]

                for kpi in ts_train.columns:
                    train = ts_train[kpi]
                    test = ts_test[kpi]
                    ts_transformed = pipe.fit_transform(train)
                    try:
                        model.fit(ts_transformed)
                        y_pred = model.predict(len(test))

                        result_records.append(
                            pd.DataFrame(
                                {
                                    "target": kpi,
                                    "Activity": activity,
                                    "Model": model_name,
                                    "Train_index": [
                                        pipe.inverse_transform(
                                            ts_transformed, partial=True
                                        )
                                        .pd_dataframe()
                                        .index.to_numpy()
                                    ],
                                    "Train_values": [
                                        pipe.inverse_transform(
                                            ts_transformed, partial=True
                                        )
                                        .pd_dataframe()
                                        .values.flatten()
                                    ],
                                    "Actuals_index": [
                                        test.pd_dataframe().index.to_numpy()
                                    ],
                                    "Actuals_values": [
                                        test.pd_dataframe().values.flatten()
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
                    except Exception as e:
                        tqdm.write(
                            f"[ERRO] Problema ao processar a série '{kpi}' na atividade '{activity}' com o modelo '{model_name}': {e}"
                        )
                        traceback.print_exc()
                        continue

        if result_records:
            result_record = pd.concat(result_records)
            output_file = os.path.join(result_path, f"{model_name}.parquet")
            result_record.to_parquet(output_file, compression="gzip")
            tqdm.write(f"[SUCESSO] Resultados de {activity} salvos em: {output_file}")
        else:
            tqdm.write(
                f"[AVISO] Nenhum resultado para a atividade '{activity}' com o modelo '{model_name}'"
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Tempo total de execução: {elapsed_time:.2f} segundos.")
            # Liberar memória
            del result_records, result_record
            gc.collect()


# Treinamento dos modelos Baselines
print("---Iniciando os treinamentos---")
print("---Modelos Baselines---")

baseline_results_path = os.path.join(data_path, "results", "baseline")
os.makedirs(baseline_results_path, exist_ok=True)
train_model_in_dict(baseline_models, baseline_results_path)

# Treinamento dos modelos de Deep Learning
print("---Modelos Deep Learning---")
os.makedirs(baseline_results_path, exist_ok=True)
dl_results_path = os.path.join(data_path, "results", "dl_models")
train_model_in_dict(dl_models, dl_results_path)
