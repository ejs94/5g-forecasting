import gc
import json
import multiprocessing
import os
import pickle
import time
import traceback
import warnings

import pandas as pd
import torch
from darts import TimeSeries, concatenate
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.models import (
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    NBEATSModel,
)
from darts.utils.missing_values import (
    extract_subseries,
    fill_missing_values,
    missing_values_ratio,
)
from tqdm.auto import tqdm

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
reduced_metrics_path = os.path.join(data_path, "reduced_metrics_datasets")

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
}

print("---Configurando os modelos Machine Learning---")

# Número total de threads disponíveis
# Configura o número de threads
num_threads = multiprocessing.cpu_count()
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

print(f"Configurando PyTorch para usar {num_threads} threads.")
dl_models = {
    "NBEATS": NBEATSModel(
        input_chunk_length=config["K"],
        output_chunk_length=config["H"],
        generic_architecture=True,
        num_stacks=10,
        num_blocks=1,
        num_layers=4,
        layer_widths=512,
        n_epochs=100,
        nr_epochs_val_period=1,
        batch_size=64,
        random_state=None,
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
    print(f"Missing ratio for {column_name}: {ratio_missing * 100:.2f}%")

    if ratio_missing > 0.4:  # Se mais de 40% dos valores estão ausentes, descartar
        print(f"Descartando a série {column_name} para o Uid {row['Uid']} (NaN > 40%)")
        return None

    # Preencher valores ausentes com backward fill
    ts = fill_missing_values(ts)

    # Dividir em sub-séries apenas se lacunas consecutivas forem maiores que o horizonte
    subseries = extract_subseries(ts, min_gap_size=horizon, mode="any")

    if not subseries:
        print(
            f"Nenhuma sub-série válida encontrada para {column_name} (Uid {row['Uid']})"
        )
        return None

    return subseries


def train_and_evaluate_models(models, time_series_dict, config, data_path):
    print("---Iniciando o treinamento dos modelos---")
    no_window_results_path = os.path.join(data_path, "results", "no_window")
    os.makedirs(no_window_results_path, exist_ok=True)

    scaler = Scaler()  # Escalona os dados
    pipe = Pipeline([scaler])

    start_time = time.time()

    for model_name, model in tqdm(models.items(), desc="Treinando Modelos"):
        result_records = []
        for activity, pd_series in tqdm(
            time_series_dict.items(), desc=f"Atividade ({model_name})", leave=False
        ):
            for column_name in config[
                "target_columns"
            ]:  # Iterar pelas colunas de métricas
                print(
                    f"Processando métrica {column_name} para {activity} com {model_name}..."
                )

                processed_series = []
                for idx, row in pd_series.iterrows():
                    # Processar a série para a métrica atual
                    subseries = train_process_timeseries(
                        row, column_name, horizon=config["H"]
                    )
                    if subseries:
                        processed_series.extend(subseries)

                if processed_series:
                    print(
                        f"Iniciando treinamento para {column_name} em {activity} com {model_name}..."
                    )

                    horizon = config["H"]

                    for series in tqdm(
                        processed_series,
                        desc=f"Treinando Série {idx}: {column_name} ({model_name})",
                        leave=False,
                    ):
                        # Ajustar o modelo com as séries processadas
                        train_data = series[:-horizon]
                        test_horizon = series[-horizon:]

                        ts_transformed = pipe.fit_transform(train_data)

                        model.fit(ts_transformed)
                        print(
                            f"Treinamento concluído para {column_name} em {activity} com {model_name}."
                        )

                        y_pred = model.predict(len(test_horizon))

                        # Salvar o modelo treinado
                        model_dir = os.path.join(
                            no_window_results_path, "models", activity, column_name
                        )
                        os.makedirs(model_dir, exist_ok=True)
                        model.save(
                            os.path.join(model_dir, f"lst_{model_name}_model.pkl")
                        )

                        # Salvar os resultados em result_records
                        result_records.append(
                            pd.DataFrame(
                                {
                                    "target": column_name,
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

            # Salvar os resultados da atividade em arquivos .parquet
            if result_records:
                result_record = pd.concat(result_records)
                output_file = os.path.join(
                    no_window_results_path, f"{model_name}_{activity}.parquet"
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                result_record.to_parquet(output_file, compression="gzip")
                tqdm.write(
                    f"[SUCESSO] Resultados de {activity} salvos em: {output_file}"
                )
            else:
                tqdm.write(
                    f"[AVISO] Nenhum resultado para a atividade '{activity}' com o modelo '{model_name}'"
                )

            # Liberar memória ao final de cada modelo
            del result_records, result_record
            gc.collect()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos.")


print("---Treinando os modelos---")

# Treinando os modelos de baseline
train_and_evaluate_models(baseline_models, time_series_dict, config, data_path)

# Treinando os modelos de machine learning (ML)
train_and_evaluate_models(dl_models, time_series_dict, config, data_path)
