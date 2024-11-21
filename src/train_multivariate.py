import os
import warnings
import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar
from tqdm.auto import tqdm

from utils import (
    convert_dfs_to_ts,
    separate_by_uid_and_frequency,
    train_and_evaluate_global_model  # Importando a função de treinamento
)


print("---Verificando se há GPU---")
# Verifica se a GPU está disponível
if torch.cuda.is_available():
    print("A GPU está disponível.")
else:
    print("A GPU NÃO está disponível. Rodando na CPU.")

print("---Configuração Utilizada---")
config = {"H": 10, "K": 50, "target_columns": ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]}
print(config)

print("---Carregando os dados preprocessados---")
data_path = os.path.join(os.curdir, "data")
df_static = pd.read_parquet(os.path.join(data_path, "5G_df_static.parquet"))
df_driving = pd.read_parquet(os.path.join(data_path, "5G_df_driving.parquet"))

print("---Separando os conjuntos em: Streaming vs. Downloading---")
list_static_strm = df_static.query("User_Activity == 'Streaming Video'")
list_driving_strm = df_driving.query("User_Activity == 'Streaming Video'")
list_static_down = df_static.query("User_Activity == 'Downloading a File'")
list_driving_down = df_driving.query("User_Activity == 'Downloading a File'")

print("---Separando os conjuntos por uid único---")
list_static_strm = separate_by_uid_and_frequency(
    list_static_strm, config["target_columns"], "S"
)
list_driving_strm = separate_by_uid_and_frequency(
    list_driving_strm, config["target_columns"], "S"
)
list_static_down = separate_by_uid_and_frequency(
    list_static_down, config["target_columns"], "S"
)
list_driving_down = separate_by_uid_and_frequency(
    list_driving_down, config["target_columns"], "S"
)

print("---Convertendo Dataframe para Timeseries (Darts)---")
list_static_strm = convert_dfs_to_ts(list_static_strm, config["target_columns"])
list_driving_strm = convert_dfs_to_ts(list_driving_strm, config["target_columns"])
list_static_down = convert_dfs_to_ts(list_static_down, config["target_columns"])
list_driving_down = convert_dfs_to_ts(list_driving_down, config["target_columns"])

# Mapeamento das listas para suas respectivas atividades
activities = {
    "static_strm": list_static_strm,
    "driving_strm": list_driving_strm,
    "static_down": list_static_down,
    "driving_down": list_driving_down,
}

print("---Configurando os modelos---")

def generate_torch_kwargs():
    # run torch models on device, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "gpu",
            "devices": [0],
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

# Mapeamento dos modelos
models = {
    "NBEATS": NBEATSModel(
    input_chunk_length=config["K"],
    output_chunk_length=config["H"],
    n_epochs=100,
    random_state=None,
    **generate_torch_kwargs()),
}

print("---Iniciando os treinamentos---")

for model_name, model in models.items():
    for activity, series_list in activities.items():
        print(
            f"---Iniciando treinamento para a atividade: {activity} com o modelo: {model_name}---"
        )

        # Nome do arquivo de saída
        output_file = f"multi_{model_name}_{activity}"

        # Chama a função para treinar e avaliar o modelo global
        success = train_and_evaluate_global_model(
            activity=activity,
            model_name=model_name,
            model=model,
            list_series=series_list,
            target_columns=config["target_columns"],
            output_file=output_file,
            H=config["H"]
        )
        
        if success:
            print(f"Modelo {model_name} para {activity} treinado e avaliado com sucesso!")
        else:
            print(f"Falha no treinamento do modelo {model_name} para {activity}.")

print("---Finalizado---")
