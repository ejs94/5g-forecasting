import os
import warnings
import torch

import pandas as pd

from tqdm.auto import tqdm
from darts import TimeSeries
from darts.utils.utils import SeasonalityMode

from utils import convert_dfs_to_ts, preprocess_list_ts, separate_by_uid_and_frequency, training_model_for_activity

warnings.filterwarnings("ignore")

from darts.models import (
    ARIMA,
    FFT,
    AutoARIMA,
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    Prophet,
    Theta,
)

from darts.utils.statistics import (
    check_seasonality,
    stationarity_test_adf,
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
df_static = pd.read_parquet(os.path.join(data_path, "5G_df_static.parquet.gzip"))
df_driving = pd.read_parquet(
    os.path.join(data_path, "5G_df_driving.parquet.gzip")
)
print("---Separando os conjuntos em: Streaming vs. Downloading---")
list_static_strm = df_static.query("User_Activity == 'Streaming Video'")
list_driving_strm = df_driving.query("User_Activity == 'Streaming Video'")
list_static_down = df_static.query("User_Activity == 'Downloading a File'")
list_driving_down = df_driving.query("User_Activity == 'Downloading a File'")

print("---Separando os conjuntos por uid único---")
list_static_strm = separate_by_uid_and_frequency(list_static_strm, config["target_columns"], "S")
list_driving_strm = separate_by_uid_and_frequency(
    list_driving_strm, config["target_columns"], "S"
)
list_static_down = separate_by_uid_and_frequency(list_static_down, config["target_columns"], "S")
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
    "driving_down": list_driving_down
}

print("---Iniciando os treinamentos---")
all_stats = []

print("---Naive Forecast---")
# Mapeamento dos modelos
models = {
    "Naive": NaiveSeasonal(K=1),
    "NaiveDrift": NaiveDrift(),
    "NaiveMovingAverage": NaiveMovingAverage(input_chunk_length=config["K"]),
    "NaiveMean":NaiveMean(),
    "ExponentialSmoothing": ExponentialSmoothing(),
    "LinearRegression": LinearRegressionModel(lags=10),
    "AutoARIMA": AutoARIMA(start_p=0, start_q=0, max_order=4, test='adf', error_action='ignore', suppress_warnings=True),
    "Theta": Theta(theta=1.0),
    "FFT": FFT(),
    "Prophet": Prophet(),
}

for model_name, model in models.items():
    for activity, series_list in activities.items():
        print(f"---Iniciando treinamento para a atividade: {activity} com o modelo: {model_name}---")
        
        # Nome do arquivo de saída
        output_file = f"uni_{model_name}_{activity}"
        
        # Treinamento do modelo para a atividade
        stats = training_model_for_activity(
            activity,
            model_name,
            model,
            series_list,
            config["target_columns"],
            output_file,
            config["K"],
            config["H"]
        )

        # Adiciona as estatísticas da atividade à lista geral
        all_stats.append(stats)

# Define o caminho para salvar o arquivo de resultados
output_all_stats_path = os.path.join(os.curdir, "data", "results", "uni_all_stats.parquet")

# Cria o diretório se não existir
os.makedirs(os.path.dirname(output_all_stats_path), exist_ok=True)

# Concatena todos os DataFrames e realiza o processamento
all_stats_df = pd.concat(all_stats).reset_index()
all_stats_df.set_index(["Model", "target", "Activity"], inplace=True)

# Salva o DataFrame resultante em um arquivo Parquet
all_stats_df.to_parquet(output_all_stats_path, compression='gzip')

# Print informando que todas as estatísticas estão sendo salvas
print(f"Salvando as estatísticas agrupadas dos modelos em {output_all_stats_path}...")


print("---Finalizado---")
