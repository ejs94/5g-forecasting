import os
import glob
import pandas as pd
import numpy as np

# Agora você pode importar o módulo utils Ou importar funções específicas do módulo
from utils import extract_5G_dataset, preprocess_5G_dataframe

original_path = os.path.join(os.curdir, "datasets", "5G-production-dataset")

print("---Extraindo o 5G Dataset---")
df_static, df_driving = extract_5G_dataset(original_path)

print("---Preprocessando o 5G Dataset---")
df_static = preprocess_5G_dataframe(df_static)
df_driving = preprocess_5G_dataframe(df_driving)

print("---Salvando o 5G Dataset Processado---")

print(df_static.info())

save_path = os.path.join(os.curdir, "data", "5G_df_static.parquet")
df_static.to_parquet(save_path, compression="gzip")


print(df_driving.info())
save_path = os.path.join(os.curdir, "data", "5G_df_driving.parquet")
df_driving.to_parquet(save_path, compression="gzip")

print("---Finalizado---")