import glob
import json
import os

import numpy as np
import pandas as pd
import shortuuid


def extract_5G_dataset(path: os.path) -> list[pd.DataFrame]:
    df_static = []
    df_driving = []

    files = glob.glob(f"{path}/**/*.csv", recursive=True)

    for file in files:
        file = os.path.normpath(file)
        df = pd.read_csv(file)
        folder_name, filename = os.path.split(file)

        df["Uid"] = shortuuid.uuid()[:8]

        streaming_services = ["Netflix", "Amazon_Prime"]
        if any(service in folder_name for service in streaming_services):
            df["User_Activity"] = "Streaming Video"

        if ("Download") in folder_name:
            df["User_Activity"] = "Downloading a File"

        if "Static" in folder_name:
            df["Mobility"] = "Static"
            df_static.append(df)

        if "Driving" in folder_name:
            df["Mobility"] = "Driving"
            df_driving.append(df)

    df_static = pd.concat(df_static, axis=0, ignore_index=True)
    df_driving = pd.concat(df_driving, axis=0, ignore_index=True)

    return [df_static, df_driving]


def load_or_create_config(config_path):
    """
    Carrega a configuração do arquivo JSON ou cria uma configuração padrão.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Lendo configuração existente de: {config_path}")
    else:
        config = {
            "test_ratio": 0.1,
            "update_interval": 10,
            "target_columns": ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"],
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Configuração inicial salva em: {config_path}")
    return config