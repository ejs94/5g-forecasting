import numpy as np
import pandas as pd

def preprocess_5G_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "Latitude",
        "Longitude",
        "Operatorname",
        "CellID",
        "PINGAVG",
        "PINGMIN",
        "PINGMAX",
        "PINGSTDEV",
        "PINGLOSS",
        "CELLHEX",
        "NODEHEX",
        "LACHEX",
        "RAWCELLID",
        "NRxRSRP",
        "NRxRSRQ",
        "Mobility",
    ]
    cleaned = df.drop(cols_to_drop, axis=1)

    # Convert unkown string to datetime64
    # add TZ +1000 for Dublin, Ireland UTC
    cleaned["Timestamp"] = (
        cleaned["Timestamp"]
        .apply(
            lambda row: row[:9].replace(".", "-")
            + row[9:].replace(".", ":").replace("_", " ")
        )
        .astype("datetime64[ns]")
    )

    # Rename '-' to NaN values
    cleaned[["RSRQ", "SNR", "CQI", "RSSI"]] = cleaned[
        ["RSRQ", "SNR", "CQI", "RSSI"]
    ].replace("-", np.nan)

    # Change objects columns to int64 dtype
    # cleaned[["RSRQ","SNR","CQI", "RSSI"]] = cleaned[["RSRQ","SNR","CQI", "RSSI"]].astype(float).astype('Int64')

    # Convert selected columns to float
    metric_columns = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate", "Speed"]
    cleaned[metric_columns] = cleaned[metric_columns].astype(float)

    # Configurar a coluna de data/hora como índice
    cleaned = cleaned.set_index("Timestamp")

    cleaned_dfs = []

    for uid in cleaned.Uid.unique():
        df_uid = cleaned[cleaned.Uid == uid]
        df_uid = df_uid[~df_uid.index.duplicated(keep="first")]
        cleaned_dfs.append(df_uid)

    cleaned = pd.concat(cleaned_dfs).sort_index()

    return cleaned