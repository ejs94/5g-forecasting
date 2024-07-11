import glob
import os
import time

import humanize
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
    cleaned[["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]] = cleaned[
        ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
    ].astype(float)

    # Configurar a coluna de data/hora como índice
    cleaned = cleaned.set_index("Timestamp")

    cleaned_dfs = []

    for uid in cleaned.Uid.unique():
        df_uid = cleaned[cleaned.Uid == uid]
        df_uid = df_uid[~df_uid.index.duplicated(keep="first")]
        cleaned_dfs.append(df_uid)

    cleaned = pd.concat(cleaned_dfs).sort_index()

    return cleaned


def compact_5G_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.reset_index()
    compact_df = (
        df.groupby("Uid")[["Timestamp", "RSRP", "RSRQ", "SNR", "CQI", "RSSI"]]
        .agg(lambda x: list(x))
        .reset_index()
    )
    return compact_df


class LogTime:
    def __init__(self, verbose=True, **humanize_kwargs) -> None:
        if "minimum_unit" not in humanize_kwargs.keys():
            humanize_kwargs["minimum_unit"] = "microseconds"
        self.humanize_kwargs = humanize_kwargs
        self.elapsed = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Exceptions are captured in *args, we’ll handle none, since failing can be timed anyway
        """
        self.elapsed = time.time() - self.start
        self.elapsed_str = humanize.precisedelta(self.elapsed, **self.humanize_kwargs)
        if self.verbose:
            print(f"Time Elapsed: {self.elapsed_str}")


def intersect_list(list1, list2):
    return list(set(list1).intersection(set(list2)))


def difference_list(list1, list2):
    return list(set(list1) - set(list2))


def union_list(list1, list2):
    return list(set(list1).union(set(list2)))
