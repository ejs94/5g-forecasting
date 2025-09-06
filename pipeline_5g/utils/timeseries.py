import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import (
    extract_subseries,
    fill_missing_values,
    missing_values_ratio,
)

def create_target_timeseries(data: pd.DataFrame, targets: list[str], timestamp_col: str ="Timestamp", freq: str | None ="s") -> TimeSeries | None:
    """
    Cria uma TimeSeries univariada a partir de um DataFrame ou de uma linha de DataFrame.
    """
    try:
        new_columns = [timestamp_col] + targets

        df = data[new_columns]

        # Verificar se há NaT nos timestamps após a conversão
        if df[timestamp_col].isnull().any():
            print(f"[WARNING] Timestamps inválidos (NaT) detectados e removidos para '{targets}'.")
            df = df.dropna(subset=[timestamp_col]) # Remove linhas com NaT

        if df.empty:
            print(f"[INFO] DataFrame vazio após processamento de timestamps para '{targets}'. Retornando None.")
            return None

        ts_darts = TimeSeries.from_dataframe(df,
            time_col="Timestamp",
            value_cols=targets,
            freq=freq,
            )
        
        return ts_darts

    except Exception as e:
        print(f"[ERROR] Falha ao criar TimeSeries alvo para '{targets}': {e}")
        return None

def create_covariates_timeseries(data: pd.DataFrame, covariate_cols: list[str], timestamp_col: str ="Timestamp", freq: str | None ="s") -> TimeSeries | None:
    """
    Cria uma TimeSeries multivariada para as covariáveis a partir de um DataFrame (ou Series/lista simulando uma linha de DataFrame).

    """
    try:

        new_columns = [timestamp_col] + covariate_cols

        df = data[new_columns]

        if df.empty:
             print("[INFO] DataFrame vazio após processar covariáveis. Retornando None.")
             return None

        ts_darts = TimeSeries.from_dataframe(df,
            time_col="Timestamp",
            value_cols=covariate_cols,
            freq=freq,
            )
        
        return ts_darts

    except Exception as e:
        print(f"[ERROR] Falha ao criar TimeSeries de covariáveis: {e}")
        return None

def impute_timeseries_missing_values(
    ts: TimeSeries | None,
    fill_all_nan_with: float | None = 0.0
) -> TimeSeries | None:
    """
    Imputa valores ausentes em uma TimeSeries Darts (univariada ou multivariada).
    """
    if ts is None or len(ts) == 0:
        return ts

    # see pandas.DataFrame.interpolate for params
    ts_imputed_initial = fill_missing_values(ts)

    # if ts_imputed_initial.width > 1:
    #     # Multivariada
    #     components_data = []
    #     for component_name in ts_imputed_initial.components:
    #         comp_ts = ts_imputed_initial[component_name]
    #         values = comp_ts.values(copy=False)

    #         if np.isnan(values).all():
    #             if fill_all_nan_with is not None:
    #                 filled = pd.Series(fill_all_nan_with, index=comp_ts.time_index, name=component_name)
    #             else:
    #                 filled = pd.Series(np.nan, index=comp_ts.time_index, name=component_name)
    #         else:
    #             filled = comp_ts.to_series().bfill()
    #             if filled.isnull().any() and fill_all_nan_with is not None:
    #                 filled = filled.fillna(fill_all_nan_with)

    #         components_data.append(filled)

    #     df_final = pd.concat(components_data, axis=1)
    #     ts_imputed = TimeSeries.from_dataframe(df_final, freq=ts_imputed_initial.freq_str, fill_missing_dates=True)

    # else:
    #     # Univariada
    #     values = ts_imputed_initial.values(copy=False)
    #     comp_name = ts_imputed_initial.components[0]

    #     if np.isnan(values).all():
    #         if fill_all_nan_with is not None:
    #             filled = pd.Series(fill_all_nan_with, index=ts_imputed_initial.time_index, name=comp_name)
    #             ts_imputed = TimeSeries.from_series(filled, freq=ts_imputed_initial.freq_str, fill_missing_dates=True)
    #         else:
    #             ts_imputed = ts_imputed_initial
    #     else:
    #         filled = ts_imputed_initial.to_series().bfill()
    #         if filled.isnull().any() and fill_all_nan_with is not None:
    #             filled = filled.fillna(fill_all_nan_with)
    #         ts_imputed = TimeSeries.from_series(filled, freq=ts_imputed_initial.freq_str, fill_missing_dates=True)

    # return ts_imputed
    return ts_imputed_initial