import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values


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

    return ts_imputed_initial