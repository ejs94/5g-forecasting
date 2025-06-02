import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import (
    extract_subseries,
    fill_missing_values,
    missing_values_ratio,
)


def create_target_timeseries(
    row: pd.Series, 
    target_col: str, 
    timestamp_col: str = "Timestamp", 
    freq: str | None = "s" # Permitir que a frequência seja None para inferência ou customizada
) -> TimeSeries | None:
    """
    Cria uma série temporal Darts univariada para a variável alvo.

    Parâmetros:
    - row: Linha do DataFrame contendo listas de medições e timestamps.
    - target_col: Nome da coluna alvo.
    - timestamp_col: Nome da coluna de timestamps.
    - freq: Frequência da série temporal (ex: "s", "min", "H", "D"). 
            Se None, o Darts tentará inferir.

    Retorna:
    - TimeSeries univariada com o alvo, ou None se dados essenciais estiverem ausentes/inválidos.
    """
    try:
        timestamps = row.get(timestamp_col)
        target_values = row.get(target_col)

        if timestamps is None or target_values is None:
            print(f"[ERROR] Coluna de timestamp ('{timestamp_col}') ou alvo ('{target_col}') não encontrada na linha.")
            return None
        
        # Garantir que sejam iteráveis (listas, np.array, pd.Series)
        if not hasattr(timestamps, '__iter__') or not hasattr(target_values, '__iter__'):
            print("[ERROR] Timestamps ou valores alvo não são iteráveis.")
            return None

        if len(timestamps) != len(target_values):
            print(f"[ERROR] Comprimento dos timestamps ({len(timestamps)}) e valores alvo ({len(target_values)}) não coincidem.")
            return None
        
        if not timestamps: # Lista de timestamps vazia
            print(f"[INFO] Lista de timestamps vazia para '{target_col}'. Retornando None.")
            return None

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, errors='coerce'), # Coerce para NaT se houver erro de parse
            target_col: pd.to_numeric(target_values, errors='coerce') # Coerce para NaN se houver erro de parse
        })
        
        # Verificar se há NaT nos timestamps após a conversão
        if df["timestamp"].isnull().any():
            print(f"[WARNING] Timestamps inválidos (NaT) detectados e removidos para '{target_col}'.")
            df = df.dropna(subset=["timestamp"]) # Remove linhas com NaT

        if df.empty:
            print(f"[INFO] DataFrame vazio após processamento de timestamps para '{target_col}'. Retornando None.")
            return None

        return TimeSeries.from_dataframe(df, time_col="timestamp", value_cols=[target_col], freq=freq)

    except Exception as e:
        print(f"[ERROR] Falha ao criar TimeSeries alvo para '{target_col}': {e}")
        return None


def create_covariates_timeseries(
    row: pd.Series, 
    covariate_cols: list[str], 
    timestamp_col: str = "Timestamp",
    freq: str | None = "s" # Permitir que a frequência seja None para inferência ou customizada
) -> TimeSeries | None:
    """
    Cria uma série temporal Darts multivariada para as covariáveis.
    Se uma covariável específica estiver ausente na linha ou tiver dados inválidos,
    ela será preenchida com NaNs.

    Parâmetros:
    - row: Linha do DataFrame contendo listas de medições e timestamps.
    - covariate_cols: Lista de nomes das colunas de covariáveis.
    - timestamp_col: Nome da coluna de timestamps.
    - freq: Frequência da série temporal. Se None, o Darts tentará inferir.

    Retorna:
    - TimeSeries multivariada com as covariáveis, ou None se timestamps estiverem ausentes/inválidos.
    """
    try:
        timestamps = row.get(timestamp_col)
        if timestamps is None:
            print(f"[ERROR] Coluna de timestamp ('{timestamp_col}') não encontrada na linha para covariáveis.")
            return None
        
        if not hasattr(timestamps, '__iter__'):
            print("[ERROR] Timestamps para covariáveis não são iteráveis.")
            return None

        num_timestamps = len(timestamps)
        if num_timestamps == 0:
            print("[INFO] Lista de timestamps vazia para covariáveis. Retornando None.")
            return None

        df_dict = {"timestamp": pd.to_datetime(timestamps, errors='coerce')}
        
        valid_timestamps_mask = ~pd.Series(df_dict["timestamp"]).isnull() # Mascara para timestamps válidos
        num_valid_timestamps = valid_timestamps_mask.sum()

        if num_valid_timestamps == 0:
            print("[INFO] Nenhum timestamp válido para covariáveis após conversão. Retornando None.")
            return None
        
        # Filtrar timestamps inválidos e ajustar num_timestamps
        df_dict["timestamp"] = pd.Series(df_dict["timestamp"])[valid_timestamps_mask].tolist()
        current_num_timestamps = len(df_dict["timestamp"])


        processed_covariates = []
        for cov_col in covariate_cols:
            cov_values = row.get(cov_col)
            
            if cov_values is None or not hasattr(cov_values, '__iter__'):
                print(f"[WARNING] Covariável '{cov_col}' não encontrada ou não iterável. Preenchendo com NaNs.")
                df_dict[cov_col] = [np.nan] * current_num_timestamps
                processed_covariates.append(cov_col)
                continue

            # Aplicar coerção e máscara de timestamp também aos valores da covariável
            cov_series_numeric = pd.to_numeric(pd.Series(cov_values), errors='coerce')
            
            if len(cov_series_numeric) != num_timestamps: # Comprimento original antes de filtrar NaT
                print(f"[WARNING] Covariável '{cov_col}' tem comprimento {len(cov_values)}, esperado {num_timestamps}. Preenchendo com NaNs.")
                df_dict[cov_col] = [np.nan] * current_num_timestamps
            else:
                # Aplicar a mesma máscara de timestamps válidos
                df_dict[cov_col] = cov_series_numeric[valid_timestamps_mask].tolist()
            processed_covariates.append(cov_col)

        if not processed_covariates: # Nenhuma covariável foi processada (lista original vazia)
             print("[INFO] Lista de `covariate_cols` está vazia. Retornando None se não houver covariáveis para criar.")
             # Se for aceitável retornar uma TS sem colunas de valor (improvável para Darts), ajuste aqui.
             # Geralmente, se covariate_cols for vazia, esta função não deveria ser chamada ou deveria retornar None.
             return None


        df = pd.DataFrame(df_dict)
        if df.empty:
             print("[INFO] DataFrame vazio após processar covariáveis. Retornando None.")
             return None
             
        return TimeSeries.from_dataframe(df, time_col="timestamp", value_cols=processed_covariates, freq=freq)

    except Exception as e:
        print(f"[ERROR] Falha ao criar TimeSeries de covariáveis: {e}")
        return None

def impute_timeseries_missing_values(
    ts: TimeSeries | None,
    fill_all_nan_with: float | None = 0.0
) -> TimeSeries | None:
    """
    Imputa valores ausentes em uma TimeSeries Darts (univariada ou multivariada).
    Se um componente permanecer totalmente NaN após a imputação padrão,
    ele pode ser preenchido com `fill_all_nan_with`.

    Args:
        ts (TimeSeries | None): Série temporal de entrada.
        fill_all_nan_with (float | None): Valor para preencher componentes totalmente NaN, se desejado.

    Returns:
        TimeSeries | None: Série temporal imputada ou None se entrada inválida.
    """
    if ts is None or len(ts) == 0:
        return ts

    ts_imputed_initial = fill_missing_values(ts)

    if ts_imputed_initial.width > 1:
        # Multivariada
        components_data = []
        for component_name in ts_imputed_initial.components:
            comp_ts = ts_imputed_initial[component_name]
            values = comp_ts.values(copy=False)

            if np.isnan(values).all():
                if fill_all_nan_with is not None:
                    filled = pd.Series(fill_all_nan_with, index=comp_ts.time_index, name=component_name)
                else:
                    filled = pd.Series(np.nan, index=comp_ts.time_index, name=component_name)
            else:
                filled = comp_ts.to_series().bfill().ffill()
                if filled.isnull().any() and fill_all_nan_with is not None:
                    filled = filled.fillna(fill_all_nan_with)

            components_data.append(filled)

        df_final = pd.concat(components_data, axis=1)
        ts_imputed = TimeSeries.from_dataframe(df_final, freq=ts_imputed_initial.freq_str, fill_missing_dates=True)

    else:
        # Univariada
        values = ts_imputed_initial.values(copy=False)
        comp_name = ts_imputed_initial.components[0]

        if np.isnan(values).all():
            if fill_all_nan_with is not None:
                filled = pd.Series(fill_all_nan_with, index=ts_imputed_initial.time_index, name=comp_name)
                ts_imputed = TimeSeries.from_series(filled, freq=ts_imputed_initial.freq_str, fill_missing_dates=True)
            else:
                ts_imputed = ts_imputed_initial
        else:
            filled = ts_imputed_initial.to_series().bfill().ffill()
            if filled.isnull().any() and fill_all_nan_with is not None:
                filled = filled.fillna(fill_all_nan_with)
            ts_imputed = TimeSeries.from_series(filled, freq=ts_imputed_initial.freq_str, fill_missing_dates=True)

    return ts_imputed