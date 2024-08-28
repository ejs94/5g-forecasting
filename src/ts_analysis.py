from darts.utils.statistics import stationarity_test_adf, check_seasonality
from darts import TimeSeries


def analyze_time_series(
    series: TimeSeries, max_lag: int = 60, alpha: float = 0.05
) -> dict:
    """
    Analyzes a time series to determine if it is stationary and if it exhibits seasonality.

    Parameters:
    - series: TimeSeries object from darts library
    - alpha: Significance level for the ADF test (default is 0.05)

    Returns:
    - result: A dictionary containing:
      - 'is_stationary': True if the series is stationary, False otherwise
      - 'p_value': The p-value from the ADF test
      - 'has_seasonality': True if seasonality is detected, False otherwise
      - 'seasonal_period': The estimated seasonal period if seasonality is detected, None otherwise
    """
    # Check for stationarity using ADF test
    p_value = stationarity_test_adf(series)[1]
    is_stationary = p_value < alpha

    # Check for seasonality
    has_seasonality, seasonal_period = check_seasonality(series, max_lag=max_lag)

    # Prepare the result dictionary
    result = {
        "is_stationary": is_stationary,
        "has_seasonality": has_seasonality,
        "p_value": p_value,
        "seasonal_period": seasonal_period if has_seasonality else None,
    }

    return result
