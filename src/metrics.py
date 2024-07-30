import time
from typing import Any, Dict, Optional, Tuple

import humanize
from darts import TimeSeries

# from darts.models import ForecastingModel
from darts.metrics import mae, mase, mse


class LogTime:
    """
    Class to measure execution time of code blocks, using the 'humanize' library
    to format the duration in a readable way.
    """

    def __init__(self, verbose: bool = True, **humanize_kwargs: Any) -> None:
        """
        Initializes the LogTime class.

        :param verbose: If True, prints the elapsed time upon exiting the context.
        :param humanize_kwargs: Additional arguments for the humanized time formatting.
        """
        if "minimum_unit" not in humanize_kwargs.keys():
            humanize_kwargs["minimum_unit"] = "microseconds"
        self.humanize_kwargs = humanize_kwargs
        self.elapsed: Optional[float] = None
        self.verbose = verbose

    def __enter__(self) -> "LogTime":
        """
        Starts timing when entering the context.
        """
        self.start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        """
        Calculates the elapsed time when exiting the context and, if verbose is True, prints the elapsed time.

        :param *args: Arguments captured, including exceptions, which are not handled here.
        """
        self.elapsed = time.time() - self.start
        self.elapsed_str = humanize.precisedelta(self.elapsed, **self.humanize_kwargs)
        if self.verbose:
            print(f"Time Elapsed: {self.elapsed_str}")


def eval_model(
    model,
    ts_train: TimeSeries,
    ts_test: TimeSeries,
    name: Optional[str] = None,
) -> Tuple[TimeSeries, Dict[str, Any]]:
    """
    Evaluates a forecasting model with specific metrics.

    :param model: Forecasting model to be evaluated.
    :param ts_train: Training time series.
    :param ts_test: Testing time series.
    :param name: Name of the model (optional). If None, uses the class name of the model.

    :return: Model predictions and a dictionary with the algorithm name and evaluation metrics (MAE, MSE, MASE).
    """

    if name is None:
        name = type(model).__name__

    model.fit(ts_train)

    y_pred = model.predict(len(ts_test))

    return y_pred, {
        "Algorithm": name,
        "MAE": mae(actual_series=ts_test, pred_series=y_pred),
        "MSE": mse(actual_series=ts_test, pred_series=y_pred),
        "MASE": mase(actual_series=ts_test, pred_series=y_pred, insample=ts_train),
    }
