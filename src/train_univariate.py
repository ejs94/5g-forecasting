import os
import warnings
import torch

import pandas as pd

from tqdm.auto import tqdm
from darts import TimeSeries

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

print("---Carregando os dados preprocessados---")
print("---Finalizado---")