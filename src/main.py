import torch
from darts import TimeSeries
import pandas as pd

# Verifica se a GPU está disponível
if torch.cuda.is_available():
    print("A GPU está disponível.")
else:
    print("A GPU NÃO está disponível. Rodando na CPU.")

# Exemplo de criação de uma série temporal simples
data = {"time": pd.date_range("20210101", periods=10), "value": [i for i in range(10)]}
df = pd.DataFrame(data)
series = TimeSeries.from_dataframe(df, "time", "value")

print(series)
