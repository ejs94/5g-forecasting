import os
import warnings
import torch

print("---Verificando se há GPU---")
# Verifica se a GPU está disponível
if torch.cuda.is_available():
    print("A GPU está disponível.")
else:
    print("A GPU NÃO está disponível. Rodando na CPU.")


print("---Finalizado---")
