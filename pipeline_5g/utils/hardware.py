import multiprocessing

import torch
from pytorch_lightning.callbacks import TQDMProgressBar


def get_torch_device_config(verbose: bool = True) -> dict:
    """
    Detecta se há GPU disponível e retorna os kwargs para instanciar um Trainer do PyTorch Lightning.

    Parâmetros:
    - verbose: Se True, imprime mensagens informativas.

    Retorna:
    - Um dicionário com as configurações do PyTorch Lightning trainer.
    """
    if verbose:
        print("---Verificando se há GPU---")

    if torch.cuda.is_available():
        if verbose:
            print("GPU detectada. Usando PyTorch com acelerador GPU.")
        return {
            "pl_trainer_kwargs": {
                "accelerator": "gpu",
                "devices": [0],
                "callbacks": [TQDMProgressBar()],
                "enable_progress_bar": True,
            }
        }

    else:
        num_threads = multiprocessing.cpu_count()
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

        if verbose:
            print(
                f"GPU não detectada. Configurando PyTorch para usar CPU com {num_threads} threads."
            )

        return {
            "pl_trainer_kwargs": {
                "accelerator": "cpu",
                "callbacks": [TQDMProgressBar()],
                "enable_progress_bar": True,
            }
        }
