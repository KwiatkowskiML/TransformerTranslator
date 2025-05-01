from pathlib import Path

from src.constants import D_MODEL_SIZE, BATCH_SIZE, EPOCH_COUNT, LEARNING_RATE, SEQ_LEN


def get_config():
    """
    Get the configuration for the model
    """
    return {
        "batch_size": BATCH_SIZE,
        "num_epochs": EPOCH_COUNT,
        "lr": LEARNING_RATE,
        "seq_len": SEQ_LEN,
        "d_model": D_MODEL_SIZE,
        "lang_src": "en",
        "lang_tgt": "pl",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Get the path for the weights file
    """
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)