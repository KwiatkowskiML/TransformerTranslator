from typing import Final

D_MODEL_SIZE: Final[int] = 512
NUMBER_OF_LAYERS: Final[int] = 6
ATTENTION_HEADS_NUMBER: Final[int] = 8
DROPOUT: Final[float] = 0.1
HIDDEN_LAYER_SIZE: Final[int] = 2048

BATCH_SIZE: Final[int] = 8
EPOCH_COUNT: Final[int] = 20
LEARNING_RATE: Final[float] = 1e-4
SEQ_LEN: Final[int] = 128

UNKNOWN_TOKEN: Final[str] = "[UNK]"
PAD_TOKEN: Final[str] = "[PAD]"
EOS_TOKEN: Final[str] = "[EOS]"
SOS_TOKEN: Final[str] = "[SOS]"

DATASET_NAME: Final[str] = "tatoeba"
DATASET_PATH: Final[str] = "./data/test_sentences.csv"