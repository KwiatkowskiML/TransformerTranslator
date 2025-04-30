from typing import Final

D_MODEL_SIZE: Final[int] = 512
NUMBER_OF_LAYERS: Final[int] = 6
ATTENTION_HEADS_NUMBER: Final[int] = 8
DROPOUT: Final[float] = 0.1
HIDDEN_LAYER_SIZE: Final[int] = 2048

UNKNOWN_TOKEN: Final[str] = "[UNK]"
PAD_TOKEN: Final[str] = "[PAD]"
EOS_TOKEN: Final[str] = "[EOS]"
SOS_TOKEN: Final[str] = "[SOS]"

DATASET_NAME: Final[str] = "opus_books"