import torch
import torch.nn as nn

from torch.utils.data import random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path

from constants import UNKNOWN_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, DATASET_NAME


def get_all_sentences(ds, lang):
    """
    Get all sentences from the dataset for the given language.
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Get or build a tokenizer for the given language.
    """
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=UNKNOWN_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=[UNKNOWN_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN],
            min_frequency=2
        )

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    """
    Get the dataset.
    """
    ds_raw = load_dataset(DATASET_NAME, f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split the data
    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])

    # TODO: dataset handling
