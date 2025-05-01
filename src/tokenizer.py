from torch.utils.data import random_split, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path

from src.constants import UNKNOWN_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, DATASET_NAME, DATASET_PATH
from src.dataset import BilingualDataset


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
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=UNKNOWN_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=[UNKNOWN_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN],
            min_frequency=1
        )

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    """
    Get the dataset.
    """
    #ds_raw = load_dataset(DATASET_NAME, lang1=config["lang_src"], lang2=config["lang_tgt"], split='train')
    ds_raw = load_dataset("json", data_files={"train": "data/test_sentences.jsonl"}, split="train")

    # Build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split the data
    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])

    # debug
    val_ds_raw = train_ds_raw

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


