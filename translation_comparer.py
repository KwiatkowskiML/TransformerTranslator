import json
import torch
import sacrebleu
from transformer.src.config import get_config, latest_weights_file_path
from transformer.train import get_model, get_ds, greedy_decode
from transformer.src.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def load_test_sentences(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())['translation']


def prepare_encoder_input(tokenizer_src, text, seq_len, device):
    tokens = tokenizer_src.encode(text).ids
    sos_id = tokenizer_src.token_to_id(SOS_TOKEN)
    eos_id = tokenizer_src.token_to_id(EOS_TOKEN)
    pad_id = tokenizer_src.token_to_id(PAD_TOKEN)

    ids = [sos_id] + tokens + [eos_id]
    if len(ids) > seq_len:
        raise ValueError(f"Sequence length {len(ids)} exceeds max seq_len={seq_len}")
    ids += [pad_id] * (seq_len - len(ids))

    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    mask = (tensor != pad_id).unsqueeze(1).unsqueeze(2)
    return tensor, mask


def decode_prediction(ids, tokenizer_tgt):
    eos_id = tokenizer_tgt.token_to_id(EOS_TOKEN)
    sos_id = tokenizer_tgt.token_to_id(SOS_TOKEN)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and ids[0] == sos_id:
        ids = ids[1:]
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    decoded_text = tokenizer_tgt.decode(ids)
    # Clean spacing around punctuation
    decoded_text = decoded_text.replace(' .', '.').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
    return decoded_text.strip()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = get_config()
    _, _, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load pretrained weights
    weights_path = latest_weights_file_path(config)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # Process each sentence and compute sentence-level BLEU
    for pair in load_test_sentences('data/test_sentences.jsonl'):
        en, pl_ref = pair['en'], pair['pl']
        encoder_input, encoder_mask = prepare_encoder_input(
            tokenizer_src, en, config['seq_len'], device
        )
        out_ids = greedy_decode(
            model, encoder_input, encoder_mask,
            tokenizer_src, tokenizer_tgt,
            config['seq_len'], device
        )
        pl_pred = decode_prediction(out_ids, tokenizer_tgt)

        # Compute sentence-level BLEU with sacrebleu
        bleu = sacrebleu.sentence_bleu(pl_pred, [pl_ref])

        print('-' * 80)
        print(f"{'SOURCE: ':>12}{en}")
        print(f"{'TARGET: ':>12}{pl_ref}")
        print(f"{'PREDICTED: ':>12}{pl_pred}")
        print(f"{'BLEU: ':>12}{bleu.score:.2f}\n")

if __name__ == '__main__':
    main()
