import torch
from transformer.src.config import get_config, latest_weights_file_path
from transformer.train import get_model, get_ds, run_validation

if __name__ == "__main__":
    # Define the device
    device = torch.device("cpu")
    print("Using device:", device)

    # Load the configuration
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                   lambda msg: print(msg), 0, None, num_examples=20)