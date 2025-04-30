import warnings
import torch.nn as nn
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import get_weights_file_path, get_config
from model import build_transformer
from tokenizer import get_ds


def get_model(model_config, vocab_src_len, vocab_tgt_len):
    """
    Build the model
    """
    model  = build_transformer(vocab_src_len, vocab_tgt_len, model_config['seq_len'], model_config['seq_len'], model_config['d_model'])
    return model


def train_model(model_config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(model_config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(model_config)
    model = get_model(model_config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(model_config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if model_config['preload']:
        model_filename = get_weights_file_path(model_config, model_config['preload'])
        print(f'Preloading model from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, model_config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model
        model_filename = get_weights_file_path(model_config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
