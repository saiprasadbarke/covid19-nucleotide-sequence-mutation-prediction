# Standard

# Local
from data_operations.tokenize_data import Tokenize
from data_operations.get_dataloaders import get_dataloader
from model_training.train_model import train_loop
from model_components.create_model import create_model
from settings.constants import NUM_SPECIAL_CHARS


def train():
    kmer_size = int(input("Enter a kmer size between [2, 4]:   "))
    tokenizer = Tokenize(kmer_length=kmer_size)
    train_inputs, train_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="train")
    val_inputs, val_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="val")
    minibatch_size = int(input("Choose a batch size.\nPreferably a power of 2 :   "))
    train_dataloader = get_dataloader(train_inputs, train_targets, minibatch_size)
    val_dataloader = get_dataloader(val_inputs, val_targets)
    len_vocabulary = 4**kmer_size + NUM_SPECIAL_CHARS
    embedding_size = int(input("Choose size of embedding :   "))
    rnn_hidden_size = int(input("Choose RNN hidden size. :   "))
    rnn_num_layers = int(input("Choose RNN number of layers. :   "))
    dropout = float(input("Choose dropout. :   "))
    model = create_model(
        vocab_size=len_vocabulary,
        embedding_size=embedding_size,
        hidden_size=rnn_hidden_size,
        num_layers=rnn_num_layers,
        dropout=dropout,
    )
    train_loop(model=model, train_dataloader=train_dataloader, validation_dataloader=val_dataloader)