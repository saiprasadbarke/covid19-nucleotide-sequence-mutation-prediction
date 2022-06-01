# Standard

# Local
from json import dump
from data_operations.tokenize_data import Tokenize
from data_operations.get_dataloaders import get_dataloader
from model_training.train_model import train_loop
from model_components.create_model import create_model
from settings.constants import CURRENT_RUN_DIR, NUM_SPECIAL_CHARS


def train():
    ##################################### Data Operations
    kmer_size = int(input("--------->Enter a kmer size between [2, 4]:   "))
    tokenizer = Tokenize(kmer_length=kmer_size)
    train_inputs, train_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="train")
    val_inputs, val_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="val")
    minibatch_size = int(input("--------->Choose a batch size.\nPreferably a power of 2 :   "))
    train_dataloader = get_dataloader(train_inputs, train_targets, minibatch_size)
    val_dataloader = get_dataloader(val_inputs, val_targets)

    ##################################### Model parameters
    len_vocabulary = 4**kmer_size + NUM_SPECIAL_CHARS
    embedding_size = int(input("--------->Choose size of embedding :   "))
    rnn_hidden_size = int(input("--------->Choose RNN hidden size. :   "))
    rnn_num_layers = int(input("--------->Choose RNN number of layers. :   "))
    dropout = float(input("--------->Choose dropout. :   "))
    embedded_dropout = float(input("--------->Choose embedded dropout. :   "))
    model = create_model(
        vocab_size=len_vocabulary,
        embedding_size=embedding_size,
        hidden_size=rnn_hidden_size,
        num_layers=rnn_num_layers,
        dropout=dropout,
        emb_dropout=embedded_dropout,
    )

    ##################################### Training parameters
    number_of_epochs = int(input("--------->Choose number of epochs to train the model :   "))
    learning_rate = float(input("--------->Choose the learning rate :   "))
    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        num_epochs=number_of_epochs,
        learning_rate=learning_rate,
    )

    ##################################### Save Model Parameters
    model_parameters_dict = {
        "vocab_size": len_vocabulary,
        "embedding_size": embedding_size,
        "hidden_size": rnn_hidden_size,
        "num_layers": rnn_num_layers,
        "dropout": dropout,
        "emb_dropout": embedded_dropout,
    }
    model_parameters_dict_path = f"{CURRENT_RUN_DIR}/model_parameters.json"
    with open(model_parameters_dict_path, "w") as f:
        dump(model_parameters_dict, f)
