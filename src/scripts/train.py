# Standard

# Local
from json import dump
from data_operations.tokenize_data import Tokenize
from data_operations.get_dataloaders import get_dataloader
from model_training.train_model import train_loop
from model_components.create_model import create_model
from model_components.weights import compute_2d_weight_vector
from settings.constants import CURRENT_RUN_DIR, NUM_SPECIAL_CHARS


def train():
    ##################################### Data Operations
    kmer_size = int(input("Enter a kmer size between [2, 4]:   "))
    tokenizer = Tokenize(kmer_length=kmer_size)
    train_inputs, train_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="train")
    val_inputs, val_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="val")
    weights = compute_2d_weight_vector(train_targets, tokenizer.vocabulary)
    print(f"Size of train set is {len(train_targets)}")
    print(f"Size of validation set is {len(val_targets)}")
    minibatch_size = int(input("Choose a batch size:   "))
    train_dataloader = get_dataloader(train_inputs, train_targets, minibatch_size)
    val_dataloader = get_dataloader(val_inputs, val_targets, minibatch_size)

    ##################################### Model parameters
    len_vocabulary = 4 ** kmer_size + NUM_SPECIAL_CHARS
    model_parameters = input(
        f"Please enter the embedding_size, rnn_hidden_size, encoder_rnn_num_layers, decoder_rnn_num_layers:   "
    )
    embedding_size, rnn_hidden_size, encoder_rnn_num_layers, decoder_rnn_num_layers, = (
        int(parameter) for parameter in list(model_parameters.split(","))
    )
    dropouts = input("Please enter the dropout and embedded_dropout:   ")
    dropout, embedded_dropout = (float(d) for d in list(dropouts.split(",")))

    model = create_model(
        vocab_size=len_vocabulary,
        embedding_size=embedding_size,
        hidden_size=rnn_hidden_size,
        encoder_num_layers=encoder_rnn_num_layers,
        decoder_num_layers=decoder_rnn_num_layers,
        dropout=dropout,
        emb_dropout=embedded_dropout,
    )

    ##################################### Training parameters
    number_of_epochs = int(input("Choose number of epochs to train the model :   "))
    learning_rate = float(input("Choose the learning rate :   "))
    training_metrics, last_best_epoch = train_loop(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        weights=weights,
        num_epochs=number_of_epochs,
        learning_rate=learning_rate,
    )

    ##################################### Save Training details
    training_parameters_performance_dict = {
        "vocab_size": len_vocabulary,
        "embedding_size": embedding_size,
        "hidden_size": rnn_hidden_size,
        "encoder_num_layers": encoder_rnn_num_layers,
        "decoder_num_layers": decoder_rnn_num_layers,
        "dropout": dropout,
        "emb_dropout": embedded_dropout,
        "training_metrics": training_metrics,
        "kmer_size": kmer_size,
        "last_best_epoch": last_best_epoch,
        "initial_learning_rate": learning_rate,
    }
    training_parameters_performance_dict_path = f"{CURRENT_RUN_DIR}/training_parameters_performance.json"
    with open(training_parameters_performance_dict_path, "w") as f:
        dump(training_parameters_performance_dict, f)
