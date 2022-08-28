from json import load
import torch
from data_operations.get_dataloaders import get_dataloader
from data_operations.tokenize_data import Tokenize
from inference.test_model import test_model
from visualization.plot_mutations_comparision_graphs import plot_mutations_comparision_graphs
from model_components.create_model import create_model
from settings.constants import (
    CURRENT_RUN_DATA_DIR,
    CURRENT_RUN_DIR,
    SAVED_MODELS_PATH,
)
from visualization.plot_mutation_sites import get_mutations_and_plot


def inference():
    dataset_file_path = f"{CURRENT_RUN_DATA_DIR}/test.json"
    print(f"Path to dataset : {dataset_file_path}")

    training_parameters_performance_dict_path = f"{CURRENT_RUN_DIR}/training_parameters_performance.json"
    training_parameters_performance_dict = load(open(training_parameters_performance_dict_path))
    model = create_model(
        vocab_size=training_parameters_performance_dict["vocab_size"],
        embedding_size=training_parameters_performance_dict["embedding_size"],
        hidden_size=training_parameters_performance_dict["hidden_size"],
        encoder_num_layers=training_parameters_performance_dict["encoder_num_layers"],
        decoder_num_layers=training_parameters_performance_dict["decoder_num_layers"],
        dropout=training_parameters_performance_dict["dropout"],
        emb_dropout=training_parameters_performance_dict["emb_dropout"],
    )
    checkpoint = torch.load(
        f"{SAVED_MODELS_PATH}/save_epoch_{training_parameters_performance_dict['last_best_epoch']}.pt"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    kmer_size = training_parameters_performance_dict["kmer_size"]

    tokenizer = Tokenize(kmer_length=kmer_size)
    test_inputs, test_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="test")
    test_dataloader = get_dataloader(test_inputs, test_targets, batch_size=1)
    predicted_sequences, alphas = test_model(test_dataloader, model, kmer_size)
    data_parameters = load(open(f"{CURRENT_RUN_DIR}/data_parameters.json"))
    get_mutations_and_plot(
        targets=predicted_sequences,
        sequence_start_postion=data_parameters["sequence_start_postion"],
        sequence_end_postion=data_parameters["sequence_end_postion"],
        seq_len=data_parameters["max_seq_length"],
        y_type="test_predicted",
    )
    # plot_mutations_comparision_graphs()
