# Local

from json import load
from math import inf
from model_components.model import EncoderDecoder
from model_training.sequence_weighted_cross_entropy_loss import SequenceWeightedCELoss
from settings.constants import (
    CURRENT_RUN_DIR,
    EARLY_STOPPING_THRESHOLD,
    SAVED_MODELS_PATH,
    SAVED_STATS_PATH,
    USE_CUDA,
)
from model_training.run_epoch import run_epoch
from model_training.loss_computation import SimpleLossCompute
from model_training.batch import rebatch

# External

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch


def train_loop(
    model: EncoderDecoder,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    print_every=100,
):
    weights = get_weights()
    criterion = SequenceWeightedCELoss(weights=weights)
    optim = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optim, patience=2)

    number_of_epochs_without_improvement = 0
    best_val_loss = inf
    last_best_epoch = 0
    returned_metrics = {
        "training_loss": [],
        "validation_loss": [],
        "training_perplexity": [],
        "validation_perplexity": [],
        "learning_rate": [],
    }

    for epoch in range(num_epochs):
        training_loss = 0
        training_perplexity = 0
        validation_loss = 0
        validation_perplexity = 0
        print("Epoch", epoch)
        model.train()
        with torch.set_grad_enabled(True):
            training_loss, training_perplexity, epoch_learning_rate = run_epoch(
                (rebatch(b) for b in train_dataloader),
                model,
                SimpleLossCompute(model=model, criterion=criterion, optimizer=optim,),
                print_every=print_every,
            )
            print(f"Training loss: {training_loss}")
            print(f"Training perplexity: {training_perplexity}")
            returned_metrics["training_loss"].append(training_loss)
            returned_metrics["training_perplexity"].append(training_perplexity)
            returned_metrics["learning_rate"] += epoch_learning_rate
        model.eval()
        with torch.no_grad():
            validation_loss, validation_perplexity, _ = run_epoch(
                (rebatch(b) for b in validation_dataloader), model, SimpleLossCompute(model=model, criterion=criterion),
            )
            print(f"Validation loss: {validation_loss}")
            print(f"Validation perplexity: {validation_perplexity}")
            returned_metrics["validation_loss"].append(validation_loss)
            returned_metrics["validation_perplexity"].append(validation_perplexity)

        # Scheduler step
        scheduler.step(metrics=validation_loss)

        # Early stopping
        if validation_loss < best_val_loss:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                f"{SAVED_MODELS_PATH}/save_epoch_{epoch}.pt",
            )
            best_val_loss = validation_loss
            last_best_epoch = epoch
            number_of_epochs_without_improvement = 0
            if last_best_epoch == num_epochs:
                return returned_metrics, last_best_epoch
        else:
            if number_of_epochs_without_improvement == EARLY_STOPPING_THRESHOLD:
                print(f"Early stopping on epoch number {epoch}!")
                return returned_metrics, last_best_epoch
            else:
                number_of_epochs_without_improvement += 1


def get_weights():
    weights_data = load(open(f"{SAVED_STATS_PATH}/difference_indices_ref_overall_data.json"))
    number_of_sequence_pairs_in_dataset = load(open(f"{CURRENT_RUN_DIR}/data_parameters.json"))[
        "number_of_sequence_pairs"
    ]
    list_weights = [i / number_of_sequence_pairs_in_dataset for i in weights_data.values()]
    list_weights.append(1)
    weights_tensor = torch.Tensor(list_weights)
    reshaped_weights_tensor = torch.reshape(weights_tensor, (-1, len(weights_data) + 1))
    return reshaped_weights_tensor.cuda() if USE_CUDA else reshaped_weights_tensor
