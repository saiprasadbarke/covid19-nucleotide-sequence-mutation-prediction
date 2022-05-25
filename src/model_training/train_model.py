# Local

from math import inf
from helpers.check_dir_exists import check_dir_exists
from model_components.model import EncoderDecoder
from settings.constants import EARLY_STOPPING_THRESHOLD, LEARNING_RATE_ALPHA, N_EPOCHS, RUN_NAME, SAVED_MODELS_PATH
from model_training.run_epoch import run_epoch
from model_training.loss_computation import SimpleLossCompute
from model_training.batch import rebatch

# External

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

# from timeit import default_timer as timer


def train_loop(
    model: EncoderDecoder,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = N_EPOCHS,
    learning_rate: float = LEARNING_RATE_ALPHA,
    print_every=100,
):

    criterion = nn.NLLLoss(reduction="sum")
    optim = Adam(model.parameters(), lr=learning_rate)
    number_of_epochs_without_improvement = 0
    best_val_loss = inf
    losses = {"training": [], "validation": []}

    for epoch in range(num_epochs):

        print("Epoch", epoch)
        model.train()
        with torch.set_grad_enabled(True):
            train_pp = run_epoch(
                (rebatch(b) for b in train_dataloader),
                model,
                SimpleLossCompute(model.generator, criterion, optim),
                print_every=print_every,
            )
            print(f"Train perplexity: {train_pp}")
            losses["training"].append(train_pp)
        model.eval()
        with torch.no_grad():
            validation_pp = run_epoch(
                (rebatch(b) for b in validation_dataloader),
                model,
                SimpleLossCompute(model.generator, criterion, None),
            )
            print(f"Validation perplexity: {validation_pp}")
            losses["validation"].append(validation_pp)

        # Early stopping
        if losses["validation"][-1] < best_val_loss:
            check_dir_exists(f"{SAVED_MODELS_PATH}/{RUN_NAME}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "loss": losses,
                },
                f"{SAVED_MODELS_PATH}/{RUN_NAME}/MODEL_{epoch}.pt",
            )
            best_val_loss = losses["validation"][-1]
            number_of_epochs_without_improvement = 0
        else:
            if number_of_epochs_without_improvement == EARLY_STOPPING_THRESHOLD:
                print(f"Early stopping on epoch number {epoch}!")
                break
            else:
                number_of_epochs_without_improvement += 1
