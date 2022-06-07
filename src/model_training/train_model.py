# Local

from math import inf

from sklearn.model_selection import learning_curve
from helpers.check_dir_exists import check_dir_exists
from model_components.model import EncoderDecoder
from settings.constants import (
    EARLY_STOPPING_THRESHOLD,
    SAVED_MODELS_PATH,
    SAVED_TENSORBOARD_LOGS_PATH,
)
from model_training.run_epoch import run_epoch
from model_training.loss_computation import SimpleLossCompute
from model_training.batch import rebatch

# External

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch

# from torch.utils.tensorboard import SummaryWriter

# from timeit import default_timer as timer


def train_loop(
    model: EncoderDecoder,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    print_every=100,
):

    criterion = nn.NLLLoss(reduction="sum")
    optim = Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode="min", patience=2)
    # check_dir_exists(SAVED_TENSORBOARD_LOGS_PATH)
    learning_curve = []
    # tb_writer = SummaryWriter(log_dir=SAVED_TENSORBOARD_LOGS_PATH)
    number_of_epochs_without_improvement = 0
    best_val_loss = inf
    returned_metrics = {
        "training_loss": [],
        "validation_loss": [],
        "training_perplexity": [],
        "validation_perplexity": [],
        "learning_rate": [],
    }

    for epoch in range(num_epochs):

        print("Epoch", epoch)
        model.train()
        model.zero_grad()
        with torch.set_grad_enabled(True):
            training_loss, training_perplexity, epoch_learning_rate = run_epoch(
                (rebatch(b) for b in train_dataloader),
                model,
                SimpleLossCompute(model.generator, criterion, optim),
                print_every=print_every,
            )
            print(f"Training loss: {training_loss}")
            print(f"Training perplexity: {training_perplexity}")
            # tb_writer.add_scalar("train/train_epoch_loss", training_loss, epoch)
            # tb_writer.add_scalar("train/train_epoch_perplexity", training_perplexity, epoch)
            # tb_writer.add_scalar("learning_rate", optim.param_groups[0]["lr"], epoch)
            returned_metrics["training_loss"].append(training_loss)
            returned_metrics["training_perplexity"].append(training_perplexity)
            returned_metrics["learning_rate"] += epoch_learning_rate
        model.eval()
        with torch.no_grad():
            validation_loss, validation_perplexity, _ = run_epoch(
                (rebatch(b) for b in validation_dataloader), model, SimpleLossCompute(model.generator, criterion),
            )
            print(f"Validation loss: {validation_loss}")
            print(f"Validation perplexity: {validation_perplexity}")
            # tb_writer.add_scalar("validation/validation_epoch_loss", validation_loss, epoch)
            # tb_writer.add_scalar("validation/validation_epoch_perplexity", validation_perplexity, epoch)
            returned_metrics["validation_loss"].append(validation_loss)
            returned_metrics["validation_perplexity"].append(validation_perplexity)
            # scheduler.step(metrics=validation_loss)
            # Early stopping
            if validation_loss < best_val_loss:
                check_dir_exists(SAVED_MODELS_PATH)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        # "scheduler_state_dict": scheduler.state_dict(),
                        "returned_metrics": returned_metrics,
                    },
                    f"{SAVED_MODELS_PATH}/MODEL_{epoch}.pt",
                )
                best_val_loss = validation_loss
                number_of_epochs_without_improvement = 0
            else:
                if number_of_epochs_without_improvement == EARLY_STOPPING_THRESHOLD:
                    print(f"Early stopping on epoch number {epoch}!")
                    break
                else:
                    number_of_epochs_without_improvement += 1
