# Local

from helpers.check_dir_exists import check_dir_exists
from model_components.model import EncoderDecoder
from settings.constants import USE_CUDA, PAD_IDX
from model_training.run_epoch import run_epoch
from model_training.loss_computation import SimpleLossCompute
from model_training.batch import rebatch

# External

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from timeit import default_timer as timer
from argparse import ArgumentParser


def train_loop(
    model: EncoderDecoder,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.0003,
    print_every=100,
):

    if USE_CUDA:
        model.cuda()

    criterion = nn.NLLLoss(reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dev_perplexities = []

    for epoch in range(num_epochs):

        print("Epoch", epoch)
        model.train()
        with torch.set_grad_enabled(True):
            train_perplexity = run_epoch(
                ((rebatch(PAD_IDX, b) for b in train_dataloader)),
                model,
                SimpleLossCompute(model.generator, criterion, optim),
                print_every=print_every,
            )
            print(f"Train perplexity: {train_perplexity}")
        model.eval()
        with torch.no_grad():

            dev_perplexity = run_epoch(
                ((rebatch(PAD_IDX, b) for b in validation_dataloader)),
                model,
                SimpleLossCompute(model.generator, criterion, None),
            )
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)

    return dev_perplexities
