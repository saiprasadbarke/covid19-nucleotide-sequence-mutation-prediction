# Local
from model_components.model import EncoderDecoder

# External
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    train_dataloader: DataLoader,
    model: EncoderDecoder,
    optimizer,
    scheduler,
    criterion,
    device,
):
    batch_losses = []
    training_loss = 0.0
    # training loop
    for data in train_dataloader:
        source_weights, target_embedded_equation = data
        source_weights, target_embedded_equation = (
            source_weights.to(device),
            target_embedded_equation.to(device),
        )
        target_embedded_equation_input = target_embedded_equation[:-1, :]
        tgt_mask, tgt_padding_mask = create_mask(target_embedded_equation_input)
        optimizer.zero_grad()
        model.train()
        mlp_transformerdecoder_logits = model(
            source_weights,
            target_embedded_equation_input,
            tgt_mask,
            tgt_padding_mask,
        )
        target_embedded_equation_out = target_embedded_equation[1:, :].reshape(-1)
        mlp_transformerdecoder_logits_out = mlp_transformerdecoder_logits.reshape(
            -1, mlp_transformerdecoder_logits.shape[-1]
        )
        loss = criterion(
            mlp_transformerdecoder_logits_out,
            target_embedded_equation_out,
        )
        loss.backward()
        batch_losses.append(loss.item())
        optimizer.step()
    training_loss = np.mean(batch_losses)
    scheduler.step()
    return training_loss


def eval_model(
    validation_dataloader: DataLoader,
    model: EncoderDecoder,
    criterion,
    device,
):

    val_losses = []
    validation_loss = 0.0
    for data in validation_dataloader:
        source_weights, target_embedded_equation = data
        source_weights, target_embedded_equation = (
            source_weights.to(device),
            target_embedded_equation.to(device),
        )
        target_embedded_equation_input = target_embedded_equation[:-1, :]
        tgt_mask, tgt_padding_mask = create_mask(target_embedded_equation_input)
        model.eval()
        mlp_transformerdecoder_logits = model(
            source_weights,
            target_embedded_equation_input,
            tgt_mask,
            tgt_padding_mask,
        )
        target_embedded_equation_out = target_embedded_equation[1:, :].reshape(-1)
        mlp_transformerdecoder_logits_out = mlp_transformerdecoder_logits.reshape(
            -1, mlp_transformerdecoder_logits.shape[-1]
        )
        loss = criterion(
            mlp_transformerdecoder_logits_out,
            target_embedded_equation_out,
        )
        val_losses.append(loss.item())
    validation_loss = np.mean(val_losses)
    return validation_loss
