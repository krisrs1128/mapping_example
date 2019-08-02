#!/usr/bin/env python
"""
Training Utilities
"""
import torch.nn as nn
import torch

def train(model, iterator, optimizer, loss_fun, device):
    """
    Train Model for an Epoch

    :param model: The model to update, a nn.Module class.
    :param iterator: A torch data iterator, from which training examples will
      be drawn.
    :param optimizer: The object that updates weights over iterations.
    :param loss_fun: A function that computes a loss, to guide weight updates.
    :param device: Either the CPU or GPU device on which to put the model when
      making updates.
    :return: A tuple with the following elements,
      - model: The model, with updated parameters.
      - epoch_loss: The average loss over the epoch
    """
    epoch_loss = 0
    model.train()

    for x, _, y in iterator:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = loss_fun(y_hat[:, 1], y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)


def evaluate(model, iterator, loss_fun, device):
    epoch_loss = 0
    model.eval()

    for x, _, y in iterator:
        with torch.no_grad():
            y_hat = model(x.to(device))
            loss = loss_fun(y_hat[:, 1], y.to(device))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
