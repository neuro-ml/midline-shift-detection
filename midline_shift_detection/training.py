import random

import torch
from torch.nn import functional
import numpy as np
from dpipe.batch_iter import combine_batches, pad_batch_equal
from dpipe.torch import optimizer_step, to_np, sequence_to_var


def train_step(inputs, targets, model, optimizer):
    model.train()

    # move the tensors to the same device as `model`
    inputs, targets = sequence_to_var(inputs, targets, device=model)
    prediction = model(inputs)

    # the model has 2 `heads`
    assert prediction.shape[1] == 2, prediction.shape
    curves, limits = prediction[:, 0], prediction[:, 1]

    limits_mask = ~torch.isnan(targets)
    loss = functional.binary_cross_entropy_with_logits(limits, limits_mask.to(dtype=limits.dtype))

    if limits_mask.any():
        # penalize only the predictions where the target is defined
        loss = loss + functional.mse_loss(curves[limits_mask], targets[limits_mask])

    optimizer_step(optimizer, loss)
    return to_np(loss)  # convert back to numpy


def get_random_slice(x, y):
    # choose a random annotation
    y = random.choice(y)
    # choose a random slice
    idx = random.randrange(x.shape[-1])
    return x[None, ..., idx], y[..., idx]


def random_flip(x, y):
    # simple augmentation by random flips along the Ox axis
    if random.randint(0, 1):
        x = np.flip(x, axis=-1)
        y = x.shape[-1] - y

    return x, y


def combiner(batches):
    images, contours = combine_batches(batches)
    # the images might be of different shapes, so we need to pad them before collating
    return pad_batch_equal(images, np.min, 0), pad_batch_equal(contours, np.nan, 0)
