import torch
import numpy as np


def enable_mc_dropout(model):
    """
    Enable dropout layers during inference for MC Dropout.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, x, n_samples=200):
    """
    Perform MC Dropout forward passes.

    Args:
        model: Trained PyTorch model with dropout
        x: Input tensor (batch=1)
        n_samples: Number of stochastic forward passes

    Returns:
        mean: (H, D)
        std:  (H, D)
        all_samples: (N, H, D)
    """
    model.train()  # IMPORTANT: FORCE DROPOUT ON

    preds = []
    for _ in range(n_samples):
        out = model(x)
        preds.append(out.cpu().numpy())

    preds = np.stack(preds, axis=0)

    mean = preds.mean(axis=0).squeeze(0)
    std = preds.std(axis=0).squeeze(0)

    return mean, std, preds
