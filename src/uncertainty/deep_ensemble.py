import torch
import numpy as np


def predict_ensemble(models, x, device):
    """
    Run multiple models and return mean + member standard deviation.

    The std uses the unbiased sample convention (ddof=1, denominator N-1),
    matching the manuscript equation. Historical results were produced with
    ddof=0 (factor sqrt((N-1)/N) ~= 0.894 smaller for N=5).
    """
    if len(models) < 2:
        raise ValueError("An ensemble needs at least 2 members for a spread estimate")
    preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(model(x.to(device)).cpu().numpy())

    preds = np.stack(preds, axis=0)  # [N, B, T, D]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0, ddof=1)

    return mean, std
