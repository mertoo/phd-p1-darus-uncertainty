import torch
import numpy as np


def predict_ensemble(models, x, device):
    """
    Run multiple models and return mean + epistemic std.
    """
    preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(model(x.to(device)).cpu().numpy())

    preds = np.stack(preds, axis=0)  # [N, B, T, D]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    return mean, std
