import torch
import numpy as np


def compute_rmse(y_true, y_pred):
    """
    y_true, y_pred: (N, horizon, target_dim)
    returns RMSE per target dim
    """
    mse = ((y_true - y_pred) ** 2).mean(axis=(0, 1))
    rmse = np.sqrt(mse)
    return rmse


def compute_rmse_per_timestep(y_true, y_pred):
    """
    returns array of shape (horizon,)
    RMSE for each forecast step
    """
    mse = ((y_true - y_pred) ** 2).mean(axis=0).mean(axis=1)
    return np.sqrt(mse)


def evaluate_model(model, loader, device):
    """
    Runs the entire loader and collects predictions + truths.
    """
    model.eval()
    Ys = []
    Y_preds = []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = model(X)

            Ys.append(Y.cpu().numpy())
            Y_preds.append(Y_hat.cpu().numpy())

    Y = np.concatenate(Ys)
    Y_hat = np.concatenate(Y_preds)
    return Y, Y_hat
