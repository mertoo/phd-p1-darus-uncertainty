import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_vs_truth(Y, Yhat, savepath):
    """Y, Yhat: (N, H, 5) arrays"""

    Y = Y[0]       # (H, 5)
    Yhat = Yhat[0] # (H, 5)

    H, D = Y.shape

    DoF_names = ["u", "v", "p", "r", "phi"]

    plt.figure(figsize=(12, 6))
    for d in range(D):
        plt.plot(Y[:, d], label=f"Truth {DoF_names[d]}")
        plt.plot(Yhat[:, d], "--", label=f"Pred {DoF_names[d]}")
    plt.xlabel("Forecast step")
    plt.ylabel("Value")
    plt.title("Prediction vs Truth")
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath)
    plt.close()


def plot_multi_horizon_rmse(Y, Yhat, savepath):
    """Compute RMSE for each horizon step"""
    Y = Y.mean(axis=2)      # collapse DoFs? (N,H,5)->(N,H)
    Yhat = Yhat.mean(axis=2)

    H = Y.shape[1]

    rmse = [np.sqrt(((Y[:, h] - Yhat[:, h])**2).mean()) for h in range(H)]

    plt.figure(figsize=(12, 5))
    plt.plot(rmse)
    plt.xlabel("Horizon step")
    plt.ylabel("RMSE")
    plt.title("RMSE per Forecast Horizon")
    plt.grid(True)
    plt.savefig(savepath)
    plt.close()
