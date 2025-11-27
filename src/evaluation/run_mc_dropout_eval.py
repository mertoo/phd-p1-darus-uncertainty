import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from src.uncertainty.mc_dropout import mc_dropout_predict
from src.training.train_baseline import create_dataloaders
from src.models.lstm import LSTMSeq2Seq


TARGET_NAMES = ["p", "phi", "r", "u", "v"]


def infer_config_from_model_dir(model_dir):
    """
    Infers YAML config name from model directory name.
    Example:
        p1_lstm_baseline -> p1_lstm_seq2seq.yaml
    """
    name = os.path.basename(model_dir)

    if "lstm" in name:
        return "p1_lstm_seq2seq.yaml"
    if "gru" in name:
        return "p1_gru_seq2seq.yaml"
    if "mlp" in name:
        return "p1_mlp.yaml"
    if "tcn" in name:
        return "p1_tcn.yaml"
    if "linear" in name:
        return "p1_linear.yaml"
    if "naive" in name:
        return "p1_naive.yaml"

    raise ValueError(f"Cannot infer config from model_dir: {model_dir}")


def plot_mc_with_uncertainty(mean, std, truth, save_dir):
    T, D = mean.shape
    t = np.arange(T)

    for i, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(7, 4))

        lower = mean[:, i] - 2 * std[:, i]
        upper = mean[:, i] + 2 * std[:, i]

        plt.fill_between(t, lower, upper, alpha=0.3, label="95% interval")
        plt.plot(t, mean[:, i], linewidth=2, label="MC Mean")
        plt.plot(t, truth[:, i], linestyle="--", linewidth=2, label="Ground Truth")

        plt.xlabel("Prediction step")
        plt.ylabel(name)
        plt.title(f"MC Dropout Prediction with Uncertainty – {name}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mc_{name}.png"), dpi=300)
        plt.close()


def plot_epistemic_growth(std, save_dir):
    T, D = std.shape
    t = np.arange(T)

    for i, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(6, 4))
        plt.plot(t, std[:, i], linewidth=2)
        plt.xlabel("Prediction step")
        plt.ylabel("Standard deviation")
        plt.yscale("log")
        plt.title(f"Epistemic Uncertainty Growth – {name}")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mc_{name}_std.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    args = parser.parse_args()

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ✅ FIX: infer config from model_dir
    config_name = infer_config_from_model_dir(args.model_dir)
    config_path = f"experiments/configs/{config_name}"
    print("Using config:", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]

    # ✅ Hard guarantees from DaRUS
    input_dim = 12
    target_dim = 5

    model = LSTMSeq2Seq(
        input_dim=input_dim,
        target_dim=target_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        horizon=horizon
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.train()  # ✅ ACTIVATE DROPOUT FOR MC

    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]


    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
    history, horizon, batch_size
    )

    loader = ood_loader if args.split == "ood" else test_loader


    x, y = next(iter(loader))
    x = x[:1]
    y = y[:1]

    mean, std, _ = mc_dropout_predict(model, x, n_samples=args.n_samples)
    truth = y[0].numpy()

    save_dir = os.path.join(args.model_dir, "mc_dropout_plots", args.split)
    os.makedirs(save_dir, exist_ok=True)

    plot_mc_with_uncertainty(mean, std, truth, save_dir)
    plot_epistemic_growth(std, save_dir)

    print("✅ MC Dropout evaluation complete.")
    print("Plots saved to:", save_dir)


if __name__ == "__main__":
    main()
