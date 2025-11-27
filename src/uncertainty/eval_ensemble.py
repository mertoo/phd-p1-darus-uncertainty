import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

from src.uncertainty.deep_ensemble import predict_ensemble
from src.data_loading.darus_dataset import create_dataloaders
from src.models.lstm import LSTMSeq2Seq

# Must match P1 ordering exactly
TARGET_NAMES = ["u", "v", "p", "r", "phi"]


def load_ensemble(model_dir, config, device):
    models = []

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]

    for sub in sorted(os.listdir(model_dir)):
        ckpt_path = os.path.join(model_dir, sub, "best_model.pt")

        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skipping missing: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)

        # ✅ THIS IS THE CRITICAL FIX:
        state_dict = ckpt["model_state"]

        model = LSTMSeq2Seq(
            input_dim=config["data"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            horizon=horizon,
            target_dim=config["data"]["output_dim"],
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    print(f"✅ Loaded {len(models)} ensemble members")
    return models



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    # Correct P1 loader interface
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    loader = test_loader if args.split == "test" else ood_loader

    models = load_ensemble(args.ensemble_dir, config, device)

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.numpy()

    mean, std = predict_ensemble(models, x, device)

    out_dir = os.path.join(
        args.ensemble_dir, "plots", args.split
    )
    os.makedirs(out_dir, exist_ok=True)

    horizon = mean.shape[1]

    for i, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(10, 5))

        plt.plot(mean[0, :, i], label="Ensemble Mean", color="black")
        plt.fill_between(
            range(horizon),
            mean[0, :, i] - 2 * std[0, :, i],
            mean[0, :, i] + 2 * std[0, :, i],
            alpha=0.3,
            label="95% Interval",
        )
        plt.plot(y[0, :, i], "--", label="Ground Truth", linewidth=2)

        plt.xlabel("Prediction Step")
        plt.ylabel(name)
        plt.title(f"Deep Ensemble Prediction – {name} ({args.split.upper()})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, f"ensemble_{name}_{args.split}.png"),
            dpi=300
        )
        plt.close()

    print(f"✅ Ensemble plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
