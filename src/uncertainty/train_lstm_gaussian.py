import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.lstm_gaussian import LSTMGaussianSeq2Seq
from src.data_loading.darus_dataset import create_dataloaders


def _get_batch_size(config: dict) -> int:
    # Prefer training.batch_size (matches your working baseline configs)
    if "training" in config and "batch_size" in config["training"]:
        return int(config["training"]["batch_size"])
    # Fallback (older experiments)
    if "data" in config and "batch_size" in config["data"]:
        return int(config["data"]["batch_size"])
    raise KeyError("batch_size not found in config['training'] or config['data'].")


def train_lstm_gaussian(config: dict):
    device = config.get("training", {}).get("device", "cpu")
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    horizon = int(config["data"]["horizon"])
    batch_size = _get_batch_size(config)

    print("🔧 create_dataloaders called!")
    train_loader, val_loader, _, _ = create_dataloaders(config, horizon, batch_size)

    # Infer input_dim from one batch
    x_example, _ = next(iter(train_loader))
    input_dim = x_example.shape[-1]
    target_dim = 5

    hidden_dim = int(config["model"]["hidden_dim"])
    num_layers = int(config["model"]["num_layers"])
    dropout = float(config["model"].get("dropout", 0.1))

    model = LSTMGaussian(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        target_dim=config["model"]["target_dim"],
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.0),
    ).to(device)


    lr = float(config["training"]["lr"])
    epochs = int(config["training"]["epochs"])

    optimizer = Adam(model.parameters(), lr=lr)
    nll_loss = nn.GaussianNLLLoss(reduction="mean")

    save_root = config["logging"]["save_dir"]
    run_name = config["logging"]["run_name"]
    model_dir = os.path.join(save_root, run_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    print(f"📁 Model will be saved to: {ckpt_path}")
    print(f"Using device: {device}")

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            mu, logvar = model(x)
            var = torch.exp(logvar).clamp(min=1e-6)

            loss = nll_loss(mu, y, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(1, len(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                mu, logvar = model(x)
                var = torch.exp(logvar).clamp(min=1e-6)

                loss = nll_loss(mu, y, var)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / max(1, len(val_losses))

        print(
            f"Epoch {epoch}/{epochs} | Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_nll": best_val,
                },
                ckpt_path,
            )
            print(f"✅ Saved new best model to {ckpt_path}")

    print(f"✅ Training complete. Best val NLL: {best_val:.6f}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_lstm_gaussian(config)


if __name__ == "__main__":
    main()
