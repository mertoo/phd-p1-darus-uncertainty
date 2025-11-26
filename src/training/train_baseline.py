import os
import yaml
import argparse
import torch
import torch.nn as nn

# Dataset utilities
from src.data_loading.darus_dataset import create_dataloaders

# Models
from src.models.lstm import LSTMSeq2Seq
from src.models.gru import GRUSeq2Seq
from src.models.tcn import TCN
from src.models.mlp import MLP
from src.models.linear import LinearBaseline
from src.models.naive import NaiveBaseline


# ============================================================
#  TRAINING UTILITIES
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Yhat = model(X)
        loss = criterion(Yhat, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            loss = criterion(Yhat, Y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
#  MAIN TRAINING SCRIPT
# ============================================================

def main():
    print("🔧 Training script started!")

    # -------------------------------
    # 1. Parse CLI arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # -------------------------------
    # 2. Load YAML config
    # -------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract data parameters
    data_cfg = config["data"]
    history = data_cfg["history"]
    horizon = data_cfg["horizon"]
    batch_size = data_cfg["batch_size"]

    # Extract model parameters
    model_cfg = config["model"]
    model_name = model_cfg["type"].lower()
    hidden_dim = model_cfg.get("hidden_dim", 128)
    num_layers = model_cfg.get("num_layers", 2)
    dropout = float(model_cfg.get("dropout", 0.0))  # cast to float

    # Training parameters
    train_cfg = config["training"]
    epochs = train_cfg["epochs"]
    learning_rate = train_cfg["lr"]
    device = train_cfg["device"]

    # Logging/save directory
    log_cfg = config["logging"]
    save_dir = os.path.join(log_cfg["save_dir"], log_cfg["run_name"])
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.pt")

    # -------------------------------
    # 3. Load dataset
    # -------------------------------
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    input_dim = train_loader.dataset.X.shape[-1]
    output_dim = train_loader.dataset.Y.shape[-1]

    # -------------------------------
    # 4. Model selection
    # -------------------------------
    if model_name == "lstm":
        model = LSTMSeq2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            target_dim=output_dim,
        )
        print("Model: LSTM")

    elif model_name == "gru":
        model = GRUSeq2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            target_dim=output_dim,
        )
        print("Model: GRU")

    elif model_name == "tcn":
        model = TCN(
            input_dim=input_dim,
            target_dim=output_dim,
            history=history,
            horizon=horizon,
        )
        print("Model: TCN")

    elif model_name == "mlp":
        model = MLP(
            input_dim=input_dim,
            history=history,
            horizon=horizon,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )
        print("Model: MLP")

    elif model_name == "linear":
        model = LinearBaseline(
            input_dim=input_dim,
            history=history,
            output_dim=output_dim,
            horizon=horizon,
        )
        print("Model: Linear Baseline")

    elif model_name == "naive":
        model = NaiveBaseline(
            output_dim=output_dim,
            horizon=horizon
        )
        print("Model: Naive Baseline (no training)")

        # Save immediately and exit
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        print(f"Saved naive baseline to {ckpt_path}")
        return



    else:
        raise ValueError(f"Unknown model: {model_name}")

    # -------------------------------
    # 5. Setup optimizer & device
    # -------------------------------
    if model_name == "naive":
        return  # prevents optimizer creation

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Using device: {device}")

    # -------------------------------
    # 6. Training loop
    # -------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            print(f"✓ Saved new best model to {ckpt_path}")

    print(f"Training complete. Best val loss: {best_val_loss}")


if __name__ == "__main__":
    main()
