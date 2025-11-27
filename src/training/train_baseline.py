import os
import yaml
import argparse
import torch
import torch.nn as nn

from src.data_loading.darus_dataset import create_dataloaders

from src.models.lstm import LSTMSeq2Seq
from src.models.gru import GRUSeq2Seq
from src.models.tcn import TCN
from src.models.mlp import MLP
from src.models.linear import LinearBaseline
from src.models.naive import NaiveBaseline


# ============================================================
# TRAIN / EVAL UTILS
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
# MAIN
# ============================================================

def main():
    print("🔧 Training script started!")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # ------------------------------------------------
    # LOAD CONFIG
    # ------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    model_name = config["model"]["type"].lower()
    hidden_dim = config["model"].get("hidden_dim", 128)
    num_layers = config["model"].get("num_layers", 2)
    dropout = float(config["model"].get("dropout", 0.0))

    num_epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["lr"])
    device = config["training"]["device"]

    # ------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    input_dim = train_loader.dataset.X.shape[-1]
    output_dim = train_loader.dataset.Y.shape[-1]

    # ------------------------------------------------
    # SAVE PATH (CRITICAL FIX FOR ENSEMBLES ✅)
    # ------------------------------------------------
    config_dir = os.path.dirname(args.config)
    ckpt_path = os.path.join(config_dir, "best_model.pt")
    print(f"📁 Model will be saved to: {ckpt_path}")

    # ------------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------------
    if model_name == "lstm":
        model = LSTMSeq2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            horizon=horizon,
            target_dim=output_dim,
            dropout=dropout
        )
        print("Model: LSTM")

    elif model_name == "gru":
        model = GRUSeq2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            horizon=horizon,
            target_dim=output_dim,
            dropout=dropout
        )
        print("Model: GRU")

    elif model_name == "tcn":
        model = TCN(
            input_dim=input_dim,
            target_dim=output_dim,
            history=history,
            horizon=horizon
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
            horizon=horizon
        )
        print("Model: Linear")

    elif model_name == "naive":
        model = NaiveBaseline(
            output_dim=output_dim,
            horizon=horizon
        )
        print("Model: Naive")

        torch.save({"model_state": model.state_dict()}, ckpt_path)
        print(f"✅ Naive baseline saved to {ckpt_path}")
        return

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # ------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Using device: {device}")

    # ------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            print(f"✅ Saved new best model to {ckpt_path}")

    print(f"✅ Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
