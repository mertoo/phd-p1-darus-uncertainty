import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Data utilities
from src.data_loading.darus_dataset import create_dataloaders, FEATURE_COLUMNS, OUTPUT_COLUMNS

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
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    args = parser.parse_args()

    # -------------------------------
    # 2. Load config
    # -------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"].lower()
    history = config["history"]
    horizon = config["horizon"]
    hidden_dim = config.get("hidden_dim", 128)
    num_layers = config.get("num_layers", 2)
    batch_size = config.get("batch_size", 256)
    learning_rate = config.get("learning_rate", 1e-3)
    num_epochs = config.get("num_epochs", 20)

    # -------------------------------
    # 3. Load dataset
    # -------------------------------
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    input_dim = train_loader.dataset.X.shape[-1]   # e.g. 11 features
    output_dim = train_loader.dataset.Y.shape[-1]  # 5 DoFs

    # Prepare save directory
    save_dir = f"experiments/results/p1_{model_name}_baseline"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.pt")

    # -------------------------------
    # 4. Model selection
    # -------------------------------
    if model_name == "lstm":
        model = LSTMSeq2Seq(input_dim, hidden_dim, num_layers, output_dim, horizon)
        print("Model: LSTM")

    elif model_name == "gru":
        model = GRUSeq2Seq(input_dim, hidden_dim, num_layers, output_dim, horizon)
        print("Model: GRU")

    elif model_name == "tcn":
        model = TCN(input_dim, output_dim, horizon)
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
    # 5. Setup training components
    # -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Using device: {device}")

        # -------------------------------
    # 6. Training loop
    # -------------------------------
    best_val_loss = float("inf")

    print(f"Using device: {device}")

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict()},
                ckpt_path
            )
            print(f"✓ Saved new best model to {ckpt_path}")

    print(f"Training complete. Best val loss: {best_val_loss}")
if __name__ == "__main__":
    main()