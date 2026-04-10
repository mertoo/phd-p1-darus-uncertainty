import os
import yaml
import torch
import numpy as np
import torch.serialization as serialization

from src.data_loading.darus_dataset import create_dataloaders
from src.evaluation.plotting import (
    plot_prediction_vs_truth,
    plot_multi_horizon_rmse
)

# Allow numpy reconstruct for PyTorch 2.6
serialization.add_safe_globals([np.core.multiarray._reconstruct])


def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            Yhat = model(X)
            preds.append(Yhat.cpu())
            trues.append(Y.cpu())
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Detect model type from folder name
    model_name = os.path.basename(model_dir)
    print(f"Detected model: {model_name}")

    # Manual mapping
    MODEL_CONFIG_MAP = {
        "p1_lstm_baseline": "p1_lstm_seq2seq.yaml",
        "p1_gru_seq2seq_baseline": "p1_gru_seq2seq.yaml",
        "p1_tcn_baseline": "p1_tcn.yaml",
        "p1_mlp_baseline": "p1_mlp.yaml",
        "p1_linear_baseline": "p1_linear.yaml",
        "p1_naive_baseline": "p1_naive.yaml",
    }

    if model_name not in MODEL_CONFIG_MAP:
        raise ValueError(f"Unknown model folder: {model_name}")

    config_path = os.path.join("experiments/configs", MODEL_CONFIG_MAP[model_name])
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]

    history = data_cfg["history"]
    horizon = data_cfg["horizon"]

    # ---------------------------
    # Load dataloaders
    # ---------------------------
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=256,
    )

    # Extract dimensions
    input_dim = train_loader.dataset.X.shape[-1]
    target_dim = train_loader.dataset.Y.shape[-1]

    # ---------------------------
    # Build Model
    # ---------------------------
    mtype = model_cfg["type"].lower()

    if mtype == "lstm":
        from src.models.lstm import LSTMSeq2Seq
        model = LSTMSeq2Seq(
            input_dim=input_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            horizon=horizon,
            target_dim=target_dim,
        )

    elif mtype == "gru":
        from src.models.gru import GRUSeq2Seq
        model = GRUSeq2Seq(
            input_dim=input_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            horizon=horizon,
            target_dim=target_dim,
        )

    elif mtype == "tcn":
        from src.models.tcn import TCN
        model = TCN(
            input_dim=input_dim,
            target_dim=target_dim,
            history=history,
            horizon=horizon,
            num_channels=model_cfg.get("num_channels", [64, 64, 64]),
            kernel_size=model_cfg.get("kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )

    elif mtype == "mlp":
        from src.models.mlp import MLP
        model = MLP(
            input_dim=input_dim,
            history=history,
            horizon=horizon,
            output_dim=target_dim,
            hidden_dim=model_cfg["hidden_dim"],
        )

    elif mtype == "linear":
        from src.models.linear import LinearBaseline
        model = LinearBaseline(
            input_dim=input_dim,
            history=history,
            output_dim=target_dim,
            horizon=horizon,
        )

    elif mtype == "naive":
        from src.models.naive import NaiveBaseline
        model = NaiveBaseline(
            output_dim=target_dim,
            horizon=horizon,
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # Load weights
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---------------------------
    # Evaluate
    # ---------------------------
    print("Evaluating on test set...")
    Y_test, Yhat_test = evaluate_model(model, test_loader)

    print("Evaluating on OOD set...")
    Y_ood, Yhat_ood = evaluate_model(model, ood_loader)

    # ---------------------------
    # Save plots
    # ---------------------------
    plot_dir = os.path.join(model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_prediction_vs_truth(Y_test, Yhat_test,
        savepath=os.path.join(plot_dir, "pred_vs_truth_test.png"))

    plot_prediction_vs_truth(Y_ood, Yhat_ood,
        savepath=os.path.join(plot_dir, "pred_vs_truth_ood.png"))

    plot_multi_horizon_rmse(Y_test, Yhat_test,
        savepath=os.path.join(plot_dir, "rmse_test.png"))

    plot_multi_horizon_rmse(Y_ood, Yhat_ood,
        savepath=os.path.join(plot_dir, "rmse_ood.png"))


    # ---------------------------
    # Print summary
    # ---------------------------
    rmse_test = np.sqrt(((Y_test - Yhat_test)**2).mean())
    rmse_ood = np.sqrt(((Y_ood - Yhat_ood)**2).mean())

    print("\n=== SUMMARY ===")
    print(f"Test RMSE: {rmse_test:.6f}")
    print(f"OOD RMSE:  {rmse_ood:.6f}")
    print(f"Plots saved to: {plot_dir}")
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
