import os
import torch
from src.data_loading.darus_dataset import create_dataloaders
from src.evaluation.plotting import plot_prediction_vs_truth, plot_multi_horizon_rmse


def evaluate_model(model, dataloader, device):
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

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    
    import numpy as np
    import torch.serialization as serialization

    # Allow numpy reconstruct to avoid PyTorch safety restrictions
    

    # Load full checkpoint (old style saving)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)


    config = ckpt["config"]
    meta = ckpt["meta"]

    model_cfg = config["model"]
    model_type = model_cfg["type"]

    input_dim = meta["input_dim"]
    target_dim = meta["target_dim"]
    history = meta["history"]
    horizon = meta["horizon"]

    # Build correct model
    if model_type == "lstm_seq2seq":
        from src.models.lstm import LSTMSeq2Seq
        model = LSTMSeq2Seq(
            input_dim=input_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            horizon=horizon,
            target_dim=target_dim,
        )
    elif model_type == "gru_seq2seq":
        from src.models.gru import GRUSeq2Seq
        model = GRUSeq2Seq(
            input_dim=input_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            horizon=horizon,
            target_dim=target_dim,
        )
    elif model_type == "tcn":
        from src.models.tcn import TCN
        model = TCN(
            input_dim=input_dim,
            target_dim=target_dim,
            history=history,
            horizon=horizon,
            num_channels=model_cfg["num_channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test & OOD loaders
    _, _, test_loader, ood_loader = create_dataloaders(
        history=history, horizon=horizon, batch_size=256
    )

    # Evaluate
    Y_test, Yhat_test = evaluate_model(model, test_loader, "cpu")
    Y_ood, Yhat_ood = evaluate_model(model, ood_loader, "cpu")

    # Save plots
    plot_dir = os.path.join(args.model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_prediction_vs_truth(Y_test, Yhat_test,
        savepath=os.path.join(plot_dir, "pred_vs_truth_test.png"))

    plot_prediction_vs_truth(Y_ood, Yhat_ood,
        savepath=os.path.join(plot_dir, "pred_vs_truth_ood.png"))

    plot_multi_horizon_rmse(Y_test, Yhat_test,
        savepath=os.path.join(plot_dir, "rmse_test.png"))

    plot_multi_horizon_rmse(Y_ood, Yhat_ood,
        savepath=os.path.join(plot_dir, "rmse_ood.png"))

    print(f"\nEvaluation complete → {plot_dir}")


if __name__ == "__main__":
    main()
