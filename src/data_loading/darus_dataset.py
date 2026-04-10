import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data_loading.darus_parser import load_darus_dataset



# Full input features (12 dims)
FEATURE_COLUMNS = [
    "time",
    "n", "deltal", "deltar",
    "Vw", "alpha_x", "alpha_y",
    "u", "v", "p", "r", "phi"
]



# Output DoFs (the 5 we predict)
OUTPUT_COLUMNS = ["u", "v", "p", "r", "phi"]

def _as_int(v, name: str, default: int | None = None) -> int:
    """
    Robustly convert config-derived values into ints.
    Handles cases where v is a dict (e.g., passing config or config['data'] by mistake).
    """
    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return int(v)

    if isinstance(v, dict):
        # common patterns:
        # 1) {"history": 30} or {"horizon": 30}
        if name in v and isinstance(v[name], (int, float)):
            return int(v[name])

        # 2) accidentally passed whole config: {"data": {"history": 30, "horizon": 30}, ...}
        if "data" in v and isinstance(v["data"], dict) and name in v["data"] and isinstance(v["data"][name], (int, float)):
            return int(v["data"][name])

        # 3) generic "value" patterns
        for k in ["value", "len", "length", "steps"]:
            if k in v and isinstance(v[k], (int, float)):
                return int(v[k])

    if default is not None:
        return default

    raise TypeError(f"{name} must be int-like, got {type(v)} with value={v}")


def load_all_splits(base_dir):
    data = load_darus_dataset(base_dir)
    return data["train"], data["val"], data["test"], data["ood_test"]



class SequenceDataset(Dataset):
    def __init__(self, df, history, horizon):
        self.history = _as_int(history, "history", default=30)
        self.horizon = _as_int(horizon, "horizon", default=30)

        self.X = df[FEATURE_COLUMNS].values
        self.Y = df[OUTPUT_COLUMNS].values

    def __len__(self):
        return len(self.X) - self.history - self.horizon

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.history]               # (H, 12)
        y = self.Y[idx + self.history : idx + self.history + self.horizon]  # (H, 5)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def create_dataloaders(history, horizon, batch_size):
    print("🔧 create_dataloaders called!")

    base_dir = "data/processed/darus"
    train_df, val_df, test_df, ood_df = load_all_splits(base_dir)

    train_ds = SequenceDataset(train_df, history, horizon)
    val_ds = SequenceDataset(val_df, history, horizon)
    test_ds = SequenceDataset(test_df, history, horizon)
    ood_ds = SequenceDataset(ood_df, history, horizon)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        DataLoader(ood_ds, batch_size=batch_size, shuffle=False),
    )
