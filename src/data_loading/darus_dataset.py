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


def load_all_splits(base_dir):
    data = load_darus_dataset(base_dir)
    return data["train"], data["val"], data["test"], data["ood_test"]



class SequenceDataset(Dataset):
    def __init__(self, df, history, horizon):
        self.history = history
        self.horizon = horizon

        self.X = df[FEATURE_COLUMNS].values
        self.Y = df[OUTPUT_COLUMNS].values

    def __len__(self):
        return len(self.X) - self.history - self.horizon

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.history]               # (H, 11)
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
