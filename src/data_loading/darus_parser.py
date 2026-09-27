import glob
import os

import pandas as pd

# Split name -> directory relative to the processed-data root. These are the
# partitions shipped with the DaRUS deposits (routine: train/validation/test;
# ood: test only); this code does not re-split recordings between them.
SPLIT_DIRS = {
    "train": os.path.join("patrol_ship_routine", "train"),
    "val": os.path.join("patrol_ship_routine", "validation"),
    "test": os.path.join("patrol_ship_routine", "test"),
    "ood_test": os.path.join("patrol_ship_ood", "test"),
}


def list_split_files(base_path, split):
    """Sorted CSV/TAB recording paths for one split."""
    split_path = os.path.join(base_path, SPLIT_DIRS[split])
    return sorted(
        glob.glob(os.path.join(split_path, "*.csv")) +
        glob.glob(os.path.join(split_path, "*.tab"))
    )


def read_recording(path):
    sep = "," if path.endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep)
    df["source_file"] = os.path.basename(path)
    return df


def load_split_recordings(base_path, split, files=None):
    """
    Load one split as a list of per-recording DataFrames (never concatenated,
    so windows cannot straddle two independent recordings).

    files: optional explicit list of file basenames to restrict to (used for
    calibration hold-outs defined by a manifest).
    """
    paths = list_split_files(base_path, split)
    if files is not None:
        wanted = set(files)
        paths = [p for p in paths if os.path.basename(p) in wanted]
        missing = wanted - {os.path.basename(p) for p in paths}
        if missing:
            raise FileNotFoundError(f"Recordings not found in split '{split}': {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(
            f"No recordings found for split '{split}' under "
            f"{os.path.join(base_path, SPLIT_DIRS[split])}"
        )
    return [read_recording(p) for p in paths]


def load_split(split_path):
    """
    Legacy helper: concatenated DataFrame of every recording in a directory,
    with a `source_file` column. Do not window this frame directly; use
    load_split_recordings / SequenceDataset instead.
    """
    files = sorted(
        glob.glob(os.path.join(split_path, "*.csv")) +
        glob.glob(os.path.join(split_path, "*.tab"))
    )
    if len(files) == 0:
        print(f"⚠️ No files found in: {split_path}")
        return None
    return pd.concat([read_recording(f) for f in files], ignore_index=True)


def load_darus_dataset(base_path):
    """
    Loads the PROCESSED DaRUS dataset as per-split lists of recordings.
    Folder structure must be:

    base_path/
        patrol_ship_routine/
            train/
            validation/
            test/
        patrol_ship_ood/
            test/
    """
    return {split: load_split_recordings(base_path, split) for split in SPLIT_DIRS}
