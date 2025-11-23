import pandas as pd
import os
import glob

def load_split(split_path):
    """
    Loads all CSV or TAB files from a split directory.
    Returns a concatenated pandas DataFrame or None.
    """
    files = sorted(
        glob.glob(os.path.join(split_path, "*.csv")) +
        glob.glob(os.path.join(split_path, "*.tab"))
    )

    if len(files) == 0:
        print(f"⚠️ No files found in: {split_path}")
        return None

    dfs = []
    for f in files:
        sep = "," if f.endswith(".csv") else "\t"
        df = pd.read_csv(f, sep=sep)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_darus_dataset(base_path):
    """
    Loads the PROCESSED DaRUS dataset.
    Folder structure must be:

    base_path/
        patrol_ship_routine/
            train/
            validation/
            test/
        patrol_ship_ood/
            test/
    """
    routine = os.path.join(base_path, "patrol_ship_routine")
    ood = os.path.join(base_path, "patrol_ship_ood")

    dataset = {
        "train": load_split(os.path.join(routine, "train")),
        "val": load_split(os.path.join(routine, "validation")),
        "test": load_split(os.path.join(routine, "test")),
        "ood_test": load_split(os.path.join(ood, "test")),
    }

    return dataset
