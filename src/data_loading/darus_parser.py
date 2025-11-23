import pandas as pd
import glob
import os

def load_darus_raw(folder_path):
    """
    Loads all .tab files from the DaRUS dataset folder.
    Returns a list of pandas DataFrames.
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*.tab")))
    dataframes = []

    for f in files:
        df = pd.read_csv(f, sep="\t")
        df["source_file"] = os.path.basename(f)
        dataframes.append(df)

    return dataframes


def concatenate_darus_runs(dfs):
    """
    Combine multiple runs into single DataFrame with run ID.
    """
    combined = []
    for idx, df in enumerate(dfs):
        df = df.copy()
        df["run_id"] = idx
        combined.append(df)
    return pd.concat(combined, ignore_index=True)
