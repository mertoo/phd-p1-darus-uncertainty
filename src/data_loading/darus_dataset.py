import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loading.darus_parser import load_split_recordings, list_split_files

DEFAULT_BASE_DIR = "data/processed/darus"

# Full input features (12 dims) used by all historical runs. `time` is the
# per-recording elapsed time in seconds (it restarts at 0 in every file), so it
# encodes position within a recording rather than a physical state.
FEATURE_COLUMNS = [
    "time",
    "n", "deltal", "deltar",
    "Vw", "alpha_x", "alpha_y",
    "u", "v", "p", "r", "phi"
]

# Output DoFs (the 5 we predict)
OUTPUT_COLUMNS = ["u", "v", "p", "r", "phi"]

FEATURE_SETS = {
    "legacy": FEATURE_COLUMNS,
    "no_time": [c for c in FEATURE_COLUMNS if c != "time"],
}


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


def resolve_feature_columns(spec):
    if spec is None:
        return list(FEATURE_COLUMNS)
    if isinstance(spec, str):
        if spec not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{spec}'. Options: {sorted(FEATURE_SETS)}")
        return list(FEATURE_SETS[spec])
    return list(spec)


def contiguous_segments(time, dt=1.0, tol=1e-6):
    """
    Split a time vector into maximal runs sampled at exactly `dt`.
    Returns a list of (start, stop) row ranges (stop exclusive). Any gap,
    repeat or reversal in `time` starts a new segment.
    """
    time = np.asarray(time, dtype=float)
    if len(time) == 0:
        return []
    breaks = np.where(np.abs(np.diff(time) - dt) > tol)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(time)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def n_windows(length, history, horizon, stride=1):
    """Number of complete (history + horizon) windows in a segment of `length` rows."""
    span = history + horizon
    if length < span:
        return 0
    return (length - span) // stride + 1


class Standardizer:
    """Per-channel affine scaling fitted on training recordings only."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, arrays, min_std=1e-8):
        stacked = np.concatenate(arrays, axis=0)
        std = stacked.std(axis=0)
        return cls(stacked.mean(axis=0), np.where(std < min_std, 1.0, std))

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse_transform(self, a):
        return a * self.std + self.mean

    def inverse_scale(self, s):
        """Map a spread (std/width) from scaled to physical units."""
        return s * self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


class SequenceDataset(Dataset):
    """
    Sliding (history -> horizon) windows built independently inside every
    contiguous, regularly sampled segment of every recording. No window spans
    two recordings or a time discontinuity.

    recordings: list of per-recording DataFrames, or a single DataFrame (split
    by its `source_file` column when present).
    """

    def __init__(self, recordings, history, horizon, stride=1,
                 feature_columns=None, target_columns=None,
                 x_scaler=None, y_scaler=None, dt=1.0):
        self.history = _as_int(history, "history", default=30)
        self.horizon = _as_int(horizon, "horizon", default=30)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.feature_columns = resolve_feature_columns(feature_columns)
        self.target_columns = list(target_columns or OUTPUT_COLUMNS)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        if isinstance(recordings, pd.DataFrame):
            if "source_file" in recordings.columns:
                recordings = [g for _, g in recordings.groupby("source_file", sort=False)]
            else:
                recordings = [recordings]

        self._segX, self._segY = [], []
        seg_index, meta = [], []
        for rec in recordings:
            name = rec["source_file"].iloc[0] if "source_file" in rec.columns else "recording"
            time = rec["time"].to_numpy(dtype=float) if "time" in rec.columns else np.arange(len(rec), dtype=float)
            X = rec[self.feature_columns].to_numpy(dtype=np.float64)
            Y = rec[self.target_columns].to_numpy(dtype=np.float64)
            for seg_no, (a, b) in enumerate(contiguous_segments(time, dt=dt)):
                k = n_windows(b - a, self.history, self.horizon, self.stride)
                if k == 0:
                    continue
                sid = len(self._segX)
                self._segX.append(X[a:b])
                self._segY.append(Y[a:b])
                starts = np.arange(k) * self.stride
                seg_index.append(np.stack([np.full(k, sid), starts], axis=1))
                meta.append(pd.DataFrame({
                    "recording": name,
                    "segment": seg_no,
                    "start_row": a + starts,
                    "start_time": time[a + starts],
                }))

        self.index = (np.concatenate(seg_index, axis=0) if seg_index
                      else np.zeros((0, 2), dtype=np.int64))
        self.meta = (pd.concat(meta, ignore_index=True) if meta
                     else pd.DataFrame(columns=["recording", "segment", "start_row", "start_time"]))
        self.meta.index.name = "window_id"

        # Scaled copies of each segment, cached once.
        self._segXs = [self._scale_x(x) for x in self._segX]
        self._segYs = [self._scale_y(y) for y in self._segY]

    # --- scaling -------------------------------------------------------------
    def _scale_x(self, x):
        return self.x_scaler.transform(x) if self.x_scaler is not None else x

    def _scale_y(self, y):
        return self.y_scaler.transform(y) if self.y_scaler is not None else y

    # --- shape info (X/Y kept for code that infers dims from them) ------------
    @property
    def input_dim(self):
        return len(self.feature_columns)

    @property
    def target_dim(self):
        return len(self.target_columns)

    @property
    def X(self):
        return np.empty((0, self.input_dim))

    @property
    def Y(self):
        return np.empty((0, self.target_dim))

    @property
    def recordings(self):
        return sorted(self.meta["recording"].unique().tolist())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, s = self.index[idx]
        x = self._segXs[sid][s: s + self.history]
        y = self._segYs[sid][s + self.history: s + self.history + self.horizon]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def targets_physical(self):
        """All target windows (N, T, D) in physical units, in dataset order."""
        out = np.empty((len(self), self.horizon, self.target_dim))
        for i, (sid, s) in enumerate(self.index):
            out[i] = self._segY[sid][s + self.history: s + self.history + self.horizon]
        return out


def select_calibration_files(train_files, n_recordings, seed, strata_values=None):
    """
    Deterministically hold out `n_recordings` training recordings for calibration.

    Random mode (strata_values=None): uniform draw without replacement from the
    sorted names. Stratified mode: recordings are sorted by `strata_values`
    (e.g. per-recording mean shaft speed n), cut into n_recordings equal-count
    strata, and one recording is drawn per stratum.
    """
    names = sorted(train_files)
    if not 0 < n_recordings < len(names):
        raise ValueError(f"n_recordings must be in (0, {len(names)})")
    rng = np.random.default_rng(seed)
    if strata_values is None:
        return sorted(rng.choice(names, size=n_recordings, replace=False).tolist())
    order = sorted(names, key=lambda f: (strata_values[f], f))
    strata = np.array_split(np.array(order), n_recordings)
    return sorted(str(rng.choice(stratum)) for stratum in strata)


def build_datasets(data_cfg):
    """
    Build every split dataset from a `data:` config block.

    Recognised keys: base_path, history, horizon, stride, features
    ('legacy' | 'no_time' | explicit list), normalize (bool; scalers fitted on
    the training recordings only), calibration ({n_recordings, seed} or
    {files: [...]}) which moves recordings out of `train` into a separate
    `cal` split. Returns (datasets dict, data_state dict).
    """
    known = {"base_path", "history", "horizon", "stride", "features", "normalize",
             "calibration", "batch_size", "num_workers", "input_dim", "output_dim"}
    unknown = set(data_cfg) - known
    if unknown:
        raise KeyError(f"Unknown data config keys: {sorted(unknown)}")

    base = data_cfg.get("base_path", DEFAULT_BASE_DIR)
    history = int(data_cfg["history"])
    horizon = int(data_cfg["horizon"])
    stride = int(data_cfg.get("stride", 1))
    features = resolve_feature_columns(data_cfg.get("features"))

    train_names = [p.split("/")[-1] for p in list_split_files(base, "train")]
    cal_cfg = data_cfg.get("calibration")
    cal_files = []
    if cal_cfg:
        mode = cal_cfg.get("method", "random")
        if "files" in cal_cfg:
            cal_files = sorted(cal_cfg["files"])
        elif mode == "random":
            cal_files = select_calibration_files(train_names, int(cal_cfg["n_recordings"]),
                                                 int(cal_cfg.get("seed", 0)))
        elif mode == "stratified_n":
            means = {r["source_file"].iloc[0]: float(r["n"].mean())
                     for r in load_split_recordings(base, "train")}
            cal_files = select_calibration_files(train_names, int(cal_cfg["n_recordings"]),
                                                 int(cal_cfg.get("seed", 0)), strata_values=means)
        else:
            raise ValueError(f"Unknown calibration method '{mode}'")
    fit_files = [f for f in train_names if f not in set(cal_files)]

    recs = {
        "train": load_split_recordings(base, "train", files=fit_files),
        "val": load_split_recordings(base, "val"),
        "test": load_split_recordings(base, "test"),
        "ood_test": load_split_recordings(base, "ood_test"),
    }
    if cal_files:
        recs["cal"] = load_split_recordings(base, "train", files=cal_files)

    x_scaler = y_scaler = None
    if data_cfg.get("normalize", False):
        x_scaler = Standardizer.fit([r[features].to_numpy(float) for r in recs["train"]])
        y_scaler = Standardizer.fit([r[OUTPUT_COLUMNS].to_numpy(float) for r in recs["train"]])

    datasets = {
        k: SequenceDataset(v, history, horizon, stride=stride, feature_columns=features,
                           x_scaler=x_scaler, y_scaler=y_scaler)
        for k, v in recs.items()
    }
    state = {
        "base_path": base,
        "history": history,
        "horizon": horizon,
        "stride": stride,
        "features": features,
        "targets": list(OUTPUT_COLUMNS),
        "split_files": {k: d.recordings for k, d in datasets.items()},
        "n_windows": {k: len(d) for k, d in datasets.items()},
        "x_scaler": x_scaler.to_dict() if x_scaler else None,
        "y_scaler": y_scaler.to_dict() if y_scaler else None,
    }
    state["split_hash"] = hashlib.sha256(
        json.dumps(state["split_files"], sort_keys=True).encode()).hexdigest()[:16]
    return datasets, state


def make_loader(ds, batch_size, shuffle=False, seed=None, num_workers=0):
    gen = torch.Generator().manual_seed(int(seed)) if (shuffle and seed is not None) else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=gen, num_workers=num_workers)


def create_dataloaders(history, horizon, batch_size, base_dir=DEFAULT_BASE_DIR, seed=None, **data_kwargs):
    """
    Backward-compatible entry point returning (train, val, test, ood) loaders
    built with recording-aware windowing. Extra keyword arguments are passed
    through as `data:` config keys (stride, features, normalize, calibration).
    """
    data_cfg = {"base_path": base_dir, "history": history, "horizon": horizon, **data_kwargs}
    ds, _ = build_datasets(data_cfg)
    return (
        make_loader(ds["train"], batch_size, shuffle=True, seed=seed),
        make_loader(ds["val"], batch_size),
        make_loader(ds["test"], batch_size),
        make_loader(ds["ood_test"], batch_size),
    )
