"""Rebuild intervals from an evaluation directory's saved predictions (CPU only).

benchmark_eval stores float32 predictions for the calibration split and every
evaluated split. The same interval constructors are re-applied here, so the
floor-sensitivity analysis and figures use exactly the evaluator's rules. The
float32 round-trip introduces tiny differences; `consistency` quantifies them
against the stored metrics before any rebuilt result is used.
"""

import json
import os

import numpy as np
import pandas as pd

from src.evaluation.uq_metrics import interval_metrics
from src.uncertainty.intervals import (conformal_intervals, raw_gaussian_intervals,
                                       scaled_spread_intervals)

SPREAD_METHODS = ("ensemble", "mc_dropout", "gaussian")


class EvalDir:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "metrics.json")) as f:
            self.metrics = json.load(f)
        prov = self.metrics["provenance"]
        self.method = prov["method"]
        self.cal_split = prov["calibration_split"]
        self.eval_splits = prov["eval_splits"]
        self.scale = np.asarray(prov["scale_for_normalised_metrics"], float)
        self.floor_rel = prov["floor_rel"]

    def arrays(self, split):
        d = np.load(os.path.join(self.path, f"predictions_{split}.npz"))
        return {k: d[k].astype(np.float64) for k in d.files}

    def windows(self, split):
        return pd.read_csv(os.path.join(self.path, f"windows_{split}.csv"))

    def interval(self, split, kind, variant, level, floor_rel=None):
        """kind: 'conformal' (abs residual on the mean), 'spread' (spread-normalised), 'raw'."""
        cal, ev = self.arrays(self.cal_split), self.arrays(split)
        if kind == "conformal":
            return conformal_intervals(cal["mean"], cal["y"], ev["mean"], level, variant,
                                       method="conformal" if self.method == "point" else f"{self.method}+conformal")
        if self.method not in SPREAD_METHODS:
            raise ValueError(f"{self.path}: method {self.method} has no spread")
        if kind == "raw":
            return raw_gaussian_intervals(ev["mean"], ev["spread"], level, self.method)
        if kind == "spread":
            return scaled_spread_intervals(cal["mean"], cal["spread"], cal["y"], ev["mean"], ev["spread"],
                                           level, variant, self.method,
                                           floor_rel=self.floor_rel if floor_rel is None else floor_rel)
        raise ValueError(kind)

    def stored(self, split, method, variant, level):
        for iv in self.metrics["intervals"]:
            if (iv["split"] == split and iv["method"] == method and iv["variant"] == variant
                    and abs(iv["level"] - level) < 1e-9):
                return iv
        raise KeyError((split, method, variant, level))

    def consistency(self, split, kind, variant, level):
        """Max absolute difference between rebuilt and stored coverage / normalised width."""
        iv = self.interval(split, kind, variant, level)
        y = self.arrays(split)["y"]
        m = interval_metrics(y, iv, scale=self.scale)
        st = self.stored(split, iv.method, iv.variant, level)
        return {"coverage": abs(m["coverage"] - st["coverage"]),
                "width_norm_mean": abs(m["width_norm_mean"] - st["width_norm_mean"])}
