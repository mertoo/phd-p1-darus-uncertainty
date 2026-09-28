"""Gate A: windows never cross recordings or time gaps; counts are exact."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loading.darus_dataset import (
    FEATURE_COLUMNS, OUTPUT_COLUMNS, SequenceDataset, Standardizer,
    contiguous_segments, n_windows,
)


def make_recording(name, length, offset=0.0, time=None):
    t = np.arange(length, dtype=float) if time is None else np.asarray(time, float)
    df = pd.DataFrame({c: offset + np.arange(length, dtype=float) + i * 1e3
                       for i, c in enumerate(FEATURE_COLUMNS)})
    df["time"] = t
    df["source_file"] = name
    return df


def test_two_recordings_count_and_no_crossing():
    # Audit example: two 100-row recordings, H = T = 30.
    recs = [make_recording("a.csv", 100), make_recording("b.csv", 100, offset=1e6)]
    ds = SequenceDataset(recs, history=30, horizon=30)
    assert len(ds) == 2 * (100 - 60 + 1) == 82
    for i in range(len(ds)):
        x, y = ds[i]
        # offset 1e6 marks recording b; a window mixing a and b would span it
        vals = torch.cat([x[:, -5:].flatten(), y.flatten()])
        assert (vals < 5e5).all() or (vals > 5e5).all()


def test_single_dataframe_split_by_source_file():
    df = pd.concat([make_recording("a.csv", 100), make_recording("b.csv", 100, offset=1e6)],
                   ignore_index=True)
    assert len(SequenceDataset(df, 30, 30)) == 82


def test_short_recording_yields_no_windows():
    ds = SequenceDataset([make_recording("short.csv", 59), make_recording("ok.csv", 60)], 30, 30)
    assert len(ds) == 1
    assert ds.meta["recording"].tolist() == ["ok.csv"]


def test_time_gap_splits_segments():
    t = np.concatenate([np.arange(70), np.arange(80, 150)]).astype(float)  # 10 s gap
    ds = SequenceDataset([make_recording("gap.csv", len(t), time=t)], 30, 30)
    assert contiguous_segments(t) == [(0, 70), (70, 140)]
    assert len(ds) == 2 * (70 - 60 + 1)


def test_stride():
    assert n_windows(100, 30, 30, stride=5) == 9
    ds = SequenceDataset([make_recording("a.csv", 100)], 30, 30, stride=5)
    assert len(ds) == 9 and ds.meta["start_row"].tolist() == list(range(0, 41, 5))


def test_target_alignment():
    rec = make_recording("a.csv", 100)
    ds = SequenceDataset([rec], 30, 30)
    x, y = ds[7]
    tgt = rec[OUTPUT_COLUMNS].to_numpy()
    feat = rec[FEATURE_COLUMNS].to_numpy()
    np.testing.assert_allclose(x.numpy(), feat[7:37], rtol=1e-6)
    np.testing.assert_allclose(y.numpy(), tgt[37:67], rtol=1e-6)
    np.testing.assert_allclose(ds.targets_physical()[7], tgt[37:67])


def test_scaling_round_trip():
    recs = [make_recording("a.csv", 100)]
    xs = Standardizer.fit([recs[0][FEATURE_COLUMNS].to_numpy(float)])
    ys = Standardizer.fit([recs[0][OUTPUT_COLUMNS].to_numpy(float)])
    ds = SequenceDataset(recs, 30, 30, x_scaler=xs, y_scaler=ys)
    _, y = ds[3]
    np.testing.assert_allclose(ys.inverse_transform(y.numpy().astype(float)),
                               ds.targets_physical()[3], rtol=1e-5, atol=1e-3)


def test_naive_respects_scaling():
    from src.models.naive import NaiveBaseline
    rec = make_recording("a.csv", 100)
    xs = Standardizer.fit([rec[FEATURE_COLUMNS].to_numpy(float)])
    ys = Standardizer.fit([rec[OUTPUT_COLUMNS].to_numpy(float)])
    ds = SequenceDataset([rec], 30, 30, x_scaler=xs, y_scaler=ys)
    m = NaiveBaseline(5, 30)
    m.set_scaling(xs.mean[-5:], xs.std[-5:], ys.mean, ys.std)
    x, _ = ds[0]
    pred = ys.inverse_transform(m(x[None]).numpy()[0].astype(float))
    last_obs = rec[OUTPUT_COLUMNS].to_numpy()[29]
    np.testing.assert_allclose(pred, np.tile(last_obs, (30, 1)), rtol=1e-4)
