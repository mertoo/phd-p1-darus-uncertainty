"""Post hoc analyses 2.1-2.3 for manuscript v5 (protocol: experiments/analysis/v3/posthoc_v5/PROTOCOL.md).

    python -m scripts.posthoc_v5 --out experiments/analysis/v3/posthoc_v5

2.1 width-matched comparison, 2.2 coverage by operating point, 2.3 calibration-draw sensitivity.
Reads stored predictions only (experiments/eval/v3); nothing is trained or tuned. Descriptive.
"""

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd

from src.analysis.rebuild import EvalDir
from src.uncertainty.intervals import conformal_threshold, spread_floor

EVAL = "experiments/eval/v3"
LEVELS = (0.5, 0.8, 0.9, 0.95)
PRIMARY = 0.9
FAMILIES = {"single": "AR-SCP", "ens": "EN-SCP", "mcd": "DN-SCP", "gauss": "GN-SCP"}
STORED_METHOD = {"ens": "ensemble", "mcd": "mc_dropout", "gauss": "gaussian"}
INPUTS = ["n", "deltal", "deltar", "Vw", "alpha_x", "alpha_y", "u", "v", "p", "r", "phi"]
DATA = {"test": "data/processed/darus/patrol_ship_routine/test", "ood_test": "data/processed/darus/patrol_ship_ood/test"}
H = 30
B_TEST, SEED_TEST = 2000, 20261030
B_CAL, SEED_CAL = 1000, 20261031
OOR_BINS = [(-0.5, 0.0, "0"), (0.0, 0.05, "(0, 0.05]"), (0.05, 0.2, "(0.05, 0.2]"), (0.2, 1.0, "> 0.2")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Agg:
    """Per-recording sums over windows and steps, per channel, for cluster bootstrap over recordings."""

    def __init__(self, rec_idx, n_rec):
        self.rec, self.R = rec_idx, n_rec
        self.count = np.bincount(rec_idx, minlength=n_rec).astype(float)          # windows per recording

    def sums(self, q):                                                              # q: (N, T, D) -> (R, D)
        out = np.zeros((self.R, q.shape[2]))
        np.add.at(out, self.rec, q.sum(axis=1))
        return out


def cov_stat(W, cov_sums, count, T, D):
    """Coverage over all scalars for recording weights W (B, R)."""
    return (W @ cov_sums.sum(axis=1)) / (W @ count * T * D)


def norm_stat(W, sums, count, T, scale):
    """Channel-averaged normalised mean (width or interval score) for recording weights W (B, R)."""
    per_ch = (W @ sums) / (W @ count * T)[:, None]
    return (per_ch / scale).mean(axis=1)


def boot_weights(R, B, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, R, size=(B, R))
    return np.stack([np.bincount(i, minlength=R) for i in idx]).astype(float)


def rec_index(windows):
    recs = pd.Index(sorted(windows.recording.unique()))
    return recs.get_indexer(windows.recording), recs


def consistency(ed, fam, splits):
    rows = []
    kinds = [("conformal", "conformal" if fam == "single" else f"{STORED_METHOD[fam]}+conformal")]
    if fam != "single":
        kinds.append(("spread", STORED_METHOD[fam]))
    for sp in splits:
        for kind, name in kinds:
            c = ed.consistency(sp, kind, "horizon_channel", PRIMARY)
            rows.append(dict(eval=os.path.basename(ed.path), split=sp, kind=name, **c))
    return rows


def window_stats(split, windows, fit_range):
    """History statistics of every window: mean u, mean n, SD of delta_l, out-of-range fraction of the inputs."""
    out = []
    for rec, g in windows.groupby("recording", sort=False):
        d = pd.read_csv(os.path.join(DATA[split], rec))
        oor = ((d[INPUTS] < fit_range.loc["min"]) | (d[INPUTS] > fit_range.loc["max"])).sum(axis=1)
        roll = lambda s, f: getattr(s.rolling(H), f)().shift(-(H - 1))
        st = pd.DataFrame({"mean_u": roll(d["u"], "mean"), "mean_n": roll(d["n"], "mean"),
                           "sd_deltal": d["deltal"].rolling(H).std(ddof=0).shift(-(H - 1)),
                           "oor_frac": roll(oor, "sum") / (H * len(INPUTS))})
        out.append(st.iloc[g.start_row.to_numpy()].set_index(g.index))
    return pd.concat(out).loc[windows.index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/analysis/v3/posthoc_v5")
    ap.add_argument("--backbones", nargs="+", default=["lstm", "mlp"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["test", "ood_test"])
    ap.add_argument("--b_cal", type=int, default=B_CAL)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cal_sel = json.load(open("experiments/manifests/calibration_selection.json"))["random"]["files"]
    sm = pd.read_csv("experiments/manifests/split_manifest.csv")
    fit_files = sorted(set(sm[sm.split == "train"].file) - set(cal_sel))
    assert len(fit_files) == 48 and len(cal_sel) == 9
    fit = pd.concat(pd.read_csv(os.path.join("data/processed/darus/patrol_ship_routine/train", f)) for f in fit_files)
    fit_range = fit[INPUTS].agg(["min", "max"])

    cons_rows, wm_rows, cd_rows = [], [], []
    boot_store = {}          # (split, key) -> list over repeats of (cov_sums, width_sums, is_sums)
    win_store = {}           # (split, label) -> list over repeats of (per-window coverage, per-window normalised width)
    split_info, Wt = {}, {}
    cal_draws = None
    inputs_used = {}

    for bb in args.backbones:
        for rep in args.reps:
            for fam in FAMILIES:
                tag = f"{bb}_{fam}_rep{rep}"
                ed = EvalDir(os.path.join(EVAL, tag))
                for f in ["metrics.json"] + [f"predictions_{s}.npz" for s in ["cal"] + args.splits]:
                    inputs_used[f"{tag}/{f}"] = sha(os.path.join(ed.path, f))
                cons_rows += consistency(ed, fam, args.splits)
                bad = [r for r in cons_rows if max(r["coverage"], r["width_norm_mean"]) > 1e-4]
                if bad:
                    raise RuntimeError(f"rebuilt intervals do not reproduce stored metrics: {bad}")
                cal = ed.arrays("cal")
                scale = ed.scale
                err_cal = np.abs(cal["y"] - cal["mean"])
                if fam != "single":
                    f_floor = spread_floor(cal["spread"], cal["y"], ed.floor_rel)
                    s_cal = np.maximum(cal["spread"], f_floor)
                score_cal = {"AR": err_cal} if fam == "single" else {"AR": err_cal, "X": err_cal / s_cal}
                q = {(k, lv): conformal_threshold(sc, lv, axis=0) for k, sc in score_cal.items() for lv in LEVELS}

                # ---- 2.3 calibration draws (recordings of the calibration split)
                wcal = ed.windows("cal")
                rcal, recs_cal = rec_index(wcal)
                if cal_draws is None:
                    rng = np.random.default_rng(SEED_CAL)
                    R = len(recs_cal)
                    draws = [("original", np.ones(R))]
                    draws += [(f"loo_{recs_cal[i]}", np.where(np.arange(R) == i, 0.0, 1.0)) for i in range(R)]
                    draws += [(f"boot_{b}", np.bincount(rng.integers(0, R, R), minlength=R).astype(float))
                              for b in range(args.b_cal)]
                    cal_draws = draws
                # weighted order statistic per (t, d): sort once, then each draw only reweights recordings
                key = "AR" if fam == "single" else "X"
                Nc = score_cal[key].shape[0]
                flat = score_cal[key].reshape(Nc, -1).T                                       # (T*D, Nc)
                order = np.argsort(flat, axis=1)
                svals = np.take_along_axis(flat, order, axis=1)
                srec = rcal[order]
                draw_q = []
                for name, wts in cal_draws:
                    n = wts[rcal].sum()
                    k = int(np.ceil((n + 1) * PRIMARY))
                    if k > n:
                        qd = np.full(flat.shape[0], np.inf)
                    else:
                        cw = np.cumsum(wts[srec], axis=1)
                        qd = svals[np.arange(flat.shape[0]), (cw < k).sum(axis=1)]
                    draw_q.append((name, qd))
                assert np.allclose(draw_q[0][1], q[(key, PRIMARY)].ravel()), "weighted order statistic != benchmark threshold"

                for sp in args.splits:
                    ev = ed.arrays(sp)
                    w = ed.windows(sp)
                    ridx, recs = rec_index(w)
                    agg = Agg(ridx, len(recs))
                    N, T, D = ev["y"].shape
                    if sp not in split_info:
                        split_info[sp] = (agg.count, len(recs), T, D, w, ridx)
                        Wt[sp] = boot_weights(len(recs), B_TEST, SEED_TEST)
                    err = np.abs(ev["y"] - ev["mean"])
                    s_ev = None if fam == "single" else np.maximum(ev["spread"], f_floor)

                    def store(key, half, lv):
                        a = 2.0 / (1 - lv)
                        cov = err <= half
                        width = np.broadcast_to(2 * half, err.shape)
                        isc = width + a * np.maximum(err - half, 0.0)
                        boot_store.setdefault((sp, key), []).append((agg.sums(cov), agg.sums(width), agg.sums(isc)))
                        return cov, width

                    if fam == "single":
                        for lv in LEVELS:
                            cov, width = store((bb, "AR-SCP", "single", lv), q[("AR", lv)], lv)
                            if lv == PRIMARY:
                                win_store.setdefault((sp, f"AR-SCP, {bb.upper()}"), []).append(
                                    (cov.mean(axis=(1, 2)), (width / scale).mean(axis=(1, 2))))
                    else:
                        for lv in LEVELS:
                            hx = q[("X", lv)] * s_ev
                            cov, width = store((bb, FAMILIES[fam], "X", lv), hx, lv)
                            qa = q[("AR", lv)]
                            store((bb, FAMILIES[fam], "AR_same_mean", lv), np.broadcast_to(qa, err.shape), lv)
                            c_ch = width.mean(axis=(0, 1)) / (2 * qa).mean(axis=0)            # (D,)
                            store((bb, FAMILIES[fam], "AR_wm_channel", lv), np.broadcast_to(c_ch * qa, err.shape), lv)
                            c_td = width.mean(axis=0) / (2 * qa)                               # (T, D)
                            store((bb, FAMILIES[fam], "AR_wm_step_channel", lv), np.broadcast_to(c_td * qa, err.shape), lv)
                            if lv == PRIMARY:
                                win_store.setdefault((sp, f"{FAMILIES[fam]}, {bb.upper()}"), []).append(
                                    (cov.mean(axis=(1, 2)), (width / scale).mean(axis=(1, 2))))

                    # 2.3: thresholds from each calibration draw at the primary level, coverage on this split
                    sc_ev = err if fam == "single" else err / s_ev
                    srt = np.sort(sc_ev.reshape(N, -1).T, axis=1)                             # (T*D, N)
                    mean_s = None if fam == "single" else s_ev.mean(axis=0)               # (T, D)
                    for name, qflat in draw_q:
                        qd = qflat.reshape(T, D)
                        cnt = sum(np.searchsorted(srt[j], qflat[j], side="right") for j in range(T * D))
                        wd = 2 * qd if fam == "single" else 2 * qd * mean_s
                        cd_rows.append(dict(backbone=bb.upper(), construction=FAMILIES[fam], rep=rep, split=sp, draw=name,
                                            coverage=cnt / (N * T * D), width_norm=float((wd.mean(axis=0) / scale).mean())))
                    del ev, err, s_ev, srt
                print(f"done {tag}", flush=True)

    pd.DataFrame(cons_rows).to_csv(f"{args.out}/consistency.csv", index=False)

    # ---- 2.1 width-matched table
    for (sp, key), lst in boot_store.items():
        bb, name, kind, lv = key
        count, R, T, D, _, _ = split_info[sp]
        scale = EvalDir(os.path.join(EVAL, f"{bb}_single_rep0")).scale
        one = np.ones((1, R))
        cov = np.mean([cov_stat(one, c, count, T, D)[0] for c, _, _ in lst])
        cov_ch = np.mean([(c.sum(0) / (count.sum() * T)) for c, _, _ in lst], axis=0)
        wid = np.mean([norm_stat(one, wsum, count, T, scale)[0] for _, wsum, _ in lst])
        isc = np.mean([norm_stat(one, isum, count, T, scale)[0] for _, _, isum in lst])
        wm_rows.append(dict(split=sp, backbone=bb.upper(), construction=name, kind=kind, level=lv, n_repeats=len(lst),
                            coverage=cov, width_norm=wid, iscore_norm=isc,
                            **{f"cov_{c}": v for c, v in zip(["u", "v", "p", "r", "phi"], cov_ch)}))
    wm = pd.DataFrame(wm_rows)

    # recording-level intervals at the primary level: coverage of each kind and the differences X - reference
    ci_rows = []
    for sp in args.splits:
        count, R, T, D, _, _ = split_info[sp]
        for bb in args.backbones:
            for name in ("EN-SCP", "DN-SCP", "GN-SCP"):
                per = {}
                for kind in ("X", "AR_same_mean", "AR_wm_channel", "AR_wm_step_channel"):
                    lst = boot_store[(sp, (bb, name, kind, PRIMARY))]
                    per[kind] = np.mean([cov_stat(Wt[sp], c, count, T, D) for c, _, _ in lst], axis=0)
                for ref in ("AR_same_mean", "AR_wm_channel", "AR_wm_step_channel"):
                    d = 100 * (per["X"] - per[ref])
                    ci_rows.append(dict(split=sp, backbone=bb.upper(), construction=name, reference=ref,
                                        delta_cov_pp_lo=np.percentile(d, 2.5), delta_cov_pp_hi=np.percentile(d, 97.5)))
    wm.merge(pd.DataFrame(ci_rows).rename(columns={"reference": "kind"}), how="left",
             on=["split", "backbone", "construction", "kind"]).to_csv(f"{args.out}/width_matched.csv", index=False)

    # ---- 2.2 operating point
    stats = {sp: window_stats(sp, split_info[sp][4], fit_range) for sp in args.splits}
    pooled = pd.concat(stats.values())
    edges = {c: np.quantile(pooled[c], [0.2, 0.4, 0.6, 0.8]) for c in ("mean_u", "mean_n", "sd_deltal")}
    op_rows = []
    for sp in args.splits:
        count, R, T, D, w, ridx = split_info[sp]
        st = stats[sp]
        bins = {c: np.digitize(st[c], edges[c]) for c in edges}
        bins["oor_frac"] = np.select([st.oor_frac <= hi for lo, hi, _ in OOR_BINS], range(len(OOR_BINS)))
        for label, lst in win_store.items():
            if label[0] != sp:
                continue
            covw = np.mean([c for c, _ in lst], axis=0)
            widw = np.mean([wd for _, wd in lst], axis=0)
            for stat, b in bins.items():
                for k in np.unique(b):
                    m = b == k
                    rc = np.bincount(ridx[m], minlength=R).astype(float)
                    rs = np.bincount(ridx[m], weights=covw[m], minlength=R)
                    rw = np.bincount(ridx[m], weights=widw[m], minlength=R)
                    den = Wt[sp] @ rc
                    bc = np.where(den > 0, (Wt[sp] @ rs) / np.where(den > 0, den, 1), np.nan)   # empty bin in a draw: undefined
                    n_rec_bin = int((rc > 0).sum())
                    if stat == "oor_frac":
                        desc = OOR_BINS[k][2]
                    elif stat == "sd_deltal" and st[stat][m].max() < 1e-5:  # constant azimuth (rolling-SD round-off ~1e-6)
                        desc = "SD = 0 (no steering)"
                    else:
                        lo = st[stat][m].min()
                        desc = f"Q{k + 1} [{lo:.3g}, {st[stat][m].max():.3g}]"
                    op_rows.append(dict(split=sp, construction=label[1], statistic=stat, bin=int(k), bin_desc=desc,
                                        n_windows=int(m.sum()), n_recordings=n_rec_bin, estimable=n_rec_bin >= 5,
                                        coverage=covw[m].mean(),
                                        cov_lo=np.nanpercentile(bc, 2.5) if n_rec_bin >= 5 else np.nan,
                                        cov_hi=np.nanpercentile(bc, 97.5) if n_rec_bin >= 5 else np.nan,
                                        width_norm=widw[m].mean()))
    pd.DataFrame(op_rows).to_csv(f"{args.out}/operating_point.csv", index=False)
    pd.Series({c: e.tolist() for c, e in edges.items()}).to_json(f"{args.out}/operating_point_edges.json")

    # ---- 2.3 summary
    cd = pd.DataFrame(cd_rows)
    cd.to_csv(f"{args.out}/calibration_draws_raw.csv", index=False)
    avg = cd.groupby(["split", "backbone", "construction", "draw"])[["coverage", "width_norm"]].mean().reset_index()
    summ = []
    for (sp, bb, name), g in avg.groupby(["split", "backbone", "construction"]):
        orig = g[g.draw == "original"].iloc[0]
        loo, boot = g[g.draw.str.startswith("loo_")], g[g.draw.str.startswith("boot_")]
        summ.append(dict(split=sp, backbone=bb, construction=name, coverage=orig.coverage,
                         loo_min=loo.coverage.min(), loo_max=loo.coverage.max(),
                         boot_p2_5=boot.coverage.quantile(0.025), boot_p97_5=boot.coverage.quantile(0.975),
                         width_norm=orig.width_norm, width_boot_p2_5=boot.width_norm.quantile(0.025),
                         width_boot_p97_5=boot.width_norm.quantile(0.975)))
    pd.DataFrame(summ).to_csv(f"{args.out}/calibration_draws.csv", index=False)

    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"protocol": "experiments/analysis/v3/posthoc_v5/PROTOCOL.md",
               "protocol_sha256": sha("experiments/analysis/v3/posthoc_v5/PROTOCOL.md"),
               "script_sha256": sha(__file__), "git_head": rev, "args": vars(args),
               "seeds": {"test_bootstrap": SEED_TEST, "calibration_draws": SEED_CAL}, "inputs": inputs_used},
              open(f"{args.out}/run_record.json", "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
