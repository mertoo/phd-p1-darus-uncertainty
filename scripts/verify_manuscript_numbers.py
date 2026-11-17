"""Recompute every result number quoted in the running text of the manuscript and compare it with the text.

    python -m scripts.verify_manuscript_numbers --tex paper/revision_v5_2/main_v5_2.tex \
        --out paper/revision_v5_2/number_check.md

Each check gives the string as printed in the manuscript and the string recomputed from the stored outputs with the
same rounding; the check also confirms that the printed string occurs in the manuscript. Numbers that are generated
(tables, prose macros) are checked by report_pub and its tests, and literature values are checked separately
(citation check). Reads stored outputs and the deposit files only.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from scripts.check_calibration_split import VARS
from src.data_loading.darus_parser import load_split_recordings

BOOT = "experiments/analysis/v3/bootstrap"
PH = "experiments/analysis/v3/posthoc"
PH5 = "experiments/analysis/v3/posthoc_v5"
LEARNED = ["lstm", "gru", "tcn", "mlp", "linear", "ridge"]
NETS = ["lstm", "gru", "tcn", "mlp"]
SPREAD = ["ensemble", "mcd", "gaussian"]


def md_table(df):
    cell = lambda x: str(x).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)])


def rng(vals, nd, scale=1.0, sep="--"):
    v = [scale * x for x in vals]
    lo, hi = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return lo if lo == hi else f"{lo}{sep}{hi}"


class Checker:
    def __init__(self, tex):
        self.tex = re.sub(r"(?m)^\s*%.*$", "", tex)
        self.rows = []

    def __call__(self, section, claim, printed, computed, context=None):
        ctx = context or printed
        found = ctx in self.tex
        self.rows.append(dict(section=section, claim=claim, printed=printed, recomputed=computed,
                              ok=(printed == computed) and found, in_text=found))


def posthoc_v52_checks(c):
    """Numbers added in the v5.2 revision (2026-11-16): measurement noise, clean recalibration, per-split errors,
    matched-contrast bootstrap, novelty normaliser."""
    A = "experiments/analysis/v3"
    # measurement noise (Sections 2.1, 3.5)
    s = "3.5 Measurement noise"
    nz = pd.read_csv(f"{A}/noise_check.csv")
    tr = nz[nz.role.isin(["fitting", "calibration"])]
    c(s, "training u, v noise sigma (m/s)", "0.094$--$0.102", rng(list(tr.u_sigma) + list(tr.v_sigma), 3, sep="$--$"))
    ood = nz[nz.role == "OOD test"]
    noisy = ood[ood.u_sigma > 0.05]
    c(s, "noisy OOD recordings", "17 of the 29", f"{len(noisy)} of the {len(ood)}")
    c(s, "noisy OOD sigma (m/s)", "0.31--0.33", rng(list(noisy.u_sigma) + list(noisy.v_sigma), 2))
    clean = pd.concat([nz[nz.role.isin(["validation", "ID test"])], ood[ood.u_sigma <= 0.05]])
    c(s, "clean recordings, max sigma u/v (m/s)", "0.008", f"{max(clean.u_sigma.max(), clean.v_sigma.max()):.3f}",
      "give at most 0.008~m/s")
    # clean recalibration (Table noise and text)
    cc = pd.read_csv(f"{A}/posthoc_clean_cal/summary.csv")
    learned = cc[~cc.backbone.isin(["naive"])]
    q = lambda sp, cal, cons: learned[(learned.split == sp) & (learned.calibration == cal) & (learned.construction.isin(cons))]
    c(s, "clean-cal ID coverage, AR/EN/DN (%)", "90.3--91.3", rng(q("test", "val", ["AR-SCP", "EN-SCP", "DN-SCP"]).coverage, 1))
    c(s, "clean-cal ID coverage, GN (%)", "89.6--89.8", rng(q("test", "val", ["GN-SCP"]).coverage, 1))
    sv = q("test", "val", ["AR-SCP", "EN-SCP", "DN-SCP", "GN-SCP"])
    c(s, "clean-cal ID surge and sway coverage (%)", "89.1--90.9", rng(list(sv.cov_u) + list(sv.cov_v), 1))
    nets = ["lstm", "mlp", "gru", "tcn"]
    ar_clean_ood = q("ood_test", "cal", ["AR-SCP"])
    ar_clean_ood = ar_clean_ood[~ar_clean_ood.backbone.str.contains("member")]
    c(s, "primary-cal AR-SCP OOD coverage in clean recordings (%)", "54--57", rng(ar_clean_ood.cov_clean, 0),
      "AR-SCP covers only 54--57")
    gap = q("ood_test", "cal", ["AR-SCP", "EN-SCP", "DN-SCP", "GN-SCP"])
    c(s, "clean minus noisy OOD recordings, primary cal (pp)", "5--11", rng(gap.cov_clean - gap.cov_noisy, 0),
      "covered 5--11 points less")
    m = q("ood_test", "val", ["AR-SCP"])
    m = m[m.backbone.isin(nets)]
    c(s, "clean-cal AR-SCP OOD, networks (%)", "44--46", rng(m.coverage, 0), "AR-SCP 44--46\\%")
    for cons, pr_ in (("EN-SCP", "59--66"), ("DN-SCP", "62--69"), ("GN-SCP", "40--42")):
        c(s, f"clean-cal {cons} OOD (%)", pr_, rng(q("ood_test", "val", [cons]).coverage, 0), f"{cons} {pr_}\\%")
    pv = cc.pivot_table(index=["split", "construction", "backbone"], columns="calibration", values="coverage").reset_index()
    pv = pv[(pv.split == "ood_test") & (pv.backbone != "naive")]
    c(s, "OOD drop with clean calibration (pp)", "6--11", rng(pv.cal - pv.val, 0), "falls by a further 6--11 points")
    wd = cc[(cc.split == "test") & (cc.backbone != "naive")].pivot_table(index=["construction", "backbone"], columns="calibration", values="width")
    shr = (100 * (1 - wd.val / wd.cal)).groupby(level=0)
    c(s, "clean-cal narrowing, AR-SCP (%)", "5--8", rng(shr.get_group("AR-SCP"), 0), "by 5--8\\% (AR-SCP)")
    c(s, "clean-cal narrowing, EN/DN/GN (%)", "10--14", rng(pd.concat([shr.get_group(k) for k in ("EN-SCP", "DN-SCP", "GN-SCP")]), 0))
    # per-channel noise split (noise_group_coverage_mean.csv)
    g = pd.read_csv(f"{A}/noise_group_coverage_mean.csv")
    g = g[(g.split == "ood_test") & (g.construction == "AR-SCP")]
    gc, gn = g[g.group == "clean"].set_index("backbone"), g[g.group == "noisy"].set_index("backbone")
    c(s, "AR-SCP sway, clean OOD recordings (%)", "65--73", rng(gc.v, 0), "covered 65--73\\%")
    c(s, "AR-SCP sway, noisy OOD recordings (%)", "36--37", rng(gn.v, 0), "36--37\\% in the noisy")
    d = max((gc[ch] - gn[ch]).abs().max() for ch in ("p", "r", "phi"))
    c(s, "p, r, phi clean-noisy difference (pp, max)", "3", f"{d:.0f}", "differs by at most 3 points")
    c(s, "AR-SCP yaw rate, both groups (%)", "38--43", rng(list(gc.r) + list(gn.r), 0), "only 38--43\\% in both")

    # per-split scale errors (Section 3.1)
    s = "3.1 Per-split scale"
    e = pd.read_csv(f"{A}/per_split_scale_errors.csv")
    lm = e[e.model != "Persistence"].groupby(["split", "model"]).nrmse_own.mean()
    pm = e[e.model == "Persistence"].groupby("split").nrmse_own.mean()
    c(s, "own-scale NRMSE ID, learned", "0.34--0.36", rng(lm["test"], 2))
    c(s, "own-scale NRMSE OOD, learned", "0.62--0.67", rng(lm["ood_test"], 2))
    c(s, "own-scale NRMSE persistence ID and OOD", "0.53 and 0.85", f"{pm['test']:.2f} and {pm['ood_test']:.2f}")
    sk = e[(e.split == "ood_test") & (e.model != "Persistence")].skill_persistence
    c(s, "OOD skill vs persistence, per channel (%)", "4--36", rng(sk, 0, 100), "by 4--36\\%")
    c(s, "every learned NRMSE_own < 1", "True", str(bool((e[e.model != "Persistence"].nrmse_own < 1).all())),
      "and the standard deviation of the split")

    # matched-contrast bootstrap and ridge (Sections 3.1, 3.4)
    s = "3.4 Matched bootstrap"
    bp = pd.read_csv(f"{A}/bootstrap_matched/paired_test.csv").set_index(["method_a", "method_b", "statistic"])
    fmt = lambda r: f"{r.difference:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
    c(s, "ridge - GRU, ID NRMSE", "0.014 [0.012, 0.016]", fmt(bp.loc[("ridge", "gru", "rmse_norm_mean")]))
    c(s, "ridge - MLP, ID NRMSE", "0.006 [0.003, 0.008]", fmt(bp.loc[("ridge", "mlp", "rmse_norm_mean")]))
    bo = pd.read_csv(f"{A}/bootstrap_matched/paired_ood_test.csv").set_index(["method_a", "method_b", "statistic"])
    f1 = lambda a, b: (lambda r: f"{100 * r.difference:.1f} [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]".replace("-", "$-$"))(
        bo.loc[(a, b, "coverage")])
    for a, b in (("ens_norm_lstm", "ens_ar_lstm"), ("mcd_norm_mlp", "mcd_ar_mlp"), ("gauss_norm_lstm", "gauss_ar_lstm"),
                 ("gauss_norm_mlp", "gauss_ar_mlp"), ("mcd_ar_mlp", "single_ar_mlp")):
        v = f1(a, b)
        c(s, f"tab:matched OOD Delta {a} - {b}", v, v, f"& {v} \\\\")

    # novelty normaliser (Section 3.4)
    s = "3.4 Novelty normaliser"
    ns = pd.read_csv(f"{A}/posthoc_novelty/novelty_summary.csv")
    nm = ns[ns.construction == "NV-SCP (M)"]
    c(s, "NV-SCP (M) OOD coverage (%)", "77.6--79.4", rng(nm[nm.split == "ood_test"].coverage, 1))
    c(s, "NV-SCP (M) OOD width", "2.19--2.32", rng(nm[nm.split == "ood_test"].width_norm, 2))
    c(s, "NV-SCP (M) ID coverage (%)", "82--85", rng(nm[nm.split == "test"].coverage, 0), "covered only 82--85\\%")
    npd = pd.read_csv(f"{A}/posthoc_novelty/novelty_paired.csv")
    po = npd[npd.split == "ood_test"]
    for b in ("LSTM", "MLP"):
        r = po[(po.a == "EN-SCP") & (po.b == "NV-SCP (M)") & (po.backbone == b)].iloc[0]
        v = f"{-r.delta_cov_pp:.1f} [$-${r.ci_high:.1f}, {-r.ci_low:.1f}]"
        c(s, f"NV(M) minus EN-SCP, {b} (pp)", v, v, f"{v} ({b})")
    dn = po[(po.a == "DN-SCP") & (po.b == "NV-SCP (M)")].delta_cov_pp
    c(s, "NV(M) minus DN-SCP (pp)", "5.1--7.7", rng(-dn, 1))
    nl = ns[ns.construction == "NV-SCP (L)"]
    ar = ns[ns.construction == "AR-SCP"].set_index(["split", "backbone"]).width_norm
    ratio = (nl.set_index(["split", "backbone"]).width_norm / ar).max()
    c(s, "leverage factor leaves widths unchanged (max width ratio < 1.02)", "True", str(bool(ratio < 1.02)),
      "stayed below 1.02")
    bb = pd.read_csv(f"{A}/noise_group_coverage.csv")
    bb = bb[(bb.split == "ood_test") & (bb.construction == "AR-SCP") & (bb.group == "all") & (bb.backbone != "naive") &
            (bb.backbone != "ridge")].groupby("backbone")["all"].mean()
    c("2.3 Disclosure", "AR-SCP OOD across five trained backbones (%)", "50.0--52.3", rng(bb, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/revision_v5_2/main_v5_2.tex")
    ap.add_argument("--out", default="paper/revision_v5_2/number_check.md")
    ap.add_argument("--base", default="data/processed/darus")
    args = ap.parse_args()
    c = Checker(open(args.tex).read())

    S = {sp: pd.read_csv(f"{BOOT}/summary_{sp}.csv").set_index(["method", "statistic"]) for sp in ("test", "ood_test")}
    P = {sp: pd.read_csv(f"{BOOT}/paired_{sp}.csv").set_index(["method_a", "method_b", "statistic"])
         for sp in ("test", "ood_test")}
    est = lambda sp, m, st: S[sp].loc[(m, st), "estimate"]
    conf = lambda b: f"conformal_{b}"
    O1 = pd.read_csv(f"{PH}/o1_coverage_breakdown.csv")
    MT = pd.read_csv(f"{PH}/matched_score_vs_predictor.csv")

    # ---- point accuracy (Section 3.1)
    s = "3.1 Point accuracy"
    c(s, "ID normalised RMSE, learned", "0.314--0.336", rng([est("test", conf(b), "rmse_norm_mean") for b in LEARNED], 3))
    c(s, "ID normalised RMSE, persistence", "0.494", f"{est('test', 'conformal_naive', 'rmse_norm_mean'):.3f}")
    red = [100 * (1 - est("test", conf(b), "rmse_norm_mean") / est("test", "conformal_naive", "rmse_norm_mean")) for b in LEARNED]
    c(s, "ID reduction vs persistence (%)", "32--36", rng(red, 0), "by 32--36\\%")
    c(s, "ID surge RMSE, ridge", "0.118", f"{est('test', 'conformal_ridge', 'rmse_u'):.3f}")
    c(s, "ID surge RMSE, networks", "0.147--0.184", rng([est("test", conf(b), "rmse_u") for b in NETS], 3))
    c(s, "ID roll-rate RMSE, every learned model", "0.0052", rng([est("test", conf(b), "rmse_p") for b in LEARNED], 4))
    c(s, "OOD normalised RMSE, learned", "0.927--1.006", rng([est("ood_test", conf(b), "rmse_norm_mean") for b in LEARNED], 3))
    c(s, "OOD normalised RMSE, persistence", "1.243", f"{est('ood_test', 'conformal_naive', 'rmse_norm_mean'):.3f}")
    red = [100 * (1 - est("ood_test", conf(b), "rmse_norm_mean") / est("ood_test", "conformal_naive", "rmse_norm_mean"))
           for b in LEARNED]
    c(s, "OOD reduction vs persistence (%)", "19--25", rng(red, 0), "by only 19--25\\%")
    f = {ch: est("ood_test", "conformal_naive", f"rmse_{ch}") / est("test", "conformal_naive", f"rmse_{ch}")
         for ch in ["u", "v", "p", "r", "phi"]}
    c(s, "persistence OOD/ID factor, roll rate (smallest)", "1.3", f"{f['p']:.1f}", "factor of 1.3 (roll rate)")
    c(s, "persistence OOD/ID factor, sway (largest)", "5.3", f"{f['v']:.1f}", "5.3 (sway velocity")
    c(s, "smallest/largest factor channels", "p/v", f"{min(f, key=f.get)}/{max(f, key=f.get)}", "factor of 1.3 (roll rate)")
    for m, pr in (("lstm", "0.72"), ("naive", "0.75"), ("ridge", "0.52"), ("linear", "0.55")):
        c(s, f"OOD surge RMSE, {m}", pr, f"{est('ood_test', conf(m), 'rmse_u'):.2f}")

    # ---- in-distribution intervals (Section 3.2)
    s = "3.2 ID intervals"
    scp = [conf(b) for b in LEARNED] + [f"{f}_{b}" for f in SPREAD for b in ("lstm", "mlp")]
    c(s, "ID coverage, all SCP with learned forecasters (%)", "92.6--93.5", rng([est("test", m, "coverage") for m in scp], 1, 100))
    c(s, "ID coverage, persistence (%)", "91.8", f"{100 * est('test', 'conformal_naive', 'coverage'):.1f}")
    c(s, "every recording-level ID interval excludes 90%", "True",
      str(all(S["test"].loc[(m, "coverage"), "ci_low"] > 0.9 for m in scp + ["conformal_naive"])),
      "every recording-level interval excludes 90\\%")
    o = O1[(O1.split == "test") & (O1.level == 0.9) & O1.construction.str.contains("conformal")]
    c(s, "coverage drop step 1 to 30 (pp)", "two to three", " to ".join({2: "two", 3: "three"}.get(round(100 * x), str(round(100 * x)))
                                   for x in ((o.cov_step1 - o.cov_step30).min(), (o.cov_step1 - o.cov_step30).max())),
      "about two to three percentage points")
    four = [conf("lstm"), conf("mlp")] + [f"{f}_{b}" for f in SPREAD for b in ("lstm", "mlp")]
    c(s, "ID surge coverage, 4 SCP x 2 backbones (%)", "94--96", rng([est("test", m, "coverage_u") for m in four], 0, 100))
    c(s, "ID sway coverage (%)", "98.5--99.1", rng([est("test", m, "coverage_v") for m in four], 1, 100))
    dev = max(abs(est("test", m, f"coverage_{ch}") - 0.9) for m in four for ch in ("p", "r", "phi"))
    c(s, "p, r, phi within 1.3 points of nominal (max |dev|, pp)", "1.3", f"{100 * dev:.1f}",
      "within 1.3 percentage points")
    # "93.2--93.7" (uncalibrated Gaussian bands, ID) left the running text in v5.2 (Section 3.2 now gives the
    # noise explanation); the values remain in Table uq-id, which report_pub checks.
    c(s, "uncalibrated ensemble bands ID (%)", "60--61", rng([est("test", f"ensemble_{b}_raw", "coverage") for b in ("lstm", "mlp")], 0, 100),
      "ensemble bands cover 60--61\\%")
    c(s, "uncalibrated dropout band ID, LSTM (%)", "26", f"{100 * est('test', 'mcd_lstm_raw', 'coverage'):.0f}", "26\\% (LSTM)")
    c(s, "uncalibrated dropout band ID, MLP (%)", "71", f"{100 * est('test', 'mcd_mlp_raw', 'coverage'):.0f}", "71\\% (MLP)")
    c(s, "GN-SCP ID width LSTM, MLP", "0.93 and 0.90",
      f"{est('test', 'gaussian_lstm', 'width_norm_mean'):.2f} and {est('test', 'gaussian_mlp', 'width_norm_mean'):.2f}")
    c(s, "DN-SCP ID width", "1.03--1.05", rng([est("test", f"mcd_{b}", "width_norm_mean") for b in ("lstm", "mlp")], 2))
    c(s, "EN-SCP ID width", "1.17--1.20", rng([est("test", f"ensemble_{b}", "width_norm_mean") for b in ("lstm", "mlp")], 2))
    c(s, "GN-SCP ID interval score", "1.35 and 1.30",
      f"{est('test', 'gaussian_lstm', 'iscore_norm_mean'):.2f} and {est('test', 'gaussian_mlp', 'iscore_norm_mean'):.2f}")
    other = [conf(b) for b in LEARNED] + [f"{f}_{b}" for f in ("ensemble", "mcd") for b in ("lstm", "mlp")]
    c(s, "ID interval score, other SCP with learned forecasters", "1.49--1.64", rng([est("test", m, "iscore_norm_mean") for m in other], 2))
    r = S["test"].loc[(conf("lstm"), "width_norm_mean")]
    c(s, "degenerate AR-SCP width interval (LSTM)", "[1.01, 1.01]", f"[{r.ci_low:.2f}, {r.ci_high:.2f}]")
    o = O1[(O1.split == "test") & (O1.level == 0.9)]
    c(s, "simultaneous trajectory coverage, ID (Table O1)", "0.16--0.40", rng(o.traj_all, 2))
    c(s, "at most 40% of trajectories covered (implications)", "40", f"{100 * o.traj_all.max():.0f}", "at most 40\\%")

    # ---- OOD intervals (Section 3.3)
    s = "3.3 OOD intervals"
    c(s, "AR-SCP OOD coverage, every point forecaster (%)", "50--53",
      rng([est("ood_test", conf(b), "coverage") for b in LEARNED + ["naive"]], 0, 100), "covers 50--53\\%")
    c(s, "ID-to-OOD coverage loss, learned (pp)", "41--43",
      rng([est("test", conf(b), "coverage") - est("ood_test", conf(b), "coverage") for b in LEARNED], 0, 100))
    for m, pr in (("ensemble_lstm", "76.2"), ("ensemble_mlp", "77.7"), ("mcd_lstm", "70.4"), ("mcd_mlp", "74.3"),
                  ("gaussian_lstm", "49.9"), ("gaussian_mlp", "48.2")):
        c(s, f"OOD coverage {m} (%)", pr, f"{100 * est('ood_test', m, 'coverage'):.1f}")
    wr = [est("ood_test", f"{f}_{b}", "width_norm_mean") / est("ood_test", conf(b), "width_norm_mean")
          for f in ("ensemble", "mcd") for b in ("lstm", "mlp")]
    c(s, "EN/DN width relative to AR-SCP, OOD", "1.9--2.4", rng(wr, 1), "1.9--2.4 times as wide")
    c(s, "EN/DN OOD width", "1.90--2.42", rng([est("ood_test", f"{f}_{b}", "width_norm_mean") for f in ("ensemble", "mcd") for b in ("lstm", "mlp")], 2))
    c(s, "AR-SCP OOD width (LSTM, MLP)", "1.00--1.01", rng([est("ood_test", conf(b), "width_norm_mean") for b in ("lstm", "mlp")], 2))
    g = [est("ood_test", f"{f}_{b}_raw", "width_norm_mean") / est("test", f"{f}_{b}_raw", "width_norm_mean")
         for f in ("ensemble", "mcd") for b in ("lstm", "mlp")]
    c(s, "raw ensemble/dropout band width growth OOD/ID", "1.8--2.1", rng(g, 1))
    c(s, "EN/DN OOD interval score", "5.22--5.69", rng([est("ood_test", f"{f}_{b}", "iscore_norm_mean") for f in ("ensemble", "mcd") for b in ("lstm", "mlp")], 2))
    c(s, "AR-SCP OOD interval score, learned", "7.32--7.97", rng([est("ood_test", conf(b), "iscore_norm_mean") for b in LEARNED], 2))
    miss = lambda m: est("ood_test", m, "iscore_norm_mean") - est("ood_test", m, "width_norm_mean")
    c(s, "EN/DN OOD miss penalty", "2.96--3.75", rng([miss(f"{f}_{b}") for f in ("ensemble", "mcd") for b in ("lstm", "mlp")], 2))
    c(s, "AR-SCP OOD miss penalty, learned", "6.35--6.94", rng([miss(conf(b)) for b in LEARNED], 2))
    c(s, "GN-SCP OOD width LSTM, MLP", "1.00 and 0.87",
      f"{est('ood_test', 'gaussian_lstm', 'width_norm_mean'):.2f} and {est('ood_test', 'gaussian_mlp', 'width_norm_mean'):.2f}")
    for b, pr in (("lstm", "from 1.00 to 1.07"), ("mlp", "from 0.94 to 0.91")):
        c(s, f"raw Gaussian band width ID to OOD, {b}", pr,
          f"from {est('test', f'gaussian_{b}_raw', 'width_norm_mean'):.2f} to {est('ood_test', f'gaussian_{b}_raw', 'width_norm_mean'):.2f}")
    c(s, "uncalibrated bands OOD coverage (%)", "12--52",
      rng([est("ood_test", f"{f}_{b}_raw", "coverage") for f in SPREAD for b in ("lstm", "mlp")], 0, 100), "cover 12--52\\%")
    oo = O1[(O1.split == "ood_test") & (O1.level == 0.9)].set_index("construction")
    en = [k for k in oo.index if "LSTM" in k and ("ensemble" in k.lower() or "EN" in k)]
    gn = [k for k in oo.index if "MLP" in k and ("gaussian" in k.lower() or "GN" in k) and "raw" not in k.lower() and "uncal" not in k.lower()]
    if en:
        c(s, "EN-SCP (LSTM) OOD coverage across repeats (%)", "75.2 to 77.7",
          f"{100 * oo.loc[en[0], 'coverage_seed_min']:.1f} to {100 * oo.loc[en[0], 'coverage_seed_max']:.1f}")
    if gn:
        c(s, "GN-SCP (MLP) OOD coverage across repeats (%)", "44.6 to 50.7",
          f"{100 * oo.loc[gn[0], 'coverage_seed_min']:.1f} to {100 * oo.loc[gn[0], 'coverage_seed_max']:.1f}")
        c(s, "GN-SCP (MLP) OOD seed range (pp)", "6.1",
          f"{100 * (oo.loc[gn[0], 'coverage_seed_max'] - oo.loc[gn[0], 'coverage_seed_min']):.1f}", "6.1-point range")

    # ---- paired comparisons (Section 3.4)
    s = "3.4 Paired comparisons"
    d = lambda sp, a, b_, st="coverage": P[sp].loc[(a, b_, st), "difference"]
    cal = lambda sp: [100 * d(sp, f"{f}_{b}", f"{f}_{b}_raw") for f in ("ensemble", "mcd") for b in ("lstm", "mlp")]
    c(s, "calibration gain ens/dropout, ID (pp)", "22.1--67.8", rng(cal("test"), 1))
    c(s, "calibration gain ens/dropout, OOD (pp)", "29.5--58.2", rng(cal("ood_test"), 1))
    gcal = [-100 * d(sp, f"gaussian_{b}", f"gaussian_{b}_raw") for sp in ("test", "ood_test") for b in ("lstm", "mlp")]
    c(s, "calibration reduces GN coverage (pp)", "0.5--1.7", rng(gcal, 1))
    c(s, "GN calibration OOD interval-score change, LSTM / MLP", "0.29 (LSTM) and 0.15",
      f"{d('ood_test', 'gaussian_lstm', 'gaussian_lstm_raw', 'iscore_norm_mean'):.2f} (LSTM) and "
      f"{d('ood_test', 'gaussian_mlp', 'gaussian_mlp_raw', 'iscore_norm_mean'):.2f}")
    bb = [100 * d("ood_test", f"{f}_mlp", f"{f}_lstm") for f in ["conformal"] + SPREAD]
    c(s, "MLP minus LSTM OOD coverage (pp)", "$-1.7$ to $+4.0$", f"${min(bb):.1f}$ to $+{max(bb):.1f}$".replace("$-", "$-"))
    sp_ar = [100 * d("ood_test", f"{f}_{b}", conf(b)) for f in ("ensemble", "mcd") for b in ("lstm", "mlp")]
    c(s, "EN/DN minus AR-SCP OOD (pp)", "19--25", rng(sp_ar, 0), "by 19--25 points")
    c(s, "GN minus AR-SCP OOD, LSTM / MLP (pp)", "1.6 (LSTM) and 4.1",
      f"{-100 * d('ood_test', 'gaussian_lstm', conf('lstm')):.1f} (LSTM) and {-100 * d('ood_test', 'gaussian_mlp', conf('mlp')):.1f}")

    # ---- post hoc matched comparison (Section 3.4, Table 9)
    s = "3.4 Matched (post hoc)"
    mt = MT.set_index(["backbone", "family", "split"])
    def mcov(bb, fam, kind, sp="ood_test", col="coverage"):
        g = MT[(MT.backbone == bb) & (MT.family == fam) & (MT.split == sp) & MT.construction.str.contains(kind, regex=False)]
        return g[col].iloc[0]
    for bb, a, b in (("LSTM", "50.6", "76.2"), ("MLP", "51.6", "77.7")):
        c(s, f"ensemble mean, AR -> EN score OOD, {bb}", f"from {a} to {b}",
          f"from {100 * mcov(bb, 'ens', 'abs-residual'):.1f} to {100 * mcov(bb, 'ens', 'spread-normalised'):.1f}")
    for bb, a, b in (("LSTM", "1.55", "1.63"), ("MLP", "1.54", "1.61")):
        c(s, f"ensemble mean, ID interval score AR -> EN, {bb}", f"from {a} to {b}",
          f"from {mcov(bb, 'ens', 'abs-residual', 'test', 'iscore_norm'):.2f} to {mcov(bb, 'ens', 'spread-normalised', 'test', 'iscore_norm'):.2f}")
    for bb, a, b in (("LSTM", "50.9", "70.4"), ("MLP", "50.8", "74.3")):
        c(s, f"dropout mean, AR -> DN score OOD, {bb}", f"from {a} to {b}",
          f"from {100 * mcov(bb, 'mcd', 'abs-residual'):.1f} to {100 * mcov(bb, 'mcd', 'spread-normalised'):.1f}")
    c(s, "Gaussian mean: GN vs AR score OOD", "49.9 and 48.2\\% against 50.8 and 51.7",
      f"{100 * mcov('LSTM', 'gauss', 'spread-normalised'):.1f} and {100 * mcov('MLP', 'gauss', 'spread-normalised'):.1f}\\% against "
      f"{100 * mcov('LSTM', 'gauss', 'abs-residual'):.1f} and {100 * mcov('MLP', 'gauss', 'abs-residual'):.1f}")
    lessn = [100 * (mcov(bb, "single", "abs-residual") - mcov(bb, fam, "abs-residual")) for bb in ("LSTM", "MLP") for fam in ("ens", "mcd")]
    c(s, "ensemble/dropout means cover less than single models (AR score, OOD, pp)", "0.6--1.5", rng(lessn, 1))

    # ---- channel discussion (Section 5.2) and data
    s = "5.2 / 2.1 Data-based statements"
    T1 = pd.read_csv("experiments/analysis/v3/supervisor_checks/table1_recomputed.csv").set_index(["split", "channel"])
    sd = lambda sp, ch: T1.loc[(sp, ch), "sd_sample"]
    c(s, "roll-rate SD train -> OOD", "0.007 to 0.008", f"{sd('train', 'p'):.3f} to {sd('ood', 'p'):.3f}")
    c(s, "yaw-rate SD train -> OOD", "0.012 to 0.026", f"{sd('train', 'r'):.3f} to {sd('ood', 'r'):.3f}")
    c(s, "roll-angle SD factor", "1.7", f"{sd('ood', 'phi') / sd('train', 'phi'):.1f}", "grows by a factor of 1.7")
    c(s, "mean surge rise (%, to the nearest 10)", "about 50", f"about {round(100 * (T1.loc[('ood', 'u'), 'mean'] / T1.loc[('train', 'u'), 'mean'] - 1), -1):.0f}",
      "rises by about 50\\%")
    c(s, "ID SD of roll rate", "0.006", f"{sd('test', 'p'):.3f}", "0.006~rad/s standard deviation")
    c(s, "AR-SCP OOD roll-rate coverage (%)", "75", rng([est("ood_test", conf(b), "coverage_p") for b in ("lstm", "mlp")], 0, 100),
      "keeps 75\\% coverage")
    c(s, "AR-SCP OOD yaw-rate coverage (%)", "39--42", rng([est("ood_test", conf(b), "coverage_r") for b in ("lstm", "mlp")], 0, 100))
    c(s, "shaft speed above training max (%)", "39", f"{100 * (T1.loc[('ood', 'n'), 'max'] / T1.loc[('train', 'n'), 'max'] - 1):.0f}",
      "by up to 39\\%")
    ratios = [sd("ood", ch) / sd("train", ch) for ch in ("deltal", "deltar", "v", "r", "phi")]
    c(s, "SD ratios of azimuths, v, r, phi", "1.4--2.2", rng(ratios, 1), "are 1.4--2.2 times larger")
    gL = np.sqrt(9.81 * 51.5)
    c(s, "Froude numbers service / max train / max OOD", "0.34, 0.41, 0.60",
      f"{7.71 / gL:.2f}, {T1.loc[('train', 'u'), 'max'] / gL:.2f}, {T1.loc[('ood', 'u'), 'max'] / gL:.2f}",
      "0.34 at the service speed, 0.41 at the maximum training speed and 0.60")
    # Perez et al. (2006), Sec. 3.2, give u_nom = 7.71 m/s (15 kt); the exact conversion is 7.72.
    c(s, f"service speed in m/s as stated by perez2006patrol (15 kn converts to {15 * 1852 / 3600:.2f})", "7.71",
      "7.71", "15~kn (7.71~m/s)")
    c(s, "windows per recording", "3542", f"{3601 - 60 + 1}", "3601-60+1=3542")
    c(s, "calibration windows", "31{,}878", f"{9 * 3542:,}".replace(",", "{,}"), "n_{\\mathrm{cal}}=31{,}878")
    c(s, "horizon in roll periods", "4.4", f"{30 / 6.76:.1f}", "about 4.4 roll periods")
    c(s, "ID windows (example-window rule)", "106{,}260", f"{30 * 3542:,}".replace(",", "{,}"))

    # ---- implications and conclusions
    s = "5.3 / 6 Implications, conclusions"
    c(s, "raw ensemble/dropout bands ID (%)", "26--71",
      rng([est("test", f"{f}_{b}_raw", "coverage") for f in ("ensemble", "mcd") for b in ("lstm", "mlp")], 0, 100), "covered 26--71\\%")
    c(s, "EN/DN OOD coverage (%)", "70--78", rng([est("ood_test", f"{f}_{b}", "coverage") for f in ("ensemble", "mcd") for b in ("lstm", "mlp")], 0, 100),
      "retained 70--78\\%")
    bb24 = pd.read_csv(f"{PH5}/backbones_2.4/summary.csv")
    bb24 = bb24[(bb24.level == 0.9) & (bb24.split == "ood_test") & (bb24.kind == "X") & bb24.construction.isin(["EN-SCP", "DN-SCP"])]
    c(s, "EN/DN OOD coverage on five backbones (%)", "69--80", rng(bb24.coverage, 0, 100), "69--80\\% on all five backbones")
    c(s, "no construction reached 80% (max OOD SCP coverage < 0.80)", "True",
      str(max(est("ood_test", m, "coverage") for m in scp) < 0.8 and bb24.coverage.max() < 0.8), "no construction reached 80\\%")
    ew = [100 * (est("test", f"ensemble_{b}", "width_norm_mean") / est("test", conf(b), "width_norm_mean") - 1) for b in ("lstm", "mlp")]
    c(s, "EN-SCP ID width relative to AR-SCP (%)", "17--19", rng(ew, 0), "17--19\\% wider")
    w5 = bb24.set_index(["backbone", "construction"]).width_norm
    ar5 = pd.read_csv(f"{PH5}/backbones_2.4/summary.csv")
    ar5 = ar5[(ar5.level == 0.9) & (ar5.split == "ood_test") & (ar5.kind == "X") & (ar5.construction == "AR-SCP")].set_index("backbone").width_norm
    c(s, "EN/DN width relative to AR-SCP, five backbones", "1.9--2.6", rng([w5[k] / ar5[k[0]] for k in w5.index], 1), "1.9--2.6 times as wide")

    # ---- timing pilot, GPU, training epochs
    s = "Training, timing, computation"
    pr = json.load(open("experiments/manifests/pilot_2026-09-28/pilot_report.json"))["evals"]
    c(s, "MC dropout inference on calibration windows (s)", "23", f"{pr['mcd_lstm']['timing']['cal']['inference_s']:.0f}", "recorded 23~s")
    c(s, "single LSTM inference (s)", "1.2", f"{pr['point_lstm']['timing']['cal']['inference_s']:.1f}", "1.2~s for one pass")
    led = pd.read_csv("experiments/manifests/gpu_ledger.tsv", sep="\t")
    c(s, "benchmark GPU-hours", "1.55", f"{led[led.scope == 'benchmark'].allocated_gpu_s.sum() / 3600:.2f}", "1.55 GPU-hours")
    # The 0.08 GPU-hour timing pilot is no longer stated (Appendix A.5 shortened, internal review comment of 5 Oct).
    c(s, "MC dropout time per forecast, batched (ms)", "0.7", f"{pr['mcd_lstm']['timing']['cal']['inference_s'] / 31878 * 1000:.1f}", "about 0.7~ms in a batch")
    logs = os.environ.get("STAGEB_LOGS", "")
    if logs:
        be = lambda pat: [json.load(open(p))["best_epoch"] for p in glob.glob(os.path.join(logs, pat, "**", "train_log.json"), recursive=True)]
        rec = be("lstm_ens") + be("gru") + be("lstm_dropout") + be("tcn") + be("lstm_gaussian")
        c(s, "best epoch: recurrent, TCN, Gaussian LSTM", "2--5", rng(rec, 0))
        c(s, "best epoch: MLP and Gaussian MLP", "3--10", rng(be("mlp_ens") + be("mlp_gaussian"), 0))
        c(s, "best epoch: MLP with dropout", "16--32", rng(be("mlp_dropout"), 0))
        c(s, "best epoch: linear (SGD)", "22--49", rng(be("linear"), 0))
        stops = {json.load(open(p))["stopped_by"] for p in glob.glob(os.path.join(logs, "**", "train_log.json"), recursive=True)}
        c(s, "every trained run ended by early stopping", "early_stopping", ",".join(sorted(stops)), "Every trained run ended by early stopping")

    # ---- appendix: elapsed time, floor, calibration draw
    s = "Appendix"
    fd = json.load(open("experiments/manifests/stageA_2026-09-28/feature_decision.json"))
    c(s, "elapsed-time ablation: validation loss increase LSTM, MLP (%)", "0.19\\% and 0.35",
      f"{-100 * fd['gain']['lstm']:.2f}\\% and {-100 * fd['gain']['mlp']:.2f}")
    fl = pd.read_csv("experiments/analysis/v3/floor_sensitivity.csv")
    prim = fl[fl.primary]
    c(s, "max fraction of calibration spreads floored", "6.3\\times10^{-6}", f"{prim.max_frac_floored_cal.max():.1e}".replace("e-06", "\\times10^{-6}"))
    alt = fl[~fl.primary]
    c(s, "floor factor change: max coverage change (pp)", "0.25", f"{100 * alt.d_coverage_vs_primary.abs().max():.2f}", "at most 0.25 percentage points")
    c(s, "floor factor change: max width change", "0.05", f"{alt.d_width_norm_vs_primary.abs().max():.2f}", "normalised width by at most 0.05")
    cs = json.load(open("experiments/manifests/calibration_selection.json"))
    rep = pd.read_csv("experiments/manifests/calibration_representativeness.csv")
    rr = rep[rep.selection == "random"].set_index("variable").smd
    c(s, "SMDs above 0.5 (random draw)", "five", {5: "five"}.get(int((rr.abs() > 0.5).sum()), str(int((rr.abs() > 0.5).sum()))), "five SMDs exceeded 0.5")
    for v, pr_ in (("deltal_std", "0.84"), ("u_mean", "$-0.52$"), ("deltar_std", "0.60"), ("r_mean", "$-0.59$"), ("r_std", "0.52")):
        val = f"{rr[v]:.2f}"
        c(s, f"SMD {v}", pr_, f"${val}$" if rr[v] < 0 else val)
    ch = cs["chance_reference"]
    c(s, "criterion passes by chance (%)", "7", f"{100 * ch['p_criterion_passes_by_chance']:.0f}", "held in 7\\% of cases")
    c(s, "median largest |SMD| over 5000 draws", "0.75", f"{ch['max_abs_smd_quantiles_5_50_95'][1]:.2f}")
    c(s, "percentile of the present draw", "68th", f"{cs['random']['max_abs_smd_chance_percentile']:.0f}th", "the 68th percentile")
    c(s, "stratified backup largest |SMD|", "0.83", f"{cs['stratified_n']['max_abs_smd']:.2f}", "largest $|\\text{SMD}|$ 0.83")

    # post-evaluation comparison with the ID test recordings (deposit files; same summaries as the selection check)
    def table(split):
        rows = []
        for rec in load_split_recordings(args.base, split):
            row = {"file": rec["source_file"].iloc[0]}
            for v in VARS:
                row[f"{v}_mean"], row[f"{v}_std"] = rec[v].mean(), rec[v].std()
            rows.append(row)
        return pd.DataFrame(rows).set_index("file")
    if os.path.isdir(args.base):
        tr, te = table("train"), table("test")
        calf = cs["random"]["files"]
        fit, calt = tr.drop(index=calf), tr.loc[calf]
        smd_fit_test = (fit.mean() - te.mean()) / fit.std()
        c(s, "fit vs ID test: summaries with |SMD| <= 0.5", "21 of the 22", f"{int((smd_fit_test.abs() <= 0.5).sum())} of the {len(smd_fit_test)}")
        c(s, "exception: within-recording SD of Vw", "Vw_std", smd_fit_test.abs().idxmax(), "The exception is the within-recording standard deviation of $V_w$")
        c(s, "SMD of Vw SD (fit vs ID test)", "2.8", f"{abs(smd_fit_test['Vw_std']):.1f}", "its SMD is 2.8")
        c(s, "Vw within-recording SD, ID test vs fit (m/s)", "0.21 against 0.23", f"{te['Vw_std'].mean():.2f} against {fit['Vw_std'].mean():.2f}")
        sct = (calt.mean() - te.mean()) / fit.std()
        c(s, "cal vs ID test SMD, mean u", "$-0.62$", f"${sct['u_mean']:.2f}$", "$\\text{SMD}=-0.62$")
        c(s, "cal vs ID test SMD, SD of delta_l", "0.71", f"{sct['deltal_std']:.2f}", "0.71 for the standard deviation of $\\delta_l$")

    posthoc_v52_checks(c)

    df = pd.DataFrame(c.rows)
    bad = df[~df.ok]
    with open(args.out, "w") as fh:
        fh.write(f"# Number check: {args.tex}\n\nGenerated by `scripts/verify_manuscript_numbers.py`. Each row compares a number as printed "
                 "in the running text with the value recomputed from the stored outputs (same rounding). Generated tables and "
                 "prose macros are checked by `src/analysis/report_pub.py` and its tests; literature values are in "
                 "`citation_check.md`.\n\n")
        fh.write(f"**{int(df.ok.sum())} of {len(df)} checks agree.**\n\n")
        if len(bad):
            fh.write("## Disagreements\n\n" + md_table(bad) + "\n\n")
        fh.write("## All checks\n\n" + md_table(df) + "\n")
    print(f"{int(df.ok.sum())}/{len(df)} agree")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
