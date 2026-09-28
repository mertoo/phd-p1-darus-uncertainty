# Reproducing the tables and figures of `main_v3.tex`

## Versions
| Step | Code version | Record |
|---|---|---|
| Feature ablation (Stage A) | `dde1848` | `experiments/manifests/stageA_2026-09-28/` |
| Training, evaluation, analysis (Stages B–D) | `83585da` | `experiments/manifests/stageB_2026-09-28/`, `stageC_2026-09-28/` |
| Results archive | `630baad` | `experiments/analysis/v3/`, `RESULTS_CHANGELOG.md` |
| Manuscript tables/figures, post hoc analyses | `eaa44a6` | `experiments/analysis/v3/report_pub/manifest.json` |

`83585da` and `dde1848` have identical training, data, model and evaluation code.

## Environment
- Cluster runs: `~/envs/darus-v3` (Python 3.11.16, torch 2.5.1+cu121). Exact package list: `experiments/manifests/pilot_2026-09-28/env_darus-v3_pip_freeze.txt`.
- Analysis and table generation: any Python ≥ 3.10 with numpy, pandas, matplotlib, PyYAML; torch is needed only to compute parameter counts.

## Inputs available in the repository
- Split and calibration manifests: `experiments/manifests/split_manifest.csv`, `data_audit.json`, `calibration_selection.json`
- Configurations: `experiments/configs/v3/*.yaml` (+ `bootstrap_full.yaml`, `analysis_rules.yaml`)
- Per-evaluation metrics (35 files): `experiments/manifests/stageC_2026-09-28/metrics/*.json`
- Bootstrap outputs: `experiments/analysis/v3/bootstrap/`
- Floor sensitivity: `experiments/analysis/v3/floor_sensitivity.csv`
- Post hoc analyses: `experiments/analysis/v3/posthoc/`
- Seeds, best epochs and step counts: training logs in `experiments/manifests/stageA_2026-09-28/train_logs/` and `stageB_2026-09-28/train_logs.tgz`

## Inputs NOT in the repository
- Model checkpoints and full prediction arrays (`experiments/runs/v3`, `experiments/eval/v3`, about 16 GB) remain on the TalTech HPC cluster and are not publicly archived.
- The bootstrap (`src.analysis.bootstrap`), floor sensitivity and the regenerated example-window figure read per-window files or predictions from `experiments/eval/v3`, so they can be re-run only with those arrays. Figure 3 needs only `metrics.json`, `predictions_{cal,test}.npz` and `windows_{cal,test}.csv` of `lstm_{single,ens,mcd,gauss}_rep0` (about 560 MB), plus the DaRUS recording `patrol_ship_routine/test/20190805-112459.csv`.

## Commands
```bash
# 0. Post hoc descriptive analyses from stored metrics (reproduces experiments/analysis/v3/posthoc exactly)
python -m scripts.posthoc_analyses

# 1. Publication tables, figures and the numbers quoted in the text (generated/prose_values.tex)
#    from the archived outputs (all 9 tables; Figs 1-2;
#    Fig 3 is regenerated when experiments/eval/v3 predictions are present, otherwise the
#    frozen figure is copied and the manifest records that)
python -m src.analysis.report_pub --out experiments/analysis/v3/report_pub

# 2. Copy into the manuscript folder with checksums
scripts/sync_paper_assets.sh

# 3. Compile (Overleaf or local TeX Live): main_v3.tex with bibtex, 3 LaTeX passes
```
Steps that need the cluster arrays (recorded; not needed to rebuild the paper):
```bash
python -m src.analysis.bootstrap --spec experiments/configs/v3/bootstrap_full.yaml --out experiments/analysis/v3/bootstrap
python -m src.analysis.floor_sensitivity --evals "experiments/eval/v3/*_ens_rep*" "experiments/eval/v3/*_mcd_rep*" "experiments/eval/v3/*_gauss_rep*" --out experiments/analysis/v3/floor_sensitivity.csv
```
