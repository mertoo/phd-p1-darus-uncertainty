# Uncertainty quantification for multi-step ship motion forecasting under an operating-condition shift

Code, configuration, split manifests and analysis outputs for the benchmark in

> V. M. Asan, M. Kõrgesaar, K. Tabri, F. Roshan. *A controlled benchmark of uncertainty quantification for multi-step ship motion forecasting under an operating-condition shift.* Submitted to Ocean Engineering.

The benchmark compares seven point forecasters (persistence, ridge, linear model trained by SGD, MLP, LSTM, GRU, TCN) and four interval constructions on the public DaRUS patrol-vessel dataset. The constructions are absolute-residual split conformal (AR-SCP) and conformal intervals normalised by a deep-ensemble (EN-SCP), MC-dropout (DN-SCP) or Gaussian (GN-SCP) spread. Forecasts cover 30 s of five manoeuvring states (u, v, p, r, φ) from a 30-s history of 11 input channels. Coverage is evaluated in distribution (30 recordings) and under a shift to faster, harder-steered operation (29 recordings), with whole held-out recordings for calibration and recording-level bootstrap intervals.

The per-window evaluation outputs (sufficient statistics for every coverage, width and bootstrap number in the paper) are archived on Zenodo: **DOI: to be added**.

## Data

The data are the DaRUS deposit of Baier and Staab (2022), *A Simulated 4-DOF Ship Motion Dataset for System Identification under Environmental Disturbances*, [doi:10.18419/darus-2905](https://doi.org/10.18419/darus-2905). They are not redistributed here. Download the deposit and arrange the CSV files as

```
data/processed/darus/
├── patrol_ship_routine/
│   ├── train/        # 57 recordings (48 fitting + 9 calibration)
│   ├── validation/   # 9 recordings
│   └── test/         # 30 recordings (ID test)
└── patrol_ship_ood/
    └── test/         # 29 recordings (OOD test)
```

that is, the `processed/<split>/` folders of the deposit without the `processed` level. The loader reads `.csv` and `.tab` files; the calibration recordings are listed by name (`.csv`) in `experiments/manifests/split_manifest.csv` and in the configs.

Note: the surge and sway velocities of all training recordings carry additive white noise of about 0.10 m/s, and those of 17 OOD recordings about 0.31 m/s, whereas the validation and ID test recordings carry none. The noise is present in the deposited files and is not described in the dataset documentation. `scripts/check_measurement_noise.py` reproduces the check.

## Environment

Python 3.11 with the packages in `requirements.txt` (install the PyTorch build that matches your CUDA version first). `scripts/slurm/v3/setup_env.sh` builds the environment used for the paper on a Slurm cluster and runs the tests:

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests
```

## Reproducing the results

All commands run from the repository root with `PYTHONPATH=.`.

1. **Train** (GPU, Slurm array; one row of `scripts/slurm/v3/run_list.tsv` per task, stages `1a`–`1e`, `5a`, `5b`; stage `0`/`0b` is the predeclared input ablation):
   ```bash
   sbatch --array=1-30 scripts/slurm/v3/train_array.sh 1a
   ```
   Ridge regression and persistence run on CPU (`sbatch scripts/slurm/v3/cpu_baselines.sh`). Configurations are in `experiments/configs/v3/`; seeds and run names are in `run_list.tsv`.
2. **Evaluate** every construction on the calibration, ID-test and OOD-test windows (`scripts/slurm/v3/eval_list.tsv`, one row per task):
   ```bash
   sbatch --array=1-35 scripts/slurm/v3/eval_array.sh
   ```
   This writes `experiments/eval/v3/<tag>/` with `metrics.json`, `predictions_<split>.npz`, `windows_<split>.csv` and `per_window_<split>_<method>_<variant>_90.csv`. The Zenodo archive holds the files of the primary variant (`horizon_channel`, 90%), the window metadata and the metrics.
3. **Recording-level bootstrap and tables**:
   ```bash
   python -m src.analysis.bootstrap --spec experiments/configs/v3/bootstrap_full.yaml --out experiments/analysis/v3/bootstrap
   python -m src.analysis.report_pub --eval_root experiments/eval/v3 --out experiments/analysis/v3/report_pub
   ```
4. **Post hoc analyses** (each script states its protocol in the docstring; outputs are already in `experiments/analysis/v3/`):
   - `scripts/check_measurement_noise.py`, `scripts/noise_group_coverage.py`: measurement noise and coverage by noise group
   - `scripts/posthoc_clean_calibration.py`, `scripts/summarise_clean_calibration.py`: recalibration on the clean validation recordings
   - `scripts/per_split_scale_errors.py`: errors on each split's own scale
   - `scripts/posthoc_novelty_normaliser.py`: non-learned novelty normaliser (NV-SCP)
   - `scripts/posthoc_mc_passes.py`: stability of the MC-dropout spread with the number of passes
   - `scripts/posthoc_v5.py`, `scripts/posthoc_v5_backbones.py`: width-matched contrasts and additional backbones

   The post hoc scripts read `predictions_<split>.npz`, which are not in the Zenodo archive because of their size (about 12 GB); they are regenerated by step 2 or available from the authors on request.

## Repository layout

```
src/
  data_loading/   DaRUS parser, recording-aware windowing, standardisation, calibration draw
  models/         LSTM/GRU seq2seq, TCN, MLP, linear, Gaussian heads, persistence
  training/       training loop with early stopping; closed-form ridge
  uncertainty/    conformal and spread-normalised intervals, MC dropout
  evaluation/     benchmark evaluation, interval metrics, checkpoint provenance
  analysis/       recording-level bootstrap, report and table generation
scripts/          post hoc analyses, checks, Slurm job scripts (scripts/slurm/v3)
experiments/
  configs/v3/     model and analysis configurations
  manifests/      split manifest, calibration draw, data audit, stage gates and training logs
  analysis/v3/    bootstrap, report and post hoc outputs used in the paper
tests/            unit tests
EXPERIMENT_PROTOCOL.md   predeclared protocol
RESULTS_CHANGELOG.md     comparison with an earlier version of the results
```

Legacy files from an earlier phase of the project (for example `experiments/configs/p1_*.yaml`, `src/evaluation/run_eval.py`, `train_all.sh`) are kept for provenance and are not used for the paper.

## Citation

If you use this code, please cite the paper above and the DaRUS dataset:

```bibtex
@dataset{baier2022darus,
  author    = {Baier, Alexandra and Staab, Steffen},
  title     = {A Simulated 4-{DOF} Ship Motion Dataset for System Identification under Environmental Disturbances},
  year      = {2022},
  publisher = {DaRUS Data Repository, University of Stuttgart},
  doi       = {10.18419/darus-2905}
}
```


## Licence

Code: MIT (see `LICENSE`). Evaluation outputs archived on Zenodo: CC BY 4.0.
