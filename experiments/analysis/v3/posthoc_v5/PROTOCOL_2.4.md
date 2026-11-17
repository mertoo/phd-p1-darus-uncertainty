# Post hoc analysis 2.4: spread-based constructions on further backbones (protocol)

Written 2026-10-30, **before any model of this analysis was trained**. Approval: Mert, 2026-10-30 (HPC session for 2.4, ≤ 3 GPU-hours; local commit before transfer to the cluster).
Post hoc, descriptive. It does not change any reported benchmark result; the AR-SCP results of GRU, TCN and the linear model in Tables 4–9 stay as they are.

## Question
The MLP was chosen as the second backbone of the spread-based constructions in preliminary experiments that used the OOD recordings. Do the main findings about the spread-based constructions also hold on backbones that were not chosen in that way?
- (F1) Under the shift, EN- and DN-SCP cover far more than AR-SCP on the same backbone (LSTM/MLP: +19 to +25 points).
- (F2) At the same mean width, uniformly widened AR intervals cover at least as much as EN- and DN-SCP (LSTM/MLP: Δ from −5.0 to −1.4 points under the shift).
- (F3) Uncalibrated ensemble and dropout bands under-cover in distribution.

## What is trained (stage `5a`, `5b` in `scripts/slurm/v3/run_list.tsv`)

| Backbone | Construction | Networks per repeat | Repeats | Config | Seeds |
|---|---|---|---|---|---|
| GRU, TCN, linear (SGD) | Deep ensemble (5 members; member 0 = single model for AR-SCP) | 5 | 3 | `gru`, `tcn`, `linear` (unchanged v3 configs) | 6000 + 100·rep + member |
| GRU, TCN | MC dropout (p = 0.2, 200 passes) | 1 | 3 | `gru_dropout`, `tcn_dropout` (v3 config with dropout 0.2, as `lstm_dropout`/`mlp_dropout`) | 6500 + rep |

Total: 51 trainings. Not included: Gaussian heads (no Gaussian GRU/TCN/linear model exists in the code and adding one would be a new architecture); MC dropout for the linear model (it has no hidden layer).
Where dropout acts: GRU between the two stacked layers of encoder and decoder (as the LSTM); TCN inside the residual blocks (`model.train()` at inference activates both, as for the LSTM and MLP).

## What is evaluated (tasks 36–59 in `scripts/slurm/v3/eval_list.tsv`)
`point` on member 0 (AR-SCP), `ensemble` on the five members (EN-SCP and raw band), `mc_dropout` (DN-SCP and raw band); same evaluator, calibration recordings, levels, floor and splits as the benchmark (`src.evaluation.benchmark_eval`, 200 passes, MC seed 0).

## Training and evaluation settings
Unchanged from the benchmark (protocol v3.0): no_time inputs, train-only standardisation, Adam 1e-3, batch 256, ≤ 100 epochs, early stopping (patience 10) on the validation recordings. No setting is chosen from test or OOD results.

## Code version and execution
The commit that contains this protocol, the configs and the lists is transferred to the cluster (git bundle; no push needed) and checked out there; job logs record the commit. Run names and evaluation tags are new; no existing checkpoint or evaluation is overwritten (checked before submission). Jobs: `train_array.sh 5a` (45 tasks), `train_array.sh 5b` (6 tasks), then `eval_array.sh` tasks 36–59. **Budget cap: 3.0 GPU-hours**; the expected use is about 1.3 GPU-hours (stages B and C of the benchmark: 51 trainings in 0.77 GPU-h, 35 evaluations in 0.72 GPU-h). A failed task is reported, not silently re-run.

## Analysis
For each backbone (GRU, TCN, linear; LSTM and MLP from the benchmark for comparison), mean over three repeats, ID and OOD test, nominal 90% (secondary 50/80/95%):
1. Coverage, normalised width and interval score of AR-SCP (member 0), EN-SCP, DN-SCP and the raw bands.
2. EN − AR and DN − AR coverage differences with 95% recording-level bootstrap intervals (2000 draws, seed 20261030, repeats averaged within draws).
3. Width-matched comparison exactly as in 2.1 (per-channel matching primary).

**Reading rule (fixed in advance):** a finding is said to hold on a backbone if its sign agrees and its recording-level interval excludes zero in the same direction as for the LSTM and MLP (F1: positive OOD difference; F2: Δ ≤ 0, i.e. the interval does not lie above 0; F3: raw ensemble/dropout coverage below 90% in distribution). The results are reported for every backbone whatever their direction.

## Outputs
`experiments/analysis/v3/posthoc_v5/backbones_2.4/` (CSV tables, run record with commit, job IDs and GPU seconds from `sacct`); GPU use added to `experiments/manifests/gpu_ledger.tsv`.

## Deviations (recorded during execution)
- 2026-10-31 10:31 (cluster time, UTC+2): jobs 1059621, 1059622, 1059623 had been pending since submission (02:55 cluster time) because both L40 GPUs of the only L40 node in the `gpu` partition (ada1) were held by other users' multi-day jobs. While still pending, the three jobs were moved with `scontrol update Partition=short`, which contains a second, idle L40 node (ada2; 4 h limit, the jobs request 15 min). Same GPU model, code, settings, seeds and resource request; no task had started before the change.
