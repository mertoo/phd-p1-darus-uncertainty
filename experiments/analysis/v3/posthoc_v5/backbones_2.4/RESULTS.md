# Post hoc analysis 2.4: results

Protocol: `../PROTOCOL_2.4.md` (written before training). Post hoc, descriptive. Inputs: 24 new evaluations (tasks 36–59, job 1059623) plus the stored LSTM/MLP evaluations; all 168 new files identical to the cluster copies (SHA-256). Rebuilt intervals reproduce the stored metrics (max abs. difference 3e-7). LSTM/MLP numbers reproduce analysis 2.1 (`../width_matched.csv`, max diff 0 for the matched quantities).
HPC: 51 trainings + 24 evaluations, 75/75 COMPLETED, 5208 allocated GPU-s (1.45 GPU-h; cap 3.0).

Nominal 90%, mean of three repeats; intervals are 95% recording-level bootstrap intervals (2000 draws, seed 20261030). Coverage in %.

## OOD test

| Backbone | AR-SCP cov. | EN-SCP cov. | EN − AR | EN − width-matched AR | DN-SCP cov. | DN − AR | DN − width-matched AR |
|---|---|---|---|---|---|---|---|
| LSTM | 51.5 | 76.2 | [23.6, 25.7] | [−5.6, −4.4] | 70.4 | [18.1, 19.6] | [−2.9, −1.9] |
| MLP | 52.3 | 77.7 | [24.4, 26.4] | [−4.5, −3.4] | 74.3 | [21.0, 23.0] | [−2.1, −0.7] |
| GRU | 52.3 | 75.6 | [22.2, 24.2] | [−4.8, −3.7] | 72.1 | [18.9, 20.5] | [−2.4, −1.3] |
| TCN | 50.3 | 76.7 | [25.3, 27.3] | [−4.8, −3.6] | 79.7 | [28.3, 30.5] | [−3.3, −1.8] |
| Linear | 50.0 | 69.4 | [18.6, 20.1] | [−3.8, −2.7] | – | – | – |

At matched mean width the uniformly widened AR intervals also have the lower interval score on every backbone (OOD, normalised: e.g. GRU EN 5.11 vs 5.56, TCN DN 5.24 vs 5.42, linear EN 5.89 vs 6.15).

## ID test
EN − AR and DN − AR intervals include zero on all backbones (as for LSTM/MLP). Raw (uncalibrated) band coverage: GRU EN 58.4, DN 30.5; TCN EN 62.7, DN 24.2; linear EN 46.6 (LSTM/MLP: 59.6/25.7, 61.0/71.0).

## Reading rule (fixed in advance)
- **F1** (OOD: EN/DN cover more than AR on the same backbone): **holds** on GRU, TCN and linear (all intervals > 0). Point estimates over the five backbones: +19 to +29 points (LSTM/MLP alone: +19 to +25).
- **F2** (OOD: width-matched AR covers at least as much as EN/DN): **holds** on GRU, TCN and linear (all intervals < 0). Point estimates over the five backbones: Δ from −5.0 to −1.4 (GRU −4.3/−1.8, TCN −4.2/−2.6, linear −3.3; EN/DN), the same range as LSTM/MLP.
- **F3** (ID: raw ensemble/dropout bands under-cover): **holds** on GRU, TCN and linear (24–63%).

Observation not covered by the rule: TCN DN-SCP has the highest OOD coverage of all spread constructions (79.7%) with the widest intervals (normalised width 2.62).
