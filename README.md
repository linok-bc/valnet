# Samuel's estimator on the test split, and the crosswind (crab) EKF fix

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

Samuel's estimator is the `capstone-dof-estimation/` submodule (fork of samuelmoyal/Restart,
branch `crab-ekf`); fetch it with `git submodule update --init`.

Samuel's `maskfit_analytic_lm` with his kinematic EKF on 88 test flights (150-1500 m, YOLOv8
masks). The EKF moved the aircraft along its nose; in a crosswind it crabs, so the prediction
slid sideways every frame. The fix (`kf_crab`, on the fork's `crab-ekf` branch) adds the crab
angle as a fourth state.

| Filter | Yaw MAE | Cross-track MAE | Rejected frames |
|---|---|---|---|
| Original EKF | 0.044 deg | 2.17 m | 367 |
| No filter | 0.032 deg | 0.54 m | 0 |
| Crab EKF | 0.046 deg | 0.52 m | 28 |

Yaw is still over-smoothed (R assumes 0.15 deg; per-frame noise is ~0.04 deg).
`compare_masks.py` feeds the estimator DeepLabV3+ masks instead: on weather/000414 cross-track
0.47 -> 0.20 m with the crab EKF. Run: `python -m experiments.restart_test.run [--filter crab|old|none]`,
`compare_masks`, `video`.
