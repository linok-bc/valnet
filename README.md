# Joint demo: Samuel's heading with our edge-line cross-track

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

Samuel's estimator is the `capstone-dof-estimation/` submodule (fork of samuelmoyal/Restart,
branch `crab-ekf`); fetch it with `git submodule update --init`. This experiment pins `kf_crab=False`, the original 3-state EKF all of its
results were made with.

Samuel's joint heading/cross-track mask fit and our edge-line cross-track on 13 test flights of
the old dataset, plus a fused variant that feeds the edge-line estimate into his EKF:

| Estimate (150-1000 m) | Cross-track MAE |
|---|---|
| Samuel's mask fit | 0.48 m |
| Edge lines | 0.50 m |
| Fused | 0.41 m |

Heading: 0.055 deg from 1500 m. Run in order: `cache_masks`, `run_heading`, `noise_model`,
`fuse`, `run_bev`, `findings`, `summary`, then `render` / `render_compare` for videos
(`python -m experiments.joint_demo.<name>`). Report pages in `experiments/joint_demo/report/`.
