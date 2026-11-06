# Bird's-eye view and Ryu et al.'s lateral estimator

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

Does rectifying the camera image to a bird's-eye view (BEV) give a better runway target for
heading and cross-track? Four arms ({perspective, BEV} x {segment, oriented box}) on one
shared metric-space scoring pipeline, plus altitude-tied BEV grids, then Ryu et al.'s
(IJRR 2024) lateral estimator on the BEV masks with an ablation of its parts.

- Perspective segmentation wins overall (2.83 m cross-track) and beyond 300 m; BEV wins
  inside 300 m.
- Once rectified, the mask carries nothing the four corners do not (BEV segment = BEV box).
- The altitude-tied grid cuts 1000-1500 m heading error 28% and cross-track 38%.
- Camera-only BEV cross-track bottoms out near 6 m: range x heading error dominates.

Details and commands: `experiments/bev_yolo/README.md`; literature survey in
`experiments/bev_yolo/LITERATURE.md`. Tests: `python -m experiments.tests.test_bev`,
`test_estimator`, `test_ryu` (synthetic, no data needed).
