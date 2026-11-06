# Legacy datasets: RLD, bars, YouTube footage

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

Data-preparation tools for datasets used early in the project, before the XP12 work; none of
the reported results depend on them.

- `scripts/split_rld.py`: train/val/test split of the RLD runway dataset (`configs/rld.yaml`).
- `scripts/convert_bars.py`: convert the BARS dataset to YOLO-seg labels (`configs/bars.yaml`).
- `scripts/generate_video_demo.py`: VALNet masks on YouTube landing footage (`configs/youtube_video.yaml`).

Train or test VALNet on one with `python scripts/train.py data.config=configs/<name>.yaml`.
