# VALNet vs YOLOv8-seg (old XP12 dataset)

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

VALNet (CEM + AFPN + OAM on a YOLOv8s-seg backbone, paper size 12.8M params) against stock
YOLOv8s-seg on the original 108-flight XP12 dataset at 640 px, including the fixes that made
VALNet train at all (pretrained weights were frozen; CEM's softmax was global; AFPN fused in
the wrong order) and the corrected mAP harness.

Test mask mAP50-95 (quarter resolution, which matches Ultralytics' validator):

| Model | mAP |
|---|---|
| VALNet | 0.661 |
| VALNet + mosaic | 0.655 |
| YOLOv8s-seg, stock | 0.654 |
| VALNet + COCO pretraining | 0.650 |
| YOLOv8s-seg, matched recipe | 0.623 |
| YOLO26s-seg, stock | 0.618 |

Run: `bash scripts/compare.sh` (VALNet vs both YOLO presets), `bash scripts/coco_pretrain.sh`
(COCO pretraining then fine-tuning), `python scripts/variant_videos.py` (per-flight videos).
Data: `configs/xp12.yaml` -> `datasets/xp12_dataset`.
