# YOLOv8-seg baseline

The reference VALNet has to beat. Stock Ultralytics, no custom training code —
if this underperforms, it is the data or the task, not our plumbing.

## Why this exists

In the VALNet paper's own ablation (Table 9) the YOLOv8-seg baseline scores
65.9 mAP and full VALNet scores 69.4 — a **+3.5 point** gain. Without that
reference number measured on *our* data, a VALNet result is uninterpretable.

It matters more here than it did for the authors. Of VALNet's 19.35M
parameters, 11.47M (**59%**) are randomly initialized (CEM + AFPN + OAM); only
the backbone and head carry COCO weights. The paper pretrained the whole
network on COCO first (§5.1.3, and Table 8 reports VALNet's COCO2017 results)
and Table 5's 50-epoch recipe describes only the *fine-tuning* stage. We are
skipping that pretraining, so undertrained modules could easily land VALNet
*below* this baseline.

A run here is cheap — roughly 40 minutes for 50 epochs at the measured
110 img/s on a 3090 Ti — so measuring beats guessing.

## Layout

    config.yaml   two presets (see below)
    train.py      Ultralytics YOLO.train()
    eval.py       evaluates with BOTH the native validator and the shared harness
    runs/         outputs (gitignored)

## Presets

Two different questions, so two presets. **Run both.**

| preset | what it answers |
|---|---|
| `controlled` | Do CEM/AFPN/OAM earn their keep? Every shared hyperparameter matched to the VALNet run, so a difference is attributable to architecture. |
| `stock` | Is VALNet better than YOLOv8 out of the box? Ultralytics defaults, including the mosaic and flip augmentation that is a real part of why YOLOv8 scores what it scores. |

If VALNet only wins under `controlled`, the win came from hobbling the baseline.

## Usage

```bash
python baseline/train.py --preset controlled
python baseline/train.py --preset stock
python baseline/eval.py --weights baseline/runs/controlled/weights/best.pt
```

## What `controlled` matches, and what it cannot

Matched to `configs/main.yaml` and `scripts/train.py`: 50 epochs, batch 12,
SGD, lr0 0.01 → 0.0001 (`lrf=0.01`), cosine schedule (`cos_lr=True`; Ultralytics
defaults to linear), momentum 0.937, weight decay 0.0005, 3-epoch warmup with
momentum ramp, `fliplr=0`, `mosaic=0`, imgsz 640, same `configs/xp12.yaml`
splits. `nbs` is pinned to `batch` so the optimizer steps every batch, matching
the VALNet loop; Ultralytics would otherwise use `nbs=64` (accumulate=5 at
batch 12).

Remaining differences, unavoidable and worth stating in the writeup:

- **Head class count.** Ultralytics rebuilds the head at `nc=1` from the data
  yaml. VALNet keeps the pretrained 80-class COCO head. The baseline therefore
  has a slightly smaller, better-matched head.
- **Loss weighting.** Both use `v8SegmentationLoss` with default gains, but the
  baseline's per-batch gradient magnitude differs because of the `nbs` and head
  differences.
- **EMA.** Both use it. Ultralytics' `ModelEMA` iterates `state_dict()`;
  VALNet uses `valnet/ema.py`, which iterates deduplicated `named_parameters()`.
  Equivalent now that the duplicate backbone registration is gone.

## Reading the two metrics

`eval.py` prints the same quantity twice on purpose. The native validator is
the number to trust and the one comparable to published YOLOv8 results. The
shared harness (`valnet/evaluate_map.py`) is where VALNet's numbers come from,
so **only those are directly comparable to a VALNet run**. If the two disagree
by much, the shared harness needs investigating before it is used to judge
anything.

## Before running

The xp12 labels are suspect. Two known problems: near-plane clipping corrupts
polygons past the threshold (~20% have IoU<0.5 against the unclipped polygon,
~11% have vertices above the horizon), and the runway corners are projected at
the airport reference elevation rather than the actual runway elevation, so the
implied label plane disagrees with logged AGL by >5 m in 134 of 250 episodes.

A baseline trained on wrong labels measures nothing. Validate the labels
before spending a run here.
