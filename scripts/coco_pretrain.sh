#!/usr/bin/env bash
# VALNet with COCO pretraining, then xp12 fine-tuning, then the test split.
#
#   1. pretrain  CEM + AFPN + OAM on COCO 2017 (80-class head). Backbone and
#                YOLOv8 head stay frozen at Ultralytics' COCO weights.
#                Ultralytics' COCO augmentation: mosaic, flips.
#   2. finetune  whole network on xp12 from the last pretrain epoch, head
#                resized to 1 class. Same recipe as the plain VALNet run.
#   3. test      shared harness on the xp12 test split -> results/coco/summary.txt
#
# Usage: bash scripts/coco_pretrain.sh [pretrain_epochs]    (default 25)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg   # train.py ends with plt.show()
EPOCHS=${1:-25}
OUT=results/coco
PRE=checkpoints/valnet_coco_pretrain
FT=checkpoints/valnet_coco_ft
mkdir -p "$OUT"

echo "=== pretrain on COCO ($EPOCHS epochs)  $(date)"
python scripts/train.py data.config=configs/coco.yaml data.num_classes=80 \
    'train.freeze=[backbone,head]' train.mosaic=1.0 train.fliplr=0.5 \
    train.epochs=$EPOCHS train.save_freq=5 train.val_freq=5 \
    train.ckpt_dir=$PRE train.logdir=$OUT/pretrain_tb > "$OUT/pretrain.log" 2>&1

echo "=== fine-tune on xp12  $(date)"
python scripts/train.py train.init_from=$PRE/valnet_epoch$EPOCHS.pt \
    train.ckpt_dir=$FT train.logdir=$OUT/finetune_tb > "$OUT/finetune.log" 2>&1

echo "=== test  $(date)"
python scripts/test.py test.checkpoint=$FT/valnet_best.pt \
    test.output_dir=$OUT/test_visuals test.save_instance=False > "$OUT/test.log" 2>&1
grep -a -A3 "full-res" "$OUT/test.log" | sed "s/^/[valnet_coco] /" | tee "$OUT/summary.txt"
echo "=== done  $(date)"
