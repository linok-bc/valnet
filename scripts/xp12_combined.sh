#!/usr/bin/env bash
# YOLOv8s-seg (stock) and VALNet on the new xp12 data (easy + weather), 1280 px.
# Train both back to back, then score each best checkpoint on the full TEST
# split with the shared harness.
#
#   yolo    baseline/train.py --preset stock
#   valnet  scripts/train.py, configs/main.yaml defaults
#   deeplab deeplab/train.py, deeplab/config.yaml (DeepLabV3+, COCO-pretrained ResNet-50)
#
# Data: configs/xp12_combined.yaml (visibility-filtered, train/val every 5th frame).
# Both start from COCO yolov8s-seg.pt. Batch 8 for both: VALNet at 1280 peaks at
# ~20 GB of the 3090 Ti's 24 GB.
#
# YOLO and VALNet are also scored with the shared mAP harness; DeepLab is semantic
# (no instances or confidences), so it is scored by per-frame mask IoU with
# deeplab/eval.py, which scores YOLO's saved predictions the same way.
#
# Usage: bash scripts/xp12_combined.sh [arm ...]     (default: yolo valnet)
# Logs and the final table land in results/xp12_combined/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg   # train.py ends with plt.show()
OUT=results/xp12_combined
DATA=configs/xp12_combined.yaml
IMGSZ=1280
BATCH=8
EPOCHS=50
mkdir -p "$OUT"
ARMS=("${@:-yolo valnet}")
ARMS=(${ARMS[@]})

for arm in "${ARMS[@]}"; do
    echo "=== train $arm  $(date)"
    case $arm in
        yolo)   python baseline/train.py --preset stock --name xp12_combined_yolo \
                    --data $DATA --imgsz $IMGSZ --batch $BATCH --epochs $EPOCHS ;;
        valnet) python scripts/train.py data.config=$DATA data.image_size=$IMGSZ \
                    train.batch_size=$BATCH train.epochs=$EPOCHS \
                    train.ckpt_dir=checkpoints/xp12_combined train.logdir=$OUT/valnet_tb ;;
        deeplab) python deeplab/train.py ;;
    esac > "$OUT/train_$arm.log" 2>&1
done

for arm in "${ARMS[@]}"; do
    echo "=== test $arm  $(date)"
    case $arm in
        yolo)   python baseline/eval.py --split test --data $DATA --imgsz $IMGSZ --batch $BATCH \
                    --weights baseline/runs/xp12_combined_yolo/weights/best.pt ;;
        valnet) python scripts/test.py data.config=$DATA data.image_size=$IMGSZ \
                    test.batch_size=$BATCH test.visuals=False \
                    test.checkpoint=checkpoints/xp12_combined/valnet_best.pt ;;
        deeplab) python deeplab/eval.py --weights deeplab/runs/xp12_combined/best.pt --out $OUT/iou_deeplab &&
                 python deeplab/eval.py --yolo-json $OUT/predictions_yolo --out $OUT/iou_yolo ;;
    esac > "$OUT/test_$arm.log" 2>&1
    grep -A3 "full-res" "$OUT/test_$arm.log" | sed "s/^/[$arm] /" || true
    grep -A1 "^  overall" "$OUT/test_$arm.log" | grep "all" | sed "s/^/[$arm] /" || true
done | tee "$OUT/summary.txt"
echo "=== done  $(date)"
