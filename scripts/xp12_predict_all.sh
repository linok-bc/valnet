#!/usr/bin/env bash
# Per-frame predictions, per-frame IoU and demo videos for the three trained models
# on the xp12 combined TEST flights (scripts/predict_flights.py, deeplab/eval.py,
# scripts/demo_video.py). YOLO and VALNet use the widened crop for boxes at the
# 960-px cap; predictions_yolo/ (the plain crop) is kept for before/after figures.
#
# Usage: bash scripts/xp12_predict_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
OUT=results/xp12_combined
FLIGHTS="xp12_dataset_weather/000197 xp12_dataset_weather/000414 xp12_dataset_easy/000116"

echo "=== predict  $(date)"
python scripts/predict_flights.py --arch yolo --weights baseline/runs/xp12_combined_yolo/weights/best.pt \
  --out $OUT/predictions_yolo_widened
python scripts/predict_flights.py --arch valnet --weights checkpoints/xp12_combined/valnet_best.pt \
  --out $OUT/predictions_valnet
python scripts/predict_flights.py --arch deeplab --weights deeplab/runs/xp12_combined/best.pt \
  --out $OUT/predictions_deeplab

echo "=== score  $(date)"
# DeepLab is scored on its raw masks (iou_deeplab, from xp12_combined.sh); the
# others from their saved polygons. Same frames, labels and metric.
python deeplab/eval.py --yolo-json $OUT/predictions_yolo_widened --out $OUT/iou_yolo_widened
python deeplab/eval.py --yolo-json $OUT/predictions_valnet --out $OUT/iou_valnet

echo "=== videos  $(date)"
for f in $FLIGHTS; do
  python scripts/demo_video.py --flight $f --pred $OUT/predictions_yolo_widened --tag yolo \
    --model-name "YOLOv8s-seg, 1280 px"
  python scripts/demo_video.py --flight $f --pred $OUT/predictions_valnet --tag valnet \
    --model-name "VALNet, 1280 px"
  python scripts/demo_video.py --flight $f --pred $OUT/predictions_deeplab --tag deeplab \
    --model-name "DeepLabV3+ (ResNet-50)"
done
echo "=== done  $(date)"
