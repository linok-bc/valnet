#!/usr/bin/env bash
# VALNet vs YOLOv8s-seg, end to end: train all three arms back to back, then
# score every best checkpoint on the TEST split with the shared harness.
#
#   valnet      scripts/train.py, configs/main.yaml
#   controlled  baseline, every shared hyperparameter matched to the VALNet run
#   stock       baseline, Ultralytics defaults (mosaic, fliplr, ...)
#
# Usage: bash scripts/compare.sh [arm ...]     (default: all three)
# Logs and the final table land in results/compare/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg   # train.py ends with plt.show()
OUT=results/compare
mkdir -p "$OUT"
ARMS=("${@:-valnet controlled stock}")
ARMS=(${ARMS[@]})

for arm in "${ARMS[@]}"; do
    echo "=== train $arm  $(date)"
    case $arm in
        valnet) python scripts/train.py train.logdir=$OUT/valnet_tb ;;
        *)      python baseline/train.py --preset "$arm" --name "$arm" ;;
    esac > "$OUT/train_$arm.log" 2>&1
done

for arm in "${ARMS[@]}"; do
    echo "=== test $arm  $(date)"
    case $arm in
        valnet) python scripts/test.py test.checkpoint=checkpoints/valnet_best.pt test.output_dir=$OUT/valnet_test_visuals ;;
        *)      python baseline/eval.py --split test --weights baseline/runs/$arm/weights/best.pt ;;
    esac > "$OUT/test_$arm.log" 2>&1
    grep -A3 "full-res" "$OUT/test_$arm.log" | sed "s/^/[$arm] /"
done | tee "$OUT/summary.txt"
echo "=== done  $(date)"
