# YOLO runway / centerline detection

Ultralytics YOLO-seg training and label utilities.

| Module | Role |
|--------|------|
| `labels.py` | `process_txt`, `convert_labels_tree`, `filter_classes` |
| `convert_labels.py` | CLI for centerline insertion (uses `line_extraction.geometric`) |
| `split.py` | CLI train/val/test split |
| `dataset_yaml.py` | CLI / API for Ultralytics data YAML |
| `train.py` | CLI training from `configs/train/*.yaml` |

Colab Drive / rsync helpers from the legacy notebooks were not migrated; use local `data/` paths instead.
