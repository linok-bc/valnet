# Data layout

Place datasets under this folder. Contents of `raw/`, `processed/`, and `splits/` are gitignored.

```text
data/
├── raw/          # original images + labels (unsorted or vendor dump)
├── processed/    # after centerline conversion / class remapping
└── splits/       # images/{train,val,test} + labels/{train,val,test}
```

Typical YOLO layout after `split`:

```text
data/splits/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Point Ultralytics YAML `path` / `train` / `val` / `test` at these directories (see `configs/data.example.yaml`).
