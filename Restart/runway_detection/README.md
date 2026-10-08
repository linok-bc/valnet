# Runway detection

Detect the runway (and optionally the centerline) in imagery.

Techniques live in subfolders:

| Folder | Status | Notes |
|--------|--------|--------|
| [`yolo/`](yolo/) | Implemented | Ultralytics YOLO-seg training + label prep |
| [`sam/`](sam/) | Stub | Segment Anything (future) |

Pipeline position: **stage 1** → feeds masks/polygons into `line_extraction/`.
