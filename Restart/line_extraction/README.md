# Line extraction

Derive a centerline (or equivalent 1D representation) from a runway mask / polygon.

| Folder | Status | Notes |
|--------|--------|--------|
| [`geometric/`](geometric/) | Implemented | Triangle centerline from runway polygon (label prep + inference geometry) |
| [`skeletonization/`](skeletonization/) | Stub | Morphological skeleton / medial axis |

Pipeline position: **stage 2** → line goes to `pose_estimation/`.
