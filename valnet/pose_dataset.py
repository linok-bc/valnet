from ultralytics.data import YOLODataset
from valnet.pose_loss import normalize_position
import torch
from pathlib import Path

class PoseYOLODataset(YOLODataset):
    """YOLODataset + per-sample 4DOF position label + per-sample 4DOF pose label"""

    def __init__(self, *args, yz_scale=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.yz_scale = yz_scale

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        # Find the matching pose file. self.im_files[index] is the image path.
        img_path = Path(self.im_files[index])
        pose_path = img_path.parents[1] / "pose" / f"{img_path.stem}.txt"
        position_path = img_path.parents[1] / "position" / f"{img_path.stem}.txt"

        position = torch.tensor(
            [float(v) for v in position_path.read_text().split()],
            dtype=torch.float32,
        )  # [3]: x, y, z
        pose = torch.tensor(
            [float(v) for v in pose_path.read_text().split()],
            dtype=torch.float32,
        )  # [4]: qx, qy, qz, qw

        vals = torch.cat([position, pose])

        pos_norm = normalize_position(vals[:3], yz_scale=self.yz_scale)
        quat = vals[3:] / vals[3:].norm().clamp(min=1e-8)  # ensure unit

        sample["pose"] = torch.cat([pos_norm, quat])  # [7]
        return sample

    @staticmethod
    def collate_fn(batch):
        new_batch = YOLODataset.collate_fn(batch)
        new_batch["pose"] = torch.stack([b["pose"] for b in batch])
        return new_batch
