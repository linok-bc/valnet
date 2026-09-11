import math
from pathlib import Path
import torch
from ultralytics.data import YOLODataset


class XTrackDataset(YOLODataset):
    """YOLODataset + per-sample (roll, pitch, altitude, cross-track) labels.

    Pose file format: "<roll> <pitch> <yaw>" in degrees.
        positive roll  -> right wing down
        positive pitch -> nose up
        positive yaw   -> right of runway
    Position file format: "<x> <y> <z>"  where y is altitude, z is cross-track.
    """

    def __getitem__(self, index):
        try:
            sample = super().__getitem__(index)
            if sample is None:
                print(f"[xtrack] None sample at index {index}: {self.im_files[index]}")
                return None
    
            img_path = Path(self.im_files[index])
            position = [float(v) for v in (img_path.parents[1] / "position" / f"{img_path.stem}.txt").read_text().split()]
            rpy_deg  = [float(v) for v in (img_path.parents[1] / "pose"     / f"{img_path.stem}.txt").read_text().split()]
    
            sample["roll"]   = torch.tensor(math.radians(rpy_deg[0]), dtype=torch.float32)
            sample["pitch"]  = torch.tensor(math.radians(rpy_deg[1]), dtype=torch.float32)
            sample["alt"]    = torch.tensor(position[1], dtype=torch.float32)
            sample["xtrack"] = torch.tensor(position[2], dtype=torch.float32)
    
            if self.augment and torch.rand(1).item() < 0.5:
                sample["img"]    = torch.flip(sample["img"],   dims=[-1])
                sample["masks"]  = torch.flip(sample["masks"], dims=[-1])
                sample["xtrack"] = -sample["xtrack"]
                sample["roll"]   = -sample["roll"]
    
            return sample
    
        except Exception as e:
            print(f"[xtrack] FAILED at index {index} ({self.im_files[index]}): {type(e).__name__}: {e}")
            return None


    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        new_batch = YOLODataset.collate_fn(batch)
        for k in ("roll", "pitch", "alt", "xtrack"):
            new_batch[k] = torch.stack([b[k] for b in batch])
        return new_batch
