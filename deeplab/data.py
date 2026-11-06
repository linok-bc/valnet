"""xp12 frames as (image, binary runway mask) for semantic segmentation.

Reads the same split lists as YOLO and VALNet (configs/xp12_combined.yaml) and
rasterises each YOLO-format polygon label (one or more lines of normalized
x y pairs) into a mask at the native 1280x720. No resizing or letterboxing: a
fully convolutional network takes the native frame, and the mask is scored at
the resolution the labels were drawn in.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet, as torchvision's weights expect
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_list(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def label_mask(img_path, h, w):
    lab = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    m = np.zeros((h, w), np.uint8)
    for line in lab.read_text().splitlines():
        v = line.split()[1:]
        if len(v) >= 6:
            pts = np.array(v, np.float32).reshape(-1, 2) * [w, h]
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


class XP12Seg(Dataset):
    def __init__(self, list_path, train=False, fliplr=0.5, jitter=0.2):
        self.files = read_list(list_path)
        self.train, self.fliplr, self.jitter = train, fliplr, jitter

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        h, w = img.shape[:2]
        mask = label_mask(path, h, w)
        if self.train:
            if np.random.rand() < self.fliplr:      # mirrors image and mask together
                img, mask = img[:, ::-1], mask[:, ::-1]
            if self.jitter:                          # brightness / contrast
                a = 1 + np.random.uniform(-self.jitter, self.jitter)
                b = np.random.uniform(-self.jitter, self.jitter) / 2
                img = np.clip(img * a + b, 0, 1)
        img = (img - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(mask)), path)
