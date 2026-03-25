import os
import shutil
import random

src_images = "images"
src_labels = "labels"
splits = {"train": 0.8, "val": 0.1, "test": 0.1}

# Get matching filenames
filenames = sorted(os.listdir(src_images))
random.seed(42)
random.shuffle(filenames)

n = len(filenames)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

split_files = {
    "train": filenames[:train_end],
    "val": filenames[train_end:val_end],
    "test": filenames[val_end:],
}

for split, files in split_files.items():
    os.makedirs(os.path.join(split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split, "labels"), exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(src_images, f), os.path.join(split, "images", f))
        label_f = os.path.splitext(f)[0] + ".txt"
        label_src = os.path.join(src_labels, label_f)
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(split, "labels", label_f))

print(f"Split complete: train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")
