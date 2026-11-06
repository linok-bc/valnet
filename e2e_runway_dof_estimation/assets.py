"""Demo assets on the Hugging Face Hub: DeepLabV3+ weights, the runway database, sample flights.

They live in the private model repo linok-bc/e2e-runway-dof-estimation, so `hf auth login`
(or HF_TOKEN) is needed once. Files are cached by huggingface_hub; sample flights are
unpacked next to the cache under ~/.cache/e2e_runway_dof_estimation/samples/.
"""

import os
import subprocess
from pathlib import Path

REPO = "linok-bc/e2e-runway-dof-estimation"
SAMPLES = ("weather_000197", "weather_000613")   # fog EGNH 28; crosswind EDDG 25
CACHE = Path(os.environ.get("E2E_RUNWAY_CACHE", Path.home() / ".cache/e2e_runway_dof_estimation"))


def _fetch(path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, path_in_repo))


def weights() -> Path:
    return _fetch("weights/deeplabv3plus_xp12.pt")


def runways_dat() -> Path:
    return _fetch("data/runways.dat")


def sample_flight(name: str) -> Path:
    """Download and unpack one sample flight; returns its folder (images/, pose/, position/, ...)."""
    if name not in SAMPLES:
        raise ValueError(f"unknown sample {name!r}; choose from {SAMPLES}")
    dataset, episode = name.split("_")
    out = CACHE / "samples" / f"xp12_dataset_{dataset}" / episode
    if not (out / "meta.txt").exists():
        archive = _fetch(f"samples/{name}.tar.zst")
        (CACHE / "samples").mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(CACHE / "samples")], check=True)
    return out


def download_all() -> None:
    print("weights:", weights())
    print("runways:", runways_dat())
    for s in SAMPLES:
        print(f"{s}:", sample_flight(s))


if __name__ == "__main__":
    download_all()
