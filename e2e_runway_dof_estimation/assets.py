"""Demo assets on the Hugging Face Hub: DeepLabV3+ weights, the runway database, sample flights,
and preloaded runs (saved results + a replay video, so the app shows them with no computation).

They live in the private model repo linok-bc/e2e-runway-dof-estimation, so `hf auth login`
(or HF_TOKEN) is needed once. Files are cached by huggingface_hub; sample flights are
unpacked next to the cache under ~/.cache/e2e_runway_dof_estimation/samples/.
"""

import os
import subprocess
from pathlib import Path

REPO = "linok-bc/e2e-runway-dof-estimation"
# Sample flights, each also a preloaded run: a showcase, the best-scoring test flight of its
# condition (DeepLab masks, runway-database geometry, crab EKF). The first opens by default.
# Over all 88 test flights the average is 0.047 deg yaw and 0.53 m cross-track.
SAMPLES = {
    "weather_000185": "ESIB 19, clear, 7.3 m/s crosswind",
    "weather_000331": "UMBB 29, heavy rain, 5.9 m/s crosswind",
    "weather_000270": "EETU 26, haze",
    "easy_000126": "USNN 21, clear",
}
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


def preloaded_run(name: str) -> Path:
    """Folder with a saved run's records.json and masks.npz (pipeline.Result.save)."""
    if name not in SAMPLES:
        raise ValueError(f"unknown run {name!r}; choose from {tuple(SAMPLES)}")
    for f in ("records.json", "masks.npz"):
        path = _fetch(f"runs/{name}/{f}")
    return path.parent


def preloaded_video(name: str) -> Path:
    return _fetch(f"runs/{name}/replay.mp4")


def download_all() -> None:
    print("weights:", weights())
    print("runways:", runways_dat())
    for s in SAMPLES:
        print(f"{s}:", sample_flight(s), preloaded_run(s), preloaded_video(s))


if __name__ == "__main__":
    download_all()
