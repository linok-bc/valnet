"""Download subsets of the DEEL-AI/LARD_V2 HuggingFace dataset."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ID = "DEEL-AI/LARD_V2"
SOURCES = ("flsim", "arcgis", "bingmaps", "ges", "xplane")
SPLITS = ("train", "test")
AVG_IMAGE_MB = 0.62
DEFAULT_WORKERS = 8


def _metadata_name(source: str, split: str) -> str:
    return f"metadata_{source}_{split}.csv"


def _read_metadata_rows(meta_path: Path) -> list[dict[str, str]]:
    with meta_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _images_for_scenarios(
    rows: list[dict[str, str]],
    scenarios: set[str],
) -> list[str]:
    return sorted(
        {
            row["image"]
            for row in rows
            if row["scenario"] in scenarios
        }
    )


def _select_scenarios_by_budget(
    rows: list[dict[str, str]],
    budget_gb: float,
) -> tuple[list[str], list[str]]:
    images_by_scenario: dict[str, set[str]] = {}
    for row in rows:
        images_by_scenario.setdefault(row["scenario"], set()).add(row["image"])

    selected_scenarios: list[str] = []
    selected_images: list[str] = []
    used_mb = 0.0
    budget_mb = budget_gb * 1024

    for scenario in sorted(images_by_scenario):
        images = sorted(images_by_scenario[scenario])
        add_mb = len(images) * AVG_IMAGE_MB
        if selected_scenarios and used_mb + add_mb > budget_mb:
            break
        selected_scenarios.append(scenario)
        selected_images.extend(images)
        used_mb += add_mb

    if not selected_scenarios:
        first = next(iter(sorted(images_by_scenario)))
        selected_scenarios = [first]
        selected_images = sorted(images_by_scenario[first])

    return selected_scenarios, selected_images


def _download_hf_files(
    repo_paths: list[str],
    local_dir: Path,
    *,
    repo_id: str = REPO_ID,
    workers: int = DEFAULT_WORKERS,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        repo_path
        for repo_path in repo_paths
        if not (local_dir / repo_path).exists()
    ]
    skipped = len(repo_paths) - len(pending)
    if skipped:
        print(f"Skipping {skipped} file(s) already present in {local_dir}")

    total = len(pending)
    if not total:
        return

    def _fetch(repo_path: str) -> str:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=str(local_dir),
        )
        return repo_path

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, path): path for path in pending}
        for future in as_completed(futures):
            completed += 1
            repo_path = future.result()
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(f"[{completed}/{total}] {repo_path}", flush=True)


def download_lard(
    out_dir: str | Path,
    *,
    sources: list[str] | None = None,
    splits: list[str] | None = None,
    budget_gb: float | None = None,
    scenarios: list[str] | None = None,
    metadata_only: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """
    Download LARD V2 images and metadata into ``out_dir``.

    Without ``budget_gb`` or ``scenarios``, all images referenced by the selected
    metadata CSV files are downloaded.
    """
    out_dir = Path(out_dir)
    sources = list(sources or ["flsim"])
    splits = list(splits or ["train"])

    for source in sources:
        if source not in SOURCES:
            raise ValueError(f"Unknown source {source!r}. Choose from {SOURCES}.")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}. Choose from {SPLITS}.")

    scenario_filter = set(scenarios or [])
    repo_paths: list[str] = []
    summary: dict = {
        "out_dir": str(out_dir),
        "sources": sources,
        "splits": splits,
        "metadata_files": [],
        "scenarios": [],
        "image_count": 0,
        "estimated_gb": 0.0,
    }

    for source in sources:
        for split in splits:
            meta_name = _metadata_name(source, split)
            summary["metadata_files"].append(meta_name)
            repo_paths.append(meta_name)

    _download_hf_files(repo_paths, out_dir, workers=workers)

    image_paths: set[str] = set()
    selected_scenarios: set[str] = set()

    for source in sources:
        for split in splits:
            meta_path = out_dir / _metadata_name(source, split)
            rows = _read_metadata_rows(meta_path)

            if scenario_filter:
                image_paths.update(_images_for_scenarios(rows, scenario_filter))
                selected_scenarios.update(scenario_filter)
            elif budget_gb is not None:
                scenarios_for_split, images = _select_scenarios_by_budget(rows, budget_gb)
                image_paths.update(images)
                selected_scenarios.update(scenarios_for_split)
            else:
                image_paths.update(row["image"] for row in rows)
                selected_scenarios.update(row["scenario"] for row in rows)

    summary["scenarios"] = sorted(selected_scenarios)
    summary["image_count"] = len(image_paths)
    summary["estimated_gb"] = round(len(image_paths) * AVG_IMAGE_MB / 1024, 2)

    if metadata_only:
        print(
            f"Metadata only: {len(summary['metadata_files'])} CSV file(s) "
            f"-> {out_dir}"
        )
        return summary

    if not image_paths:
        raise ValueError("No images matched the requested filters.")

    print(
        f"Downloading {summary['image_count']} images "
        f"(~{summary['estimated_gb']} GB) to {out_dir}"
    )
    _download_hf_files(sorted(image_paths), out_dir, workers=workers)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download a subset of DEEL-AI/LARD_V2 into data/LARD.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/LARD"),
        help="Destination directory (default: data/LARD)",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        dest="sources",
        help="Data source (repeatable). Default: flsim",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        dest="splits",
        help="Split (repeatable). Default: train",
    )
    parser.add_argument(
        "--budget-gb",
        type=float,
        default=None,
        help="Approximate download budget in GB (selects whole scenarios)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Explicit scenario IDs to download",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download metadata CSV files only",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args(argv)

    summary = download_lard(
        args.out_dir,
        sources=args.sources,
        splits=args.splits,
        budget_gb=args.budget_gb,
        scenarios=args.scenarios,
        metadata_only=args.metadata_only,
        workers=args.workers,
    )
    print(
        f"Done. {summary['image_count']} images, "
        f"{len(summary['scenarios'])} scenario(s), "
        f"~{summary['estimated_gb']} GB -> {summary['out_dir']}"
    )


if __name__ == "__main__":
    main()
