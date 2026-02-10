"""
Wrapper around the ARCADE architectural smell detector.

Produces per-version smell JSON files under ``data/smells/<system>/<algorithm>/``.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path
from typing import Iterable, List

from src.preprocessing.constants import (
    CLUSTERS_ROOT,
    SMELLS_ROOT,
    time_print,
)
from src.preprocessing.facts_lookup import resolve_facts_rsf
from src.utils.system_paths import resolve_system_subdir

ARCADE_JAR = Path("ARCADE_Core.jar")
SUPPORTED_ALGORITHMS = ["acdc"]


class SmellDetectionError(RuntimeError):
    """Raised when ARCADE smell detection fails for a given system/version."""


def _resolve_cluster_file(version_dir: Path) -> Path:
    cluster_candidates = sorted(version_dir.glob("*_clusters.rsf"))
    if not cluster_candidates:
        raise SmellDetectionError(f"No *_clusters.rsf file found in {version_dir}")
    return cluster_candidates[0]


def _doc_topics_path(version_dir: Path, cluster_file: Path) -> str:
    parts = cluster_file.name.split("_")
    if len(parts) < 3:
        return ""

    measure = parts[1]
    if measure != "JS":
        return ""

    size = parts[2]
    candidate = version_dir / f"{parts[0]}_{measure}_{size}_clusteredDocTopics.json"
    return str(candidate) if candidate.exists() else ""


def run_smell_detector(system_name: str, algorithm: str) -> None:
    """Run ARCADE smell detection for a single algorithm."""
    if not ARCADE_JAR.exists():
        raise SmellDetectionError(
            f"Missing ARCADE_Core.jar at {ARCADE_JAR.resolve()}. "
            "Smell detection cannot proceed."
        )

    try:
        cluster_root = resolve_system_subdir(CLUSTERS_ROOT, system_name, create=False)
    except FileNotFoundError as error:
        raise SmellDetectionError(
            f"No cluster directories found for {system_name} (missing root under {CLUSTERS_ROOT})."
        ) from error

    version_dirs = sorted((cluster_root / algorithm).glob("*"))
    if not version_dirs:
        raise SmellDetectionError(
            f"No cluster directories found for {system_name}/{algorithm} "
            f"under {cluster_root / algorithm}"
        )

    smell_root = resolve_system_subdir(SMELLS_ROOT, system_name, create=True)
    output_dir = smell_root / algorithm
    output_dir.mkdir(parents=True, exist_ok=True)

    for version_dir in version_dirs:
        cluster_file = _resolve_cluster_file(version_dir)
        version = cluster_file.name.split("_")[0]
        try:
            facts_file = resolve_facts_rsf(system_name, version)
        except FileNotFoundError as error:
            raise SmellDetectionError(
                f"Required facts file missing for {version}: {error}"
            ) from error

        doc_topics = _doc_topics_path(version_dir, cluster_file)
        output_file = output_dir / f"{version}_smells.json"

        command: List[str] = [
            "/usr/lib/jvm/java-11-openjdk-amd64/bin/java",
            "-cp",
            str(ARCADE_JAR),
            "edu.usc.softarch.arcade.antipattern.detection.ArchSmellDetector",
            str(facts_file),
            str(cluster_file),
            str(output_file),
        ]
        if doc_topics:
            command.append(doc_topics)

        time_print(
            f"Running ARCADE smell detector for {system_name} {algorithm} {version}"
        )
        subprocess.run(command, check=True)


def run_smell_detection(system_name: str, algorithms: Iterable[str] | None = None) -> None:
    """Run smell detection for the requested algorithms."""
    targets = list(algorithms) if algorithms is not None else SUPPORTED_ALGORITHMS
    for algorithm in targets:
        time_print(f"Smell detection for {system_name} with {algorithm.upper()}")
        run_smell_detector(system_name, algorithm)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARCADE smell detection.")
    parser.add_argument("system", help="Subject system name (e.g., inkscape, graphviz)")
    parser.add_argument(
        "--algorithm",
        "-a",
        action="append",
        choices=SUPPORTED_ALGORITHMS,
        help="Algorithm(s) to process (default: all supported)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_smell_detection(args.system, args.algorithm)
    except Exception as error:  # pragma: no cover - CLI wrapper
        raise SystemExit(f"Smell detection failed: {error}") from error
