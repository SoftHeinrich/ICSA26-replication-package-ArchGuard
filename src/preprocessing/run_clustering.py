"""
Command helpers for running ARCADE clustering (ACDC only).

Historically this module managed ARC/Limbo/PKG in addition to ACDC.
OpenRGB preprocessing currently relies on ACDC exclusively, so this file
now focuses on that single algorithm to reduce dependencies and runtime.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

if __package__:
    from .constants import FACTS_ROOT, CLUSTERS_ROOT, time_print
else:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.preprocessing.constants import FACTS_ROOT, CLUSTERS_ROOT, time_print

from src.preprocessing.facts_lookup import resolve_facts_rsf
from src.utils.system_paths import resolve_system_subdir


def _facts_dir(system_name: str, create: bool = False) -> Path:
    return resolve_system_subdir(FACTS_ROOT, system_name, create=create)


def _clusters_dir(system_name: str, create: bool = True) -> Path:
    return resolve_system_subdir(CLUSTERS_ROOT, system_name, create=create)


def run_acdc(project_name: str, project_version: str) -> None:
    facts_file = resolve_facts_rsf(project_name, project_version)
    cluster_dir = _clusters_dir(project_name) / "acdc" / project_version
    cluster_dir.mkdir(parents=True, exist_ok=True)
    output_file = cluster_dir / f"{project_name}-{project_version}_ACDC_clusters.rsf"

    command = [
        "java",
        "-cp",
        "ARCADE_Core.jar",
        "edu.usc.softarch.arcade.clustering.acdc.ACDC",
        str(facts_file),
        str(output_file),
    ]
    time_print(f"Running ACDC on {project_name}-{project_version}.")
    subprocess.run(command, check=True)


def _parse_versions(system_name: str) -> List[str]:
    facts_dir = _facts_dir(system_name)
    prefix = f"{system_name.lower()}-"
    versions: List[str] = []

    for entry in facts_dir.glob("*_deps.rsf"):
        filename = entry.name
        if not filename.lower().endswith("_deps.rsf"):
            continue

        without_suffix = filename[: -len("_deps.rsf")]
        if without_suffix.lower().startswith(prefix):
            versions.append(without_suffix[len(prefix) :])

    return sorted(set(versions))


def cluster_system(system_name: str, language: str, memory: str) -> None:
    """Cluster every available version of the requested system using ACDC."""
    _ = (language, memory)  # Retained for backward compatibility with older callers.
    versions = _parse_versions(system_name)
    if not versions:
        raise FileNotFoundError(
            f"No *_deps.rsf files were found under {_facts_dir(system_name)} for {system_name}."
        )

    time_print(f"Initiating clustering of {system_name} ({len(versions)} versions).")
    for version in versions:
        time_print(f"Starting clustering for {system_name}-{version}.")
        run_acdc(system_name, version)
        print()

    time_print("Clustering completed (ACDC only).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARCADE ACDC clustering.")
    parser.add_argument("system_name", help="System slug (matches data/facts/<system_name>).")
    parser.add_argument("language", help="Programming language for the target system (retained for backward compatibility).")
    parser.add_argument(
        "--memory",
        default="4",
        help="Deprecated placeholder retained for CLI compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cluster_system(args.system_name, args.language, args.memory)


if __name__ == "__main__":
    main()
