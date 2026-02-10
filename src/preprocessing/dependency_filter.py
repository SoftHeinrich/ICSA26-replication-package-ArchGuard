"""
Utilities for filtering dependency RSF files generated during fact extraction.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from src.preprocessing.constants import FACTS_ROOT, time_print
from src.utils.system_paths import resolve_system_subdir

DEFAULT_KEYWORDS = ["test"]


def _collect_dependency_files(system_name: str) -> List[Path]:
    facts_dir = resolve_system_subdir(FACTS_ROOT, system_name, create=False)
    return sorted(facts_dir.glob("*_deps.rsf"))


def remove_dependencies_by_keyword(
    system_name: str,
    keywords: Iterable[str] | None = None,
) -> None:
    """
    Remove dependency rows whose paths contain any of the provided keywords.

    Args:
        system_name: Target system slug (e.g., ``inkscape``).
        keywords: Collection of substrings to match (case-insensitive). Defaults to ``["test"]``.
    """
    dependency_files = _collect_dependency_files(system_name)
    if not dependency_files:
        time_print(
            f"No dependency RSF files found under {FACTS_ROOT} for {system_name}; skipping keyword filtering."
        )
        return

    normalized_keywords = [
        keyword.lower() for keyword in (keywords or DEFAULT_KEYWORDS) if keyword
    ]
    if not normalized_keywords:
        time_print("No keywords provided for dependency filtering; skipping step.")
        return

    archive_root = (
        resolve_system_subdir(FACTS_ROOT, system_name, create=False)
        / "archive"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    archive_root.mkdir(parents=True, exist_ok=True)

    total_removed = 0

    for rsf_file in dependency_files:
        original_lines = rsf_file.read_text(encoding="utf-8").splitlines()
        kept_lines: List[str] = []
        removed_lines: List[str] = []

        for line in original_lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in normalized_keywords):
                removed_lines.append(line)
            else:
                kept_lines.append(line)

        if not removed_lines:
            continue

        total_removed += len(removed_lines)
        shutil.copy2(rsf_file, archive_root / rsf_file.name)

        removed_path = archive_root / f"{rsf_file.stem}_removed.txt"
        removed_path.write_text("\n".join(removed_lines) + "\n", encoding="utf-8")

        rsf_file.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

    if total_removed == 0:
        time_print(
            f"No dependency rows matched keywords {normalized_keywords} for {system_name}."
        )
        archive_root.rmdir()
        return

    time_print(
        f"Filtered {total_removed} dependency rows for {system_name}; original files archived under {archive_root}."
    )
