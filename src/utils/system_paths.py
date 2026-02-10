"""
Utilities for locating per-system directories regardless of casing.

Many of our datasets were collected with inconsistent capitalization
(`OpenRGB` vs `openrgb`). These helpers make it easier to resolve the
actual directory used on disk while letting callers reason about a
canonical slug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_system_subdir(base_dir: str | Path, system_name: str, create: bool = False) -> Path:
    """
    Locate (or optionally create) the directory for ``system_name`` under ``base_dir``.

    Args:
        base_dir: Root directory that contains per-system folders.
        system_name: Requested system slug (case-insensitive).
        create: When True, create the directory if it does not already exist.

    Returns:
        Path to the resolved system directory (existing or newly created).

    Raises:
        FileNotFoundError: If the directory cannot be found and ``create`` is False.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        if create:
            base_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Base directory does not exist: {base_path}")

    candidate = base_path / system_name
    if candidate.exists():
        return candidate

    normalized = system_name.lower()
    matches: list[Path] = [
        entry for entry in base_path.iterdir() if entry.name.lower() == normalized
    ]
    if matches:
        # Prefer the first match (directory names are expected to be unique)
        return matches[0]

    if create:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    raise FileNotFoundError(
        f"Could not locate directory for system '{system_name}' under {base_path}"
    )


def ensure_system_file(
    base_dir: str | Path,
    system_name: str,
    relative_path: str | Path,
    create_parents: bool = False,
) -> Path:
    """
    Build a file path rooted at the resolved system directory.

    Args:
        base_dir: Root directory that contains per-system folders.
        system_name: Requested system slug (case-insensitive).
        relative_path: File path under the system directory.
        create_parents: Whether to create parent directories automatically.

    Returns:
        Path to the requested file.
    """
    system_dir = resolve_system_subdir(base_dir, system_name, create=create_parents)
    target = system_dir / relative_path
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target
