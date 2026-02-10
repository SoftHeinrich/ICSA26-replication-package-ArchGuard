"""
Helpers for locating fact extraction artifacts independent of casing.

A number of our legacy fact directories were generated with mixed-case
system prefixes (for example ``OpenRGB-0.8_deps.rsf`` even though the
canonical slug is ``openrgb``).  The clustering and smell pipelines now
rely on this module so they can accept the normalized slug while still
opening the correct files on disk.
"""

from __future__ import annotations

from pathlib import Path

from src.preprocessing.constants import FACTS_ROOT
from src.utils.system_paths import resolve_system_subdir


def resolve_facts_rsf(system_name: str, version_slug: str) -> Path:
    """
    Find the ``*_deps.rsf`` file for ``system_name`` and ``version_slug``.

    Args:
        system_name: Canonical system slug (e.g., ``openrgb``).
        version_slug: Either the bare version (``0.8``) or the fully
            qualified prefix used by ARCADE (``openrgb-0.8``).

    Returns:
        Path to the matching RSF file.

    Raises:
        FileNotFoundError: When the RSF file cannot be resolved.
    """
    facts_dir = resolve_system_subdir(FACTS_ROOT, system_name, create=False)
    normalized_slug = version_slug.strip()
    prefix = f"{system_name.lower()}-"

    if not normalized_slug.lower().startswith(prefix):
        normalized_slug = f"{system_name}-{normalized_slug}"

    target_name = f"{normalized_slug}_deps.rsf"
    target_lower = target_name.lower()

    for entry in facts_dir.glob("*_deps.rsf"):
        if entry.name.lower() == target_lower:
            return entry

    raise FileNotFoundError(
        f"Could not locate facts file '{target_name}' under {facts_dir}. "
        "Ensure fact extraction has completed for this system/version."
    )

