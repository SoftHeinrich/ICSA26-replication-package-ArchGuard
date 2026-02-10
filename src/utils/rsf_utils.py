#!/usr/bin/env python3
"""
RSF (Relation Set Format) Utilities

This module provides utilities for parsing and working with RSF files
used in architecture clustering analysis.
"""

import glob
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from src.utils.system_paths import resolve_system_subdir


class RSFParser:
    """Parser for RSF (Relation Set Format) files containing component clustering information."""

    def __init__(self):
        self.component_to_files = defaultdict(set)
        self.file_to_components = defaultdict(set)

    def parse_rsf_file(self, rsf_file_path: Path) -> Dict[str, Any]:
        """
        Parse an RSF file to extract component-file mappings.

        Args:
            rsf_file_path: Path to the RSF file

        Returns:
            Dictionary containing component mappings and statistics
        """
        component_to_files = defaultdict(set)
        file_to_components = defaultdict(set)

        try:
            with open(rsf_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # RSF format: relation source target
                    # e.g., "contain component_name file_path"
                    parts = line.split(None, 2)  # Split on whitespace, max 3 parts

                    if len(parts) >= 3 and parts[0] == 'contain':
                        component = parts[1]
                        file_path = parts[2]

                        component_to_files[component].add(file_path)
                        file_to_components[file_path].add(component)

        except Exception as e:
            print(f"Error parsing RSF file {rsf_file_path}: {e}")
            return {}

        return {
            'component_to_files': {k: list(v) for k, v in component_to_files.items()},
            'file_to_components': {k: list(v) for k, v in file_to_components.items()},
            'stats': {
                'total_components': len(component_to_files),
                'total_files': len(file_to_components),
                'total_mappings': sum(len(files) for files in component_to_files.values())
            }
        }

    def is_file_match(self, target_file: str, rsf_file: str) -> bool:
        """
        Check if target file matches RSF file entry with fuzzy matching.

        Args:
            target_file: File path from commit
            rsf_file: File path from RSF

        Returns:
            True if files match
        """
        # Normalize paths
        target_normalized = target_file.replace('\\', '/').strip('/')
        rsf_normalized = rsf_file.replace('\\', '/').strip('/')

        # Exact match
        if target_normalized == rsf_normalized:
            return True

        # Suffix match (file might have different prefix in different contexts)
        if target_normalized.endswith(rsf_normalized) or rsf_normalized.endswith(target_normalized):
            return True

        # Base filename match for cases where directory structure differs
        target_basename = Path(target_normalized).name
        rsf_basename = Path(rsf_normalized).name

        return target_basename == rsf_basename and target_basename != ''

    def map_files_to_components(self, affected_files: Set[str], rsf_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Map affected files to architectural components using RSF data.

        Args:
            affected_files: Set of affected file paths
            rsf_data: RSF parsing results

        Returns:
            Dictionary mapping files to components
        """
        file_to_components = rsf_data.get('file_to_components', {})
        component_mapping = {}

        for file_path in affected_files:
            components = []

            # Direct match
            if file_path in file_to_components:
                components.extend(file_to_components[file_path])
            else:
                # Try fuzzy matching for file paths
                for rsf_file, rsf_components in file_to_components.items():
                    if self.is_file_match(file_path, rsf_file):
                        components.extend(rsf_components)
                        break

            if components:
                component_mapping[file_path] = list(set(components))  # Remove duplicates

        return component_mapping


class VersionMatcher:
    """Matches MR versions to appropriate RSF files."""

    def __init__(self, subject_system: str, data_root: Path):
        self.subject_system = subject_system
        self.data_root = Path(data_root)
        self.cluster_algorithms = ['acdc', 'arc', 'pkg', 'limbo']
        self._version_numeric_pattern = re.compile(r'(\d+(?:[.\-]\d+)*)')

    def find_rsf_files(self) -> Dict[str, List[Path]]:
        """
        Find all available RSF cluster files for the subject system.

        Returns:
            Dictionary mapping algorithm names to lists of RSF file paths
        """
        rsf_files = {}

        try:
            system_cluster_dir = resolve_system_subdir(
                self.data_root / "clusters", self.subject_system, create=False
            )
        except FileNotFoundError:
            return {algorithm: [] for algorithm in self.cluster_algorithms}

        for algorithm in self.cluster_algorithms:
            algorithm_files = []

            # Search pattern: data-smell/clusters/{system}/{algorithm}/*/*.rsf
            pattern = str(system_cluster_dir / algorithm / "*" / "*.rsf")
            files = glob.glob(pattern)

            for file_path in files:
                algorithm_files.append(Path(file_path))

            rsf_files[algorithm] = sorted(algorithm_files)

        return rsf_files

    def extract_version_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract version from RSF filename.

        Args:
            filename: RSF filename

        Returns:
            Version string or None if not found
        """
        # Pattern: {system}-{version}_{algorithm}_clusters.rsf
        # e.g., "wireshark-non-patch-1.10.0_ACDC_clusters.rsf"
        #       "opencv-non-patch-2.4.0_IL_50_clusters.rsf"
        #       "opencv-non-patch-4.11.0_JS_50_clusters.rsf"
        pattern = rf"{re.escape(self.subject_system)}-(.+?)_[A-Z][A-Z0-9_]*_clusters\.rsf"
        match = re.search(pattern, filename)

        if match:
            return match.group(1)

        return None

    def select_best_rsf_file(self, mr_version: str, available_files: Dict[str, List[Path]],
                           preferred_algorithm: str = 'acdc') -> Optional[Path]:
        """
        Select the best RSF file for a given MR version.

        Args:
            mr_version: Target version for the MR
            available_files: Available RSF files by algorithm
            preferred_algorithm: Preferred clustering algorithm

        Returns:
            Path to the best matching RSF file or None
        """
        # Try preferred algorithm first
        if preferred_algorithm in available_files:
            best_file = self._find_version_match(mr_version, available_files[preferred_algorithm])
            if best_file:
                return best_file

        # Try other algorithms
        for algorithm in self.cluster_algorithms:
            if algorithm != preferred_algorithm and algorithm in available_files:
                best_file = self._find_version_match(mr_version, available_files[algorithm])
                if best_file:
                    return best_file

        return None

    def _find_version_match(self, target_version: str, files: List[Path]) -> Optional[Path]:
        """
        Find exact version match from a list of files.

        Args:
            target_version: Target version to match
            files: List of RSF file paths

        Returns:
            Matching file path or None if no exact match found
        """
        normalized_target = self._normalize_version(target_version)
        target_numeric_core = self._extract_numeric_core(target_version)

        for file_path in files:
            file_version = self.extract_version_from_filename(file_path.name)
            if not file_version:
                continue

            if file_version == target_version:
                return file_path

            normalized_file_version = self._normalize_version(file_version)
            if normalized_file_version and normalized_target and normalized_file_version == normalized_target:
                return file_path

            file_numeric_core = self._extract_numeric_core(file_version)
            if target_numeric_core and file_numeric_core and target_numeric_core == file_numeric_core:
                return file_path

        return None

    def _normalize_version(self, version: Optional[str]) -> Optional[str]:
        """Normalize version strings for consistent comparisons."""
        if not version:
            return None

        raw = version.strip()
        if not raw:
            return None

        match = self._version_numeric_pattern.search(raw)
        if not match:
            return raw.lower()

        prefix = raw[:match.start()].strip(" -_")
        numeric_part = match.group(1)
        numeric_part = re.sub(r'[^0-9]+', '.', numeric_part)
        segments = [segment for segment in numeric_part.split('.') if segment.isdigit()]
        if not segments:
            return raw.lower()

        while len(segments) < 3:
            segments.append('0')

        normalized_numeric = '.'.join(segments)
        if prefix:
            return f"{prefix.lower()}-{normalized_numeric}"
        return normalized_numeric

    def _extract_numeric_core(self, version: Optional[str]) -> Optional[str]:
        """Return the numeric portion of a normalized version string."""
        normalized = self._normalize_version(version)
        if not normalized:
            return None
        if '-' in normalized and not normalized.split('-', 1)[0].replace('.', '').isdigit():
            return normalized.split('-', 1)[1]
        return normalized
