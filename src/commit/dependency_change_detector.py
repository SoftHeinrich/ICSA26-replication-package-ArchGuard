#!/usr/bin/env python3
"""
Dependency Change Detector

Analyzes commit diffs to determine if they change file dependencies.
Supports multiple programming languages.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported programming languages."""
    C = "c"
    CPP = "c++"
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class DependencyChange:
    """Represents a dependency change in a file."""
    file_path: str
    added_dependencies: List[str]
    removed_dependencies: List[str]
    has_dependency_change: bool
    language: Language
    # For C/C++: track system headers (in <>) separately with counts
    added_system_headers: Optional[Dict[str, int]] = None
    removed_system_headers: Optional[Dict[str, int]] = None


class DependencyPatterns:
    """Regex patterns for detecting dependencies in different languages."""

    # C/C++: #include <...> (system) or #include "..." (local)
    # Captures bracket type and dependency name
    C_CPP_INCLUDE = re.compile(r'^\s*#include\s+([<"])([^>"]+)[>"]')

    # Python: import X, from X import Y
    PYTHON_IMPORT = re.compile(r'^\s*(?:import\s+([^\s]+)|from\s+([^\s]+)\s+import)')

    # Java: import X;
    JAVA_IMPORT = re.compile(r'^\s*import\s+([^;]+);')

    # JavaScript/TypeScript: import ... from '...', require('...')
    JS_IMPORT = re.compile(r'^\s*(?:import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]|(?:const|let|var)\s+.*?=\s*require\([\'"]([^\'"]+)[\'"]\))')

    # Go: import "..." or import (...)
    GO_IMPORT = re.compile(r'^\s*import\s+(?:[\'"]([^\'"]+)[\'"]|\()')

    # Rust: use ...;
    RUST_USE = re.compile(r'^\s*use\s+([^;]+);')


class DependencyChangeDetector:
    """Detects dependency changes in commit diffs."""

    def __init__(self):
        self.language_extensions = {
            '.c': Language.C,
            '.h': Language.C,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.cxx': Language.CPP,
            '.hpp': Language.CPP,
            '.hxx': Language.CPP,
            '.py': Language.PYTHON,
            '.java': Language.JAVA,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.JAVASCRIPT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.TYPESCRIPT,
            '.go': Language.GO,
            '.rs': Language.RUST,
        }

    def detect_language(self, file_path: str) -> Language:
        """Detect programming language from file extension."""
        for ext, lang in self.language_extensions.items():
            if file_path.endswith(ext):
                return lang
        return Language.UNKNOWN

    def extract_cpp_dependency(self, line: str) -> Optional[Tuple[str, bool]]:
        """
        Extract C/C++ dependency with system header flag.

        Returns:
            Tuple of (dependency_name, is_system_header) or None
            is_system_header is True for <>, False for ""
        """
        match = DependencyPatterns.C_CPP_INCLUDE.match(line)
        if match:
            bracket_type = match.group(1)  # '<' or '"'
            dependency_name = match.group(2)
            is_system_header = (bracket_type == '<')
            return (dependency_name, is_system_header)
        return None

    def extract_dependency_from_line(self, line: str, language: Language) -> Optional[str]:
        """Extract dependency statement from a line of code."""
        if language in (Language.C, Language.CPP):
            result = self.extract_cpp_dependency(line)
            if result:
                return result[0]  # Return just the dependency name

        elif language == Language.PYTHON:
            match = DependencyPatterns.PYTHON_IMPORT.match(line)
            if match:
                return match.group(1) or match.group(2)

        elif language == Language.JAVA:
            match = DependencyPatterns.JAVA_IMPORT.match(line)
            if match:
                return match.group(1).strip()

        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            match = DependencyPatterns.JS_IMPORT.match(line)
            if match:
                return match.group(1) or match.group(2)

        elif language == Language.GO:
            match = DependencyPatterns.GO_IMPORT.match(line)
            if match:
                return match.group(1)

        elif language == Language.RUST:
            match = DependencyPatterns.RUST_USE.match(line)
            if match:
                return match.group(1).strip()

        return None

    def parse_cpp_diff_changes(self, diff: str) -> Tuple[Set[str], Set[str], Dict[str, int], Dict[str, int]]:
        """
        Parse C/C++ diff to extract dependencies and count system headers.

        Returns:
            Tuple of (added_deps, removed_deps, added_system_counts, removed_system_counts)
        """
        added = set()
        removed = set()
        added_system_counts = {}
        removed_system_counts = {}

        for line in diff.split('\n'):
            # Added lines start with '+'
            if line.startswith('+') and not line.startswith('+++'):
                clean_line = line[1:]  # Remove the '+' prefix
                result = self.extract_cpp_dependency(clean_line)
                if result:
                    dep_name, is_system = result
                    added.add(dep_name)
                    if is_system:
                        added_system_counts[dep_name] = added_system_counts.get(dep_name, 0) + 1

            # Removed lines start with '-'
            elif line.startswith('-') and not line.startswith('---'):
                clean_line = line[1:]  # Remove the '-' prefix
                result = self.extract_cpp_dependency(clean_line)
                if result:
                    dep_name, is_system = result
                    removed.add(dep_name)
                    if is_system:
                        removed_system_counts[dep_name] = removed_system_counts.get(dep_name, 0) + 1

        return added, removed, added_system_counts, removed_system_counts

    def parse_diff_changes(self, diff: str, language: Language) -> Tuple[Set[str], Set[str]]:
        """
        Parse unified diff to extract added and removed dependencies.

        Returns:
            Tuple of (added_dependencies, removed_dependencies)
        """
        added = set()
        removed = set()

        for line in diff.split('\n'):
            # Added lines start with '+'
            if line.startswith('+') and not line.startswith('+++'):
                clean_line = line[1:]  # Remove the '+' prefix
                dep = self.extract_dependency_from_line(clean_line, language)
                if dep:
                    added.add(dep)

            # Removed lines start with '-'
            elif line.startswith('-') and not line.startswith('---'):
                clean_line = line[1:]  # Remove the '-' prefix
                dep = self.extract_dependency_from_line(clean_line, language)
                if dep:
                    removed.add(dep)

        return added, removed

    def analyze_file_diff(self, file_path: str, diff: str, language: Optional[Language] = None) -> DependencyChange:
        """
        Analyze a single file diff for dependency changes.

        Args:
            file_path: Path to the file
            diff: Unified diff string
            language: Optional language override (auto-detected if None)

        Returns:
            DependencyChange object
        """
        if language is None:
            language = self.detect_language(file_path)

        if language == Language.UNKNOWN:
            return DependencyChange(
                file_path=file_path,
                added_dependencies=[],
                removed_dependencies=[],
                has_dependency_change=False,
                language=language
            )

        # Use specialized C/C++ parser to track system headers
        if language in (Language.C, Language.CPP):
            added, removed, added_system, removed_system = self.parse_cpp_diff_changes(diff)
            return DependencyChange(
                file_path=file_path,
                added_dependencies=sorted(list(added)),
                removed_dependencies=sorted(list(removed)),
                has_dependency_change=len(added) > 0 or len(removed) > 0,
                language=language,
                added_system_headers=added_system if added_system else None,
                removed_system_headers=removed_system if removed_system else None
            )
        else:
            added, removed = self.parse_diff_changes(diff, language)
            return DependencyChange(
                file_path=file_path,
                added_dependencies=sorted(list(added)),
                removed_dependencies=sorted(list(removed)),
                has_dependency_change=len(added) > 0 or len(removed) > 0,
                language=language
            )

    def analyze_commit_diffs(self, commit_data: Dict, language: Optional[Language] = None) -> List[DependencyChange]:
        """
        Analyze all diffs in a commit.

        Args:
            commit_data: Commit data dictionary with 'full_diffs' key
            language: Optional language filter (analyze all if None)

        Returns:
            List of DependencyChange objects for files with dependency changes
        """
        results = []

        for file_diff in commit_data.get('full_diffs', []):
            file_path = file_diff.get('new_path') or file_diff.get('old_path')
            diff = file_diff.get('diff', '')

            if not file_path or not diff:
                continue

            file_language = self.detect_language(file_path)

            # Skip if language filter is set and doesn't match
            if language and file_language != language:
                continue

            dep_change = self.analyze_file_diff(file_path, diff, file_language)

            # Only include files with actual dependency changes
            if dep_change.has_dependency_change:
                results.append(dep_change)

        return results

    def analyze_commit_batch(self, batch_data: Dict, language: Optional[Language] = None) -> Dict[str, List[DependencyChange]]:
        """
        Analyze a batch of commits from a diff batch file.

        Args:
            batch_data: Dictionary with 'commit_diffs' key containing commit data
            language: Optional language filter

        Returns:
            Dictionary mapping commit_id to list of DependencyChange objects
        """
        results = {}

        for commit_id, commit_data in batch_data.get('commit_diffs', {}).items():
            changes = self.analyze_commit_diffs(commit_data, language)
            if changes:
                results[commit_id] = changes

        return results


def print_dependency_changes(changes: List[DependencyChange]) -> None:
    """Print dependency changes in a readable format."""
    for change in changes:
        print(f"\n{change.file_path} ({change.language.value}):")
        if change.added_dependencies:
            print(f"  Added dependencies:")
            for dep in change.added_dependencies:
                print(f"    + {dep}")
        if change.removed_dependencies:
            print(f"  Removed dependencies:")
            for dep in change.removed_dependencies:
                print(f"    - {dep}")

        # Print system header information for C/C++ files
        if change.added_system_headers:
            print(f"  Added system headers (in <>):")
            for header, count in sorted(change.added_system_headers.items()):
                print(f"    + {header} (count: {count})")
        if change.removed_system_headers:
            print(f"  Removed system headers (in <>):")
            for header, count in sorted(change.removed_system_headers.items()):
                print(f"    - {header} (count: {count})")


def main():
    """Example usage."""
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Detect dependency changes in commit diffs")
    parser.add_argument('--diff_file', help='Path to commit diff batch JSON file')
    parser.add_argument('--language', choices=[l.value for l in Language if l != Language.UNKNOWN],
                       help='Filter by programming language')
    parser.add_argument('--commit', help='Analyze specific commit ID only')

    args = parser.parse_args()

    # Load diff batch file
    with open(args.diff_file, 'r') as f:
        batch_data = json.load(f)

    # Create detector
    detector = DependencyChangeDetector()

    # Parse language if specified
    lang = Language(args.language) if args.language else None

    # Analyze commits
    if args.commit:
        # Analyze single commit
        commit_data = batch_data['commit_diffs'].get(args.commit)
        if not commit_data:
            print(f"Commit {args.commit} not found")
            return

        changes = detector.analyze_commit_diffs(commit_data, lang)
        print(f"\nCommit: {args.commit}")
        print(f"Title: {commit_data.get('commit_title')}")
        print(f"Files with dependency changes: {len(changes)}")
        print_dependency_changes(changes)
    else:
        # Analyze all commits
        results = detector.analyze_commit_batch(batch_data, lang)

        print(f"\nAnalyzed {len(batch_data['commit_diffs'])} commits")
        print(f"Commits with dependency changes: {len(results)}")

        for commit_id, changes in results.items():
            commit_data = batch_data['commit_diffs'][commit_id]
            print(f"\n{'='*80}")
            print(f"Commit: {commit_id}")
            print(f"Title: {commit_data.get('commit_title')}")
            print(f"MR: {commit_data.get('mr_iid')}")
            print(f"Files with dependency changes: {len(changes)}")
            print_dependency_changes(changes)


if __name__ == "__main__":
    main()
