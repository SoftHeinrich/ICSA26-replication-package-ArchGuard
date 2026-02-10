#!/usr/bin/env python3
"""
Batch Dependency Change Analyzer

Processes multiple commit diff batch files and generates a comprehensive
commit-to-dependency-change checklist.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from src.commit.dependency_change_detector import DependencyChangeDetector, DependencyChange, Language


@dataclass
class CommitDependencyInfo:
    """Information about a commit's dependency changes."""
    commit_id: str
    commit_title: str
    mr_iid: Optional[int]
    has_dependency_changes: bool
    dependency_changes: List[Dict]  # List of DependencyChange dicts
    total_files_changed: int
    files_with_dependency_changes: int


class BatchDependencyAnalyzer:
    """Analyzes multiple diff batch files for dependency changes."""

    def __init__(self):
        self.detector = DependencyChangeDetector()
        # Store raw DependencyChange objects for printing (not saved to JSON)
        self._raw_changes_by_commit = {}

    def process_batch_file(self, batch_file_path: str, language: Optional[Language] = None) -> Dict[str, CommitDependencyInfo]:
        """
        Process a single batch file.

        Args:
            batch_file_path: Path to the batch JSON file
            language: Optional language filter

        Returns:
            Dictionary mapping commit_id to CommitDependencyInfo
        """
        with open(batch_file_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)

        results = {}

        for commit_id, commit_data in batch_data.get('commit_diffs', {}).items():
            changes = self.detector.analyze_commit_diffs(commit_data, language)

            # Store raw changes for printing (includes system header info)
            if changes:
                self._raw_changes_by_commit[commit_id] = changes

            # Convert DependencyChange objects to dicts for JSON serialization
            changes_dicts = [
                {
                    'file_path': c.file_path,
                    'added_dependencies': c.added_dependencies,
                    'removed_dependencies': c.removed_dependencies,
                    'has_dependency_change': c.has_dependency_change,
                    'language': c.language.value
                }
                for c in changes
            ]

            commit_info = CommitDependencyInfo(
                commit_id=commit_id,
                commit_title=commit_data.get('commit_title', 'N/A'),
                mr_iid=commit_data.get('mr_iid'),
                has_dependency_changes=len(changes) > 0,
                dependency_changes=changes_dicts,
                total_files_changed=len(commit_data.get('full_diffs', [])),
                files_with_dependency_changes=len(changes)
            )

            results[commit_id] = commit_info

        return results

    def process_directory(self, directory: str, pattern: str = "issues_all_diffs_batch_*.json",
                          language: Optional[Language] = None) -> Dict[str, CommitDependencyInfo]:
        """
        Process all batch files in a directory.

        Args:
            directory: Directory containing batch files
            pattern: Glob pattern for batch files
            language: Optional language filter

        Returns:
            Dictionary mapping commit_id to CommitDependencyInfo
        """
        directory_path = Path(directory)
        batch_files = sorted(directory_path.glob(pattern))

        if not batch_files:
            print(f"No files matching pattern '{pattern}' found in {directory}")
            return {}

        print(f"Found {len(batch_files)} batch files to process")

        all_commits = {}

        for i, batch_file in enumerate(batch_files, 1):
            print(f"Processing [{i}/{len(batch_files)}]: {batch_file.name}")

            try:
                batch_results = self.process_batch_file(str(batch_file), language)
                all_commits.update(batch_results)
            except Exception as e:
                print(f"  Error processing {batch_file.name}: {e}")
                continue

        return all_commits

    def categorize_commits(self, commits: Dict[str, CommitDependencyInfo]) -> Dict:
        """
        Categorize commits by dependency type and collect statistics.

        Returns:
            Dictionary with categorization results and filtered commits
        """
        system_header_freq = defaultdict(int)
        commits_only_nonsys = {}
        commits_only_sys = {}
        commits_both = {}

        for commit_id, info in commits.items():
            raw_changes = self._raw_changes_by_commit.get(commit_id, [])

            has_sys_changes = False
            has_nonsys_changes = False
            nonsys_changes = []  # Filtered changes with only non-system deps

            for raw_change in raw_changes:
                # Track system header changes
                if raw_change.added_system_headers:
                    has_sys_changes = True
                    for header in raw_change.added_system_headers.keys():
                        system_header_freq[header] += 1
                if raw_change.removed_system_headers:
                    has_sys_changes = True
                    for header in raw_change.removed_system_headers.keys():
                        system_header_freq[header] += 1

                # Check for non-system changes
                nonsys_added = []
                nonsys_removed = []

                if raw_change.language in (Language.C, Language.CPP):
                    # For C/C++, filter out system headers
                    sys_headers = set()
                    if raw_change.added_system_headers:
                        sys_headers.update(raw_change.added_system_headers.keys())
                    if raw_change.removed_system_headers:
                        sys_headers.update(raw_change.removed_system_headers.keys())

                    nonsys_added = [d for d in raw_change.added_dependencies if d not in sys_headers]
                    nonsys_removed = [d for d in raw_change.removed_dependencies if d not in sys_headers]
                else:
                    # For non-C/C++ languages, all changes are non-system
                    nonsys_added = raw_change.added_dependencies
                    nonsys_removed = raw_change.removed_dependencies

                if nonsys_added or nonsys_removed:
                    has_nonsys_changes = True
                    nonsys_changes.append({
                        'file_path': raw_change.file_path,
                        'added_dependencies': nonsys_added,
                        'removed_dependencies': nonsys_removed,
                        'has_dependency_change': True,
                        'language': raw_change.language.value
                    })

            # Categorize commit
            if has_sys_changes and has_nonsys_changes:
                # Store only non-system changes for cross-cutting commits
                modified_info = CommitDependencyInfo(
                    commit_id=info.commit_id,
                    commit_title=info.commit_title,
                    mr_iid=info.mr_iid,
                    has_dependency_changes=True,
                    dependency_changes=nonsys_changes,
                    total_files_changed=info.total_files_changed,
                    files_with_dependency_changes=len(nonsys_changes)
                )
                commits_both[commit_id] = modified_info
            elif has_sys_changes:
                commits_only_sys[commit_id] = info
            elif has_nonsys_changes:
                commits_only_nonsys[commit_id] = info

        return {
            'system_header_freq': dict(system_header_freq),
            'commits_only_nonsys': commits_only_nonsys,
            'commits_only_sys': commits_only_sys,
            'commits_both': commits_both
        }

    def generate_checklist(self, commits: Dict[str, CommitDependencyInfo],
                          output_file: Optional[str] = None,
                          only_with_changes: bool = True) -> Dict:
        """
        Generate a comprehensive dependency change checklist.

        Only saves commits with non-system dependency changes.
        For commits with both system and non-system changes, only non-system changes are saved.

        Args:
            commits: Dictionary of commit dependency info
            output_file: Optional path to save JSON output
            only_with_changes: Only include commits with dependency changes

        Returns:
            Dictionary containing the checklist and categorization info
        """
        # Categorize commits by dependency type
        categorization = self.categorize_commits(commits)

        # Only save commits with non-system dependency changes
        # This includes: commits_only_nonsys + commits_both (with filtered changes)
        commits_to_save = {}
        commits_to_save.update(categorization['commits_only_nonsys'])
        commits_to_save.update(categorization['commits_both'])

        # Filter commits if requested
        if only_with_changes:
            filtered_commits = commits_to_save
        else:
            # Include all original commits
            filtered_commits = commits

        # Calculate statistics
        total_nonsys_commits = len(categorization['commits_only_nonsys']) + len(categorization['commits_both'])

        # Create checklist structure
        checklist = {
            'summary': {
                'total_commits_analyzed': len(commits),
                'commits_with_dependency_changes': sum(1 for c in commits.values() if c.has_dependency_changes),
                'commits_without_dependency_changes': sum(1 for c in commits.values() if not c.has_dependency_changes),
                'total_files_with_dependency_changes': sum(c.files_with_dependency_changes for c in commits.values()),
                # Add categorization statistics
                'commits_only_nonsys_changes': len(categorization['commits_only_nonsys']),
                'commits_only_sys_changes': len(categorization['commits_only_sys']),
                'commits_both_types': len(categorization['commits_both']),
                'commits_saved_to_json': total_nonsys_commits
            },
            'commits': {
                commit_id: asdict(info)
                for commit_id, info in filtered_commits.items()
            },
            # Store categorization for printing summary
            '_categorization': categorization
        }

        # Create output data (without internal categorization) for API access
        output_data = {
            'summary': checklist['summary'],
            'commits': checklist['commits']
        }

        # Store for API access
        self._last_output_data = output_data

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nChecklist saved to: {output_file}")
            print(f"  Commits with non-system dependency changes: {total_nonsys_commits}")
            print(f"  Commits with only system header changes (excluded): {len(categorization['commits_only_sys'])}")

        return checklist

    def get_output_data(self) -> Optional[Dict]:
        """
        Get the output data from the last generate_checklist() call.

        This returns the JSON-serializable output data (without internal categorization)
        that is saved to file when output_file is specified in generate_checklist().

        Returns:
            Dictionary with 'summary' and 'commits' keys, or None if generate_checklist()
            has not been called yet.

        Example:
            >>> analyzer = BatchDependencyAnalyzer()
            >>> commits = analyzer.process_directory('data/')
            >>> checklist = analyzer.generate_checklist(commits)
            >>> output_data = analyzer.get_output_data()
            >>> print(output_data['summary']['total_commits_analyzed'])
        """
        return getattr(self, '_last_output_data', None)

    def get_dependency_change_commits(self, checklist: Dict) -> List[str]:
        """
        Get list of commit IDs with dependency changes.

        Args:
            checklist: Checklist dictionary from generate_checklist()

        Returns:
            List of commit IDs that have dependency changes
        """
        return [
            commit_id
            for commit_id, info in checklist['commits'].items()
            if info.get('has_dependency_changes', False)
        ]

    def get_dependency_change_mrs(self, checklist: Dict) -> List[int]:
        """
        Get list of MR IIDs with dependency changes.

        Args:
            checklist: Checklist dictionary from generate_checklist()

        Returns:
            List of unique MR IIDs that have dependency changes
        """
        mr_iids = set()
        for commit_id, info in checklist['commits'].items():
            if info.get('has_dependency_changes', False):
                mr_iid = info.get('mr_iid')
                if mr_iid is not None:
                    mr_iids.add(mr_iid)
        return sorted(list(mr_iids))

    def get_commit_to_mr_mapping(self, checklist: Dict) -> Dict[str, int]:
        """
        Get mapping from commit IDs to MR IIDs.

        Args:
            checklist: Checklist dictionary from generate_checklist()

        Returns:
            Dictionary mapping commit_id -> mr_iid
        """
        return {
            commit_id: info['mr_iid']
            for commit_id, info in checklist['commits'].items()
            if info.get('mr_iid') is not None
        }

    def get_mr_to_commits_mapping(self, checklist: Dict) -> Dict[int, List[str]]:
        """
        Get mapping from MR IIDs to commit IDs.

        Args:
            checklist: Checklist dictionary from generate_checklist()

        Returns:
            Dictionary mapping mr_iid -> list of commit_ids
        """
        mr_commits = defaultdict(list)
        for commit_id, info in checklist['commits'].items():
            mr_iid = info.get('mr_iid')
            if mr_iid is not None:
                mr_commits[mr_iid].append(commit_id)
        return dict(mr_commits)

    def print_summary(self, checklist: Dict) -> None:
        """Print a human-readable summary of the checklist."""
        summary = checklist['summary']
        categorization = checklist.get('_categorization', {})

        print("\n" + "="*80)
        print("DEPENDENCY CHANGE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total commits analyzed: {summary['total_commits_analyzed']}")
        print(f"Commits WITH dependency changes: {summary['commits_with_dependency_changes']}")
        print(f"Commits WITHOUT dependency changes: {summary['commits_without_dependency_changes']}")
        print(f"Total files with dependency changes: {summary['total_files_with_dependency_changes']}")
        print("="*80)

        # Get statistics from categorization
        system_header_freq = categorization.get('system_header_freq', {})
        commits_only_nonsys = summary.get('commits_only_nonsys_changes', 0)
        commits_only_sys = summary.get('commits_only_sys_changes', 0)
        commits_both = summary.get('commits_both_types', 0)

        # Print C/C++ dependency statistics
        if system_header_freq:
            print("\nC/C++ System Header Change Frequency:")
            print("-"*80)
            print(f"{'Header':<40} {'Commits':>10}")
            print("-"*80)

            # Sort by frequency (descending)
            sorted_headers = sorted(system_header_freq.items(), key=lambda x: x[1], reverse=True)
            for header, count in sorted_headers[:30]:  # Show top 30
                print(f"{header:<40} {count:>10}")

            if len(sorted_headers) > 30:
                print(f"\n... and {len(sorted_headers) - 30} more headers")

        # Print commit categorization
        print("\n" + "="*80)
        print("COMMIT CATEGORIZATION BY DEPENDENCY TYPE:")
        print("="*80)
        print(f"Commits with ONLY non-system dependency changes: {commits_only_nonsys}")
        print(f"Commits with ONLY system header changes: {commits_only_sys}")
        print(f"Commits with BOTH types of changes: {commits_both}")
        print("-"*80)
        print(f"Commits saved to JSON (non-system changes only): {summary.get('commits_saved_to_json', 0)}")
        print(f"  = Only non-system: {commits_only_nonsys}")
        print(f"  + Both (filtered to non-system only): {commits_both}")
        print("="*80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze multiple commit diff batch files for dependency changes"
    )
    parser.add_argument(
        'directory',
        help='Directory containing diff batch JSON files'
    )
    parser.add_argument(
        '--pattern',
        default='issues_all_diffs_batch_*.json',
        help='Glob pattern for batch files (default: issues_all_diffs_batch_*.json)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output JSON file path for the checklist'
    )
    parser.add_argument(
        '--language',
        choices=[l.value for l in Language if l != Language.UNKNOWN],
        help='Filter by programming language'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include commits without dependency changes in the output'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing the summary'
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        return 1

    # Parse language if specified
    lang = Language(args.language) if args.language else None

    # Create analyzer
    analyzer = BatchDependencyAnalyzer()

    # Process all batch files
    print(f"Analyzing dependency changes in: {args.directory}")
    if lang:
        print(f"Language filter: {lang.value}")
    print()

    commits = analyzer.process_directory(args.directory, args.pattern, lang)

    if not commits:
        print("No commits found to analyze")
        return 1

    # Generate checklist
    checklist = analyzer.generate_checklist(
        commits,
        output_file=args.output,
        only_with_changes=not args.include_all
    )

    # Print summary unless suppressed
    if not args.no_summary:
        analyzer.print_summary(checklist)

    return 0


if __name__ == "__main__":
    exit(main())
