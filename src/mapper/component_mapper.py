#!/usr/bin/env python3
"""
Component Mapper

Maps merge requests to affected architectural components by analyzing commit files
and using version-specific architecture clustering RSF files.

This module:
1. Processes issues and their related merge requests
2. Analyzes commit data to identify affected files
3. Selects appropriate RSF clustering files based on MR version
4. Maps files to architectural components using RSF cluster data
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import glob
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.release_utils import detect_subject_system
from src.utils.rsf_utils import RSFParser, VersionMatcher
from src.commit.batch_dependency_analyzer import BatchDependencyAnalyzer
from src.mapper.version_stats import compute_version_stats


class ComponentMapper:
    """Maps merge requests to affected architectural components."""

    def __init__(self, issue_dir: Path, subject_system: str = None, data_root: Path = None, algorithm: str = 'acdc'):
        """
        Initialize the Component Mapper.

        Args:
            issue_dir: Directory containing issue data files and batch diff files
            subject_system: Name of the subject system (auto-detected if None)
            data_root: Root directory for data files (default: project root)
            algorithm: RSF clustering algorithm to use (default: 'acdc')
        """
        self.issue_dir = Path(issue_dir)
        self.subject_system = subject_system or detect_subject_system(str(issue_dir))
        self.data_root = data_root or project_root
        self.algorithm = algorithm

        self.rsf_parser = RSFParser()
        self.version_matcher = VersionMatcher(self.subject_system, self.data_root)

        # Cache for RSF data - pre-parse all RSF files at initialization
        self.rsf_cache = {}

        # Dependency change checklist - mapping commit_id -> Set[file_paths] with dependency changes
        self.dependency_checklist = {}

        # Statistics
        self.stats = {
            'issues_processed': 0,
            'mrs_processed': 0,
            'mrs_with_components': 0,
            'mrs_with_commits': 0,
            'commits_total': 0,
            'commits_with_dep_changes': 0,
            'commits_filtered_out': 0,
            'files_total': 0,
            'files_with_dep_changes': 0,
            'files_filtered_out': 0,
            'unique_components': set(),
            'rsf_files_used': set(),
            'errors': []
        }

        print(f"Initialized Component Mapper for system: {self.subject_system}")
        print(f"Issue directory: {self.issue_dir}")
        print(f"Data root: {self.data_root}")
        print(f"Algorithm: {self.algorithm}")

        # Generate dependency change checklist from batch diff files
        self._generate_dependency_checklist()

        # Pre-parse all RSF files for the selected algorithm
        self._preload_rsf_files()


    @staticmethod
    def _process_issue_file_static(file_path: Path, subject_system: str, data_root: Path, algorithm: str,
                                   rsf_cache: Dict[str, Dict[str, Any]], dependency_checklist: Dict[str, bool]) -> Dict[str, Any]:
        """
        Static method to process a single issue file (for parallel processing).

        Args:
            file_path: Path to issue data file
            subject_system: Subject system name
            data_root: Data root path
            algorithm: RSF clustering algorithm
            rsf_cache: Pre-loaded RSF cache to share across workers
            dependency_checklist: Pre-loaded dependency checklist to share across workers

        Returns:
            Component mapping results for all issues in the file
        """
        # Create a new mapper instance for this worker
        mapper = ComponentMapper.__new__(ComponentMapper)
        mapper.issue_dir = file_path.parent
        mapper.subject_system = subject_system
        mapper.data_root = data_root
        mapper.algorithm = algorithm
        mapper.rsf_parser = RSFParser()
        mapper.version_matcher = VersionMatcher(subject_system, data_root)
        mapper.rsf_cache = rsf_cache  # Share pre-loaded cache
        mapper.dependency_checklist = dependency_checklist  # Share dependency checklist
        mapper.stats = {
            'issues_processed': 0,
            'mrs_processed': 0,
            'mrs_with_components': 0,
            'mrs_with_commits': 0,
            'commits_total': 0,
            'commits_with_dep_changes': 0,
            'commits_filtered_out': 0,
            'files_total': 0,
            'files_with_dep_changes': 0,
            'files_filtered_out': 0,
            'unique_components': set(),
            'rsf_files_used': set(),
            'errors': []
        }

        return mapper.process_issue_file(file_path)

    def _update_stats_from_file_result(self, file_result: Dict[str, Any]):
        """
        Update global statistics from a file processing result.

        Args:
            file_result: File processing result dictionary
        """
        if 'error' in file_result:
            return

        self.stats['issues_processed'] += file_result.get('total_issues', 0)

        for issue_result in file_result.get('issue_results', []):
            if 'error' in issue_result:
                continue

            for mr_result in issue_result.get('mr_results', []):
                self.stats['mrs_processed'] += 1

                if mr_result.get('processing_status') == 'success':
                    if mr_result.get('affected_components'):
                        self.stats['mrs_with_components'] += 1

                if mr_result.get('processing_status') in ['success', 'no_files', 'no_rsf_file']:
                    self.stats['mrs_with_commits'] += 1

                if mr_result.get('affected_components'):
                    self.stats['unique_components'].update(mr_result['affected_components'])

                if mr_result.get('rsf_file_used'):
                    self.stats['rsf_files_used'].add(mr_result['rsf_file_used'])

    def _preload_rsf_files(self):
        """Pre-parse all RSF files for the selected algorithm at initialization."""
        print(f"\nPre-loading RSF files for algorithm: {self.algorithm}")

        # Find all RSF files for the selected algorithm
        available_files = self.version_matcher.find_rsf_files()

        if self.algorithm not in available_files:
            print(f"[WARN] No RSF files found for algorithm: {self.algorithm}")
            return

        rsf_files = available_files[self.algorithm]
        print(f"Found {len(rsf_files)} RSF files to pre-load")

        # Parse all RSF files and cache them
        for rsf_file in rsf_files:
            cache_key = str(rsf_file)
            print(f"  Parsing: {rsf_file.name}...", end='')
            self.rsf_cache[cache_key] = self.rsf_parser.parse_rsf_file(rsf_file)
            print(" OK")

        print(f"Pre-loaded {len(self.rsf_cache)} RSF files into memory\n")

    def _generate_dependency_checklist(self):
        """Generate dependency change checklist from batch diff files with file-level information."""
        print(f"\nGenerating dependency change checklist...")

        # Find batch diff files
        batch_diff_pattern = str(self.issue_dir / 'issues_all_diffs_batch_*.json')
        batch_files = sorted(glob.glob(batch_diff_pattern))

        if not batch_files:
            print(f"[WARN] No batch diff files found in {self.issue_dir}")
            print(f"    Continuing without dependency filtering...")
            return

        print(f"Found {len(batch_files)} batch diff files")

        # Use BatchDependencyAnalyzer to process all batch files
        analyzer = BatchDependencyAnalyzer()
        commits = analyzer.process_directory(str(self.issue_dir), pattern='issues_all_diffs_batch_*.json')

        if not commits:
            print(f"[WARN] No commits found in batch diff files")
            return

        # Generate checklist
        analyzer.generate_checklist(commits, only_with_changes=True)

        # Get clean output data using the new API
        output_data = analyzer.get_output_data()

        # Build lookup dictionary: commit_id -> Set[file_paths] with dependency changes
        total_files_with_changes = 0
        for commit_id, commit_info in output_data['commits'].items():
            # Extract files with dependency changes for this commit
            files_with_dep_changes = set()
            for dep_change in commit_info.get('dependency_changes', []):
                if dep_change.get('has_dependency_change', False):
                    files_with_dep_changes.add(dep_change['file_path'])

            self.dependency_checklist[commit_id] = files_with_dep_changes
            total_files_with_changes += len(files_with_dep_changes)

        # Print summary
        summary = output_data['summary']
        total_commits = summary['total_commits_analyzed']
        commits_with_changes = summary['commits_with_dependency_changes']
        total_files = summary['total_files_with_dependency_changes']

        print(f"Dependency checklist generated:")
        print(f"   Total commits: {total_commits}")
        print(f"   Commits with dependency changes: {commits_with_changes}")
        print(f"   Commits without dependency changes: {total_commits - commits_with_changes}")
        print(f"   Total files with dependency changes: {total_files}\n")

    def filter_commits_by_dependency_changes(self, commits: List[Dict]) -> Tuple[List[Dict], Set[str]]:
        """
        Filter commits to only those with dependency changes and collect files with dependency changes.

        Args:
            commits: List of commit data from MR

        Returns:
            Tuple of (filtered commits, set of file paths with dependency changes)

        Raises:
            RuntimeError: If dependency checklist is not available
        """
        if not self.dependency_checklist:
            raise RuntimeError("Dependency checklist not available. Cannot perform file-level filtering.")

        filtered_commits = []
        files_with_dep_changes = set()

        for commit in commits:
            # Get commit_id
            commit_id = commit['commit_metadata']['id']

            # Get files with dependency changes for this commit
            commit_dep_files = self.dependency_checklist.get(commit_id, set())

            # Only include commit if it has at least one file with dependency changes
            if commit_dep_files:
                filtered_commits.append(commit)
                files_with_dep_changes.update(commit_dep_files)

        return filtered_commits, files_with_dep_changes

    def get_rsf_data(self, rsf_file_path: Path) -> Dict[str, Any]:
        """
        Get RSF data from cache (already pre-loaded).

        Args:
            rsf_file_path: Path to RSF file

        Returns:
            RSF parsing results
        """
        cache_key = str(rsf_file_path)

        if cache_key in self.rsf_cache:
            self.stats['rsf_files_used'].add(rsf_file_path.name)
            return self.rsf_cache[cache_key]

        # Fallback: parse on demand if not in cache (shouldn't happen normally)
        print(f"  [WARN] RSF file not pre-loaded, parsing on demand: {rsf_file_path.name}")
        rsf_data = self.rsf_parser.parse_rsf_file(rsf_file_path)
        self.rsf_cache[cache_key] = rsf_data
        self.stats['rsf_files_used'].add(rsf_file_path.name)
        return rsf_data

    def extract_affected_files_from_commits(self, commits: List[Dict], files_with_dep_changes: Set[str]) -> Tuple[Set[str], int, int]:
        """
        Extract affected files from commit data, filtering to only files with dependency changes.

        Args:
            commits: List of commit data
            files_with_dep_changes: Set of file paths that have dependency changes

        Returns:
            Tuple of (filtered file paths, total files, files filtered out)
        """
        all_files = set()
        for commit in commits:
            # Check for commit_diff_summary (enriched format)
            diff_summary = commit['commit_diff_summary']
            for file_info in diff_summary['files_summary']:
                all_files.add(file_info['new_path'])

        # Filter to only files with dependency changes
        filtered_files = all_files & files_with_dep_changes

        total_files = len(all_files)
        files_filtered_out = total_files - len(filtered_files)

        return filtered_files, total_files, files_filtered_out

    def map_files_to_components(self, affected_files: Set[str], rsf_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Map affected files to architectural components using RSF data.

        Args:
            affected_files: Set of affected file paths
            rsf_data: RSF parsing results

        Returns:
            Dictionary mapping files to components
        """
        return self.rsf_parser.map_files_to_components(affected_files, rsf_data)

    def process_mr(self, mr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single merge request to map it to affected components.

        Args:
            mr_data: Merge request data

        Returns:
            Component mapping results for this MR
        """
        mr_iid = mr_data['mr_metadata'].get('iid', 'unknown')
        result = {
            'mr_iid': mr_iid,
            'affected_components': [],
            'affected_files': [],
            'file_to_components': {},
            'rsf_file_used': None,
            'mr_version': None,
            'processing_status': 'failed'
        }

        try:
            # Get MR metadata
            mr_metadata = mr_data.get('mr_metadata', {})

            # Determine MR version (from existing version info or compute from merge date)
            mr_version = None
            if 'mr_version' in mr_metadata:
                mr_version = mr_metadata['mr_version']
            elif 'version_info' in mr_data and 'mr_version' in mr_data['version_info']:
                mr_version = mr_data['version_info']['mr_version']

            if not mr_version or mr_version == 'Unknown':
                result['processing_status'] = 'no_version'
                return result
            if mr_version == "not yet":
                result['processing_status'] = 'not_released_yet'
                return result

            result['mr_version'] = mr_version

            # Get commits data
            commits = mr_data.get('commits', [])

            self.stats['mrs_with_commits'] += 1

            # Track total commits
            total_commits = len(commits)
            self.stats['commits_total'] += total_commits

            # Filter commits to only those with dependency changes and get files with dep changes
            filtered_commits, files_with_dep_changes = self.filter_commits_by_dependency_changes(commits)
            commits_with_dep_changes = len(filtered_commits)
            self.stats['commits_with_dep_changes'] += commits_with_dep_changes
            self.stats['commits_filtered_out'] += (total_commits - commits_with_dep_changes)

            # Store commit filtering info in result
            result['total_commits'] = total_commits
            result['commits_with_dep_changes'] = commits_with_dep_changes
            result['commits_filtered_out'] = total_commits - commits_with_dep_changes

            if not filtered_commits:
                result['processing_status'] = 'no_dep_change_commits'
                return result

            # Extract affected files from filtered commits, filtering to only files with dependency changes
            affected_files, total_files, files_filtered_out = self.extract_affected_files_from_commits(
                filtered_commits, files_with_dep_changes
            )

            # Track file-level statistics
            self.stats['files_total'] += total_files
            self.stats['files_with_dep_changes'] += len(affected_files)
            self.stats['files_filtered_out'] += files_filtered_out

            # Store file filtering info in result
            result['total_files'] = total_files
            result['files_with_dep_changes'] = len(affected_files)
            result['files_filtered_out'] = files_filtered_out
            result['affected_files'] = list(affected_files)

            if not affected_files:
                result['processing_status'] = 'no_files_with_dep_changes'
                return result

            # Find appropriate RSF file for this version
            available_rsf_files = self.version_matcher.find_rsf_files()
            rsf_file = self.version_matcher.select_best_rsf_file(mr_version, available_rsf_files, preferred_algorithm=self.algorithm)

            if not rsf_file:
                result['processing_status'] = 'no_rsf_file'
                return result

            result['rsf_file_used'] = rsf_file.name

            # Parse RSF file and map files to components
            rsf_data = self.get_rsf_data(rsf_file)
            if not rsf_data:
                result['processing_status'] = 'rsf_parse_error'
                return result

            # Map files to components
            file_to_components = self.map_files_to_components(affected_files, rsf_data)
            result['file_to_components'] = file_to_components

            # Collect all affected components
            all_components = set()
            for components in file_to_components.values():
                all_components.update(components)

            result['affected_components'] = list(all_components)
            result['processing_status'] = 'success'

            if all_components:
                self.stats['mrs_with_components'] += 1
                self.stats['unique_components'].update(all_components)

        except Exception as e:
            error_msg = f"Error processing MR !{mr_iid}: {e}"
            self.stats['errors'].append(error_msg)
            result['processing_status'] = 'error'
            result['error'] = str(e)

        return result

    def process_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single issue and map all its related MRs to components.

        Args:
            issue_data: Issue data containing related_mrs

        Returns:
            Component mapping results for all MRs in this issue
        """
        try:
            # Extract issue metadata
            # if 'issue_data' in issue_data:
            issue_metadata = issue_data['issue_data'].get('issue_metadata', {})
            related_mrs = issue_data.get('related_mrs', [])
            # else:
            #     issue_metadata = issue_data.get('issue_metadata', {})
            #     related_mrs = issue_data.get('related_mrs', [])

            issue_id = issue_metadata.get('iid', 'unknown')

            # Process each MR
            mr_results = []
            for mr in related_mrs:
                mr_result = self.process_mr(mr)
                mr_results.append(mr_result)
                self.stats['mrs_processed'] += 1

            # Aggregate components by version
            version_to_components = defaultdict(set)
            for mr in mr_results:
                if mr['processing_status'] == 'success' and mr.get('mr_version'):
                    version = mr['mr_version']
                    components = mr.get('affected_components', [])
                    version_to_components[version].update(components)

            # Convert to list format
            aggregated_components_by_version = [
                {
                    'version': version,
                    'components': sorted(list(components))
                }
                for version, components in sorted(version_to_components.items())
            ]

            # Aggregate results for the issue
            issue_result = {
                'issue_id': issue_id,
                'total_mrs': len(related_mrs),
                'mrs_with_components': sum(1 for mr in mr_results if mr['processing_status'] == 'success' and mr['affected_components']),
                'mr_results': mr_results,
                'aggregated_components_by_version': aggregated_components_by_version,
                'processing_timestamp': datetime.now().isoformat()
            }

            return issue_result

        except Exception as e:
            error_msg = f"Error processing issue: {e}"
            self.stats['errors'].append(error_msg)
            return {
                'issue_id': 'unknown',
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }

    def process_issue_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single issue data file.

        Args:
            file_path: Path to issue data file

        Returns:
            Component mapping results for all issues in the file
        """
        print(f"\nProcessing file: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            issues = data.get('issues', [])
            print(f"  Found {len(issues)} issues")

            file_results = {
                'file_name': file_path.name,
                'total_issues': len(issues),
                'issue_results': [],
                'processing_timestamp': datetime.now().isoformat()
            }

            for issue_data in issues:
                issue_result = self.process_issue(issue_data)
                file_results['issue_results'].append(issue_result)
                self.stats['issues_processed'] += 1

            return file_results

        except Exception as e:
            error_msg = f"Error processing file {file_path}: {e}"
            self.stats['errors'].append(error_msg)
            return {
                'file_name': file_path.name,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }

    def process_all_issues(self, file_pattern: str = "*.json", max_workers: int = 8) -> Dict[str, Any]:
        """
        Process all issue files in the issue directory with parallelization.

        Args:
            file_pattern: Glob pattern for issue files
            max_workers: Maximum number of parallel workers (default: 8)

        Returns:
            Complete component mapping results
        """
        print(f"\n{'='*60}")
        print("COMPONENT MAPPING PROCESS")
        print(f"{'='*60}")

        # Find issue files
        search_pattern = str(self.issue_dir / file_pattern)
        files = sorted(glob.glob(search_pattern))

        if not files:
            print(f" No files found matching pattern: {search_pattern}")
            return {'error': 'No input files found'}

        print(f"Found {len(files)} files to process")
        print(f"Using {max_workers} parallel workers")

        # Process each file
        all_results = {
            'subject_system': self.subject_system,
            'algorithm': self.algorithm,
            'processing_timestamp': datetime.now().isoformat(),
            'file_results': []
        }

        # Process files in parallel
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all file processing tasks, sharing the pre-loaded RSF cache and dependency checklist
                future_to_file = {
                    executor.submit(self._process_issue_file_static,
                                  Path(file_path),
                                  self.subject_system,
                                  self.data_root,
                                  self.algorithm,
                                  self.rsf_cache,
                                  self.dependency_checklist): file_path
                    for file_path in files
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_result = future.result()
                        all_results['file_results'].append(file_result)

                        # Update stats from file result
                        self._update_stats_from_file_result(file_result)

                        print(f"[OK] Completed: {Path(file_path).name}")
                    except Exception as e:
                        error_msg = f"Error processing file {file_path}: {e}"
                        self.stats['errors'].append(error_msg)
                        print(f"[FAIL] Failed: {Path(file_path).name} - {e}")
        else:
            # Sequential processing
            for file_path in files:
                file_result = self.process_issue_file(Path(file_path))
                all_results['file_results'].append(file_result)

        # Add summary statistics
        all_results['summary'] = {
            'files_processed': len(files),
            'issues_processed': self.stats['issues_processed'],
            'mrs_processed': self.stats['mrs_processed'],
            'mrs_with_components': self.stats['mrs_with_components'],
            'mrs_with_commits': self.stats['mrs_with_commits'],
            'commits_total': self.stats['commits_total'],
            'commits_with_dep_changes': self.stats['commits_with_dep_changes'],
            'commits_filtered_out': self.stats['commits_filtered_out'],
            'files_total': self.stats['files_total'],
            'files_with_dep_changes': self.stats['files_with_dep_changes'],
            'files_filtered_out': self.stats['files_filtered_out'],
            'unique_components': list(self.stats['unique_components']),
            'rsf_files_used': list(self.stats['rsf_files_used']),
            'error_count': len(self.stats['errors'])
        }

        if self.stats['errors']:
            all_results['errors'] = self.stats['errors']

        return all_results

    def save_results(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save component mapping results to JSON file with algorithm-specific naming.
        Restructures data to a flat list of issues with at least one MR.

        Args:
            results: Mapping results to save
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to saved file
        """

        # Use algorithm-specific filename
        output_dir = self.data_root / "mapping"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"mapping_{self.subject_system}_{self.algorithm}.json"

        # Restructure data: flatten to list of issues with at least 1 MR and components
        flattened_issues = []

        for file_result in results.get('file_results', []):
            batch_file = file_result.get('file_name', 'unknown')

            for issue_result in file_result.get('issue_results', []):
                # Skip issues with errors, no MRs, or no components
                if 'error' in issue_result:
                    continue
                if issue_result.get('total_mrs', 0) == 0:
                    continue
                if not issue_result.get('aggregated_components_by_version', []):
                    continue

                # Add batch file field and include issue
                issue_result['batch_file'] = batch_file
                flattened_issues.append(issue_result)

        # Create final output structure
        output_data = {
            'subject_system': results.get('subject_system'),
            'algorithm': results.get('algorithm'),
            'processing_timestamp': results.get('processing_timestamp'),
            'total_issues': len(flattened_issues),
            'issues': flattened_issues,
            'summary': results.get('summary', {})
        }

        if results.get('errors'):
            output_data['errors'] = results['errors']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")
        print(f"Total issues with MRs: {len(flattened_issues)}")
        return output_path

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the component mapping results."""
        print(f"\n{'='*60}")
        print("COMPONENT MAPPING SUMMARY")
        print(f"{'='*60}")

        summary = results.get('summary', {})

        print(f"Files processed: {summary.get('files_processed', 0)}")
        print(f"Issues processed: {summary.get('issues_processed', 0)}")
        print(f"MRs processed: {summary.get('mrs_processed', 0)}")
        print(f"MRs with components: {summary.get('mrs_with_components', 0)}")
        print(f"MRs with commits: {summary.get('mrs_with_commits', 0)}")

        # Commit filtering statistics
        commits_total = summary.get('commits_total', 0)
        commits_with_dep = summary.get('commits_with_dep_changes', 0)
        commits_filtered = summary.get('commits_filtered_out', 0)

        if commits_total > 0:
            print(f"\nCommit-Level Filtering:")
            print(f"  Total commits: {commits_total}")
            print(f"  Commits with dependency changes: {commits_with_dep}")
            print(f"  Commits filtered out: {commits_filtered}")
            dep_change_rate = (commits_with_dep / commits_total) * 100 if commits_total > 0 else 0
            print(f"  Commit dependency change rate: {dep_change_rate:.1f}%")

        # File filtering statistics
        files_total = summary.get('files_total', 0)
        files_with_dep = summary.get('files_with_dep_changes', 0)
        files_filtered = summary.get('files_filtered_out', 0)

        if files_total > 0:
            print(f"\nFile-Level Filtering:")
            print(f"  Total files (from commits with dep changes): {files_total}")
            print(f"  Files with dependency changes: {files_with_dep}")
            print(f"  Files filtered out: {files_filtered}")
            file_dep_change_rate = (files_with_dep / files_total) * 100 if files_total > 0 else 0
            print(f"  File dependency change rate: {file_dep_change_rate:.1f}%")

        print(f"\nComponents:")
        print(f"  Unique components found: {len(summary.get('unique_components', []))}")
        print(f"  RSF files used: {len(summary.get('rsf_files_used', []))}")

        file_results = results.get('file_results', [])
        if file_results:
            all_stats = compute_version_stats(file_results, dep_only=False)
            dep_stats = compute_version_stats(file_results, dep_only=True)

            def _print_stats(label: str, stats: Dict[str, Dict[str, int]]):
                if not stats:
                    return
                print(f"\n{label}:")
                for version, counts in sorted(stats.items()):
                    print(
                        f"  {version}: issues={counts['issues']} "
                        f"mrs={counts['mrs']} commits={counts['commits']}"
                    )

            _print_stats("By version (all commits)", all_stats)
            _print_stats("By version (dependency-change commits)", dep_stats)

        if summary.get('mrs_processed', 0) > 0:
            success_rate = (summary.get('mrs_with_components', 0) / summary.get('mrs_processed', 1)) * 100
            print(f"\nComponent mapping success rate: {success_rate:.1f}%")

        if summary.get('error_count', 0) > 0:
            print(f"\nErrors encountered: {summary.get('error_count', 0)}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Map merge requests to affected architectural components")
    parser.add_argument('--issue_dir', help='Directory containing issue data files')
    parser.add_argument('--system', help='Subject system name (auto-detected if not provided)')
    parser.add_argument('--pattern', default='*.json', help='File pattern for issue files')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--data-root', help='Root directory for data files')
    parser.add_argument('--algorithm', default='acdc', help='RSF clustering algorithm to use (default: acdc)')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers (default: 8)')

    args = parser.parse_args()

    # Validate issue directory
    issue_dir = Path(args.issue_dir)
    if not issue_dir.exists():
        print(f"Error: Issue directory not found: {issue_dir}")
        return 1

    # Create component mapper
    data_root = Path(args.data_root) if args.data_root else None
    mapper = ComponentMapper(issue_dir, args.system, data_root, args.algorithm)

    # Process all issues
    results = mapper.process_all_issues(args.pattern, max_workers=args.workers)

    if 'error' in results:
        print(f"Processing failed: {results['error']}")
        return 1

    # Save results
    output_path = Path(args.output) if args.output else None
    mapper.save_results(results, output_path)

    # Print summary
    mapper.print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
