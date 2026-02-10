#!/usr/bin/env python3
"""
Merge Request Version Merger

This module adds version information to merge requests in issue data files.
Given an issue data path and subject system name, it enriches the related_mrs
field with version information computed using release dates and gold version logic.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import glob

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.release_utils import (
    get_release_dates_for_system,
    compute_gold_version,
    detect_subject_system
)


class MRVersionMerger:
    """Merges version information into issue merge request data."""

    def __init__(self, issue_data_path: str, subject_system: str = None):
        """
        Initialize the MR Version Merger.

        Args:
            issue_data_path (str): Path to directory containing issue data files
            subject_system (str, optional): Name of the subject system (auto-detected if None)
        """
        self.issue_data_path = Path(issue_data_path)
        self.subject_system = subject_system or detect_subject_system(str(issue_data_path))
        self.release_dates = None
        self.mr_cache = {}  # Cache for MR version info to prevent duplicate processing
        self.stats = {
            'files_processed': 0,
            'issues_processed': 0,
            'mrs_enriched': 0,
            'mrs_cached': 0,
            'errors': []
        }

        print(f"Initialized MR Version Merger for system: {self.subject_system}")
        print(f"Issue data path: {self.issue_data_path}")

    def load_release_dates(self):
        """Load release dates for the subject system."""
        print(f"\nLoading release dates for {self.subject_system}...")

        try:
            self.release_dates = get_release_dates_for_system(self.subject_system)
            print(f"Loaded {len(self.release_dates)} release dates")

            if not self.release_dates:
                print(" Warning: No release dates found. Version computation may be limited.")
                return False

            # Show sample release dates
            sorted_releases = sorted(self.release_dates.items(), key=lambda x: x[1], reverse=True)
            print("Recent releases:")
            for version, date in sorted_releases[:5]:
                print(f"   {version}: {date}")

            return True

        except Exception as e:
            error_msg = f"Error loading release dates: {e}"
            self.stats['errors'].append(error_msg)
            print(f" {error_msg}")
            return False

    def compute_mr_version_info(self, mr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute version information for a merge request using MR merge time.
        Uses cache to avoid duplicate processing of the same MR.

        Args:
            mr_data (dict): Merge request data containing merged_at field

        Returns:
            dict: Version information to add to the MR
        """
        mr_iid = mr_data.get('iid', 'unknown')

        # Create cache key using MR IID and relevant fields for version computation
        merge_date = mr_data.get('merged_at') or mr_data.get('merge_at')
        mr_state = mr_data.get('merge_status', '').lower()
        cache_key = f"{mr_iid}_{merge_date}_{mr_state}"

        # Check cache first
        if cache_key in self.mr_cache:
            self.stats['mrs_cached'] += 1
            return self.mr_cache[cache_key].copy()  # Return copy to avoid mutation issues

        version_info = {}

        try:
            # Compute version based on MR merge time (not issue creation time)
            if merge_date and self.release_dates:
                gold_version = compute_gold_version(merge_date, self.release_dates)
                if gold_version != 'Unknown':
                    version_info['mr_version'] = gold_version
                    version_info['version_computed_from'] = 'mr_merge_date'
                    version_info['merge_date_used'] = merge_date
                else:
                    version_info['version_computed_from'] = 'unknown_merge_date'
            else:
                version_info['version_computed_from'] = 'no_merge_date_available'

            # Determine MR status based on state
            if mr_state == 'merged':
                version_info['mr_status'] = 'fixed'
                # For merged MRs, use the computed version as target version
                if 'mr_version' in version_info:
                    version_info['target_version'] = version_info['mr_version']
            elif mr_state == 'closed':
                version_info['mr_status'] = 'abandoned'
            else:
                version_info['mr_status'] = 'pending'

            # Add relationship context
            relationship = mr_data.get('relationship_type', 'related')
            if relationship == 'closes':
                version_info['fix_type'] = 'complete_fix'
            elif relationship == 'related':
                version_info['fix_type'] = 'partial_fix_or_related'
            else:
                version_info['fix_type'] = 'other'

            # Add computation timestamp
            version_info['version_computation_timestamp'] = datetime.now().isoformat()

            # Cache the result
            self.mr_cache[cache_key] = version_info.copy()

        except Exception as e:
            error_msg = f"Error computing version info for MR {mr_iid}: {e}"
            self.stats['errors'].append(error_msg)
            print(f"  {error_msg}")

        return version_info

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['mrs_enriched'] + self.stats['mrs_cached']
        return {
            'cache_entries': len(self.mr_cache),
            'cache_hits': self.stats['mrs_cached'],
            'total_requests': total_requests,
            'hit_rate': (self.stats['mrs_cached'] / total_requests * 100) if total_requests > 0 else 0
        }

    def process_issue(self, issue_data: Dict[str, Any]) -> bool:
        """
        Process a single issue and enrich its merge request data.

        Args:
            issue_data (dict): Issue data containing related_merge_requests

        Returns:
            bool: True if processing was successful
        """
        try:
            # Extract issue metadata - handle nested structure
            if 'issue_data' in issue_data:
                # GraphViz format: nested under issue_data
                issue_metadata = issue_data['issue_data'].get('issue_metadata', {})
                related_mrs_detailed = issue_data.get('related_mrs', [])
            else:
                # Flat format
                issue_metadata = issue_data.get('issue_metadata', {})
                related_mrs_detailed = issue_data.get('related_mrs', [])

            issue_id = issue_metadata.get('iid', 'unknown')

            # Process both types of MR data
            mrs_enriched_in_issue = 0

            # Process detailed MR data (in related_mrs with mr_metadata)
            for mr in related_mrs_detailed:
                mr_metadata = mr['mr_metadata']
                version_info = self.compute_mr_version_info(mr_metadata)

                if version_info:
                    # Add version info directly to mr_metadata
                    mr_metadata.update(version_info)
                    mrs_enriched_in_issue += 1

            self.stats['mrs_enriched'] += mrs_enriched_in_issue

            return True

        except Exception as e:
            error_msg = f"Error processing issue: {e}"
            self.stats['errors'].append(error_msg)
            print(f"[ERROR] {error_msg}")
            return False

    def process_file(self, file_path: Path) -> bool:
        """
        Process a single issue data file.

        Args:
            file_path (Path): Path to the issue data file

        Returns:
            bool: True if processing was successful
        """
        print(f"\nProcessing file: {file_path.name}")

        try:
            # Load the file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get issues from the data
            issues = data.get('issues', [])
            print(f"  Found {len(issues)} issues")

            # Process each issue (quietly)
            for issue_data in issues:
                self.process_issue(issue_data)
                self.stats['issues_processed'] += 1

            # Save the enriched data back to file
            output_path = file_path.parent / f"enriched_{file_path.name}"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"  Saved enriched data to: {output_path.name}")
            self.stats['files_processed'] += 1

            return True

        except Exception as e:
            error_msg = f"Error processing file {file_path}: {e}"
            self.stats['errors'].append(error_msg)
            print(f" {error_msg}")
            return False

    def process_all_files(self, file_pattern: str = "issues_all_issues_batch_*.json") -> bool:
        """
        Process all issue files in the data directory.

        Args:
            file_pattern (str): Glob pattern for issue files

        Returns:
            bool: True if all files were processed successfully
        """
        print(f"\n{'='*60}")
        print("PROCESSING ALL ISSUE FILES")
        print(f"{'='*60}")

        # Find all matching files
        search_pattern = str(self.issue_data_path / file_pattern)
        files = sorted(glob.glob(search_pattern))

        if not files:
            enriched_pattern = str(self.issue_data_path / f"enriched_{file_pattern}")
            files = sorted(glob.glob(enriched_pattern))

        if not files:
            print(f" No files found matching pattern: {search_pattern}")
            return False

        print(f"Found {len(files)} files to process")

        # Load release dates
        if not self.load_release_dates():
            print(" Failed to load release dates. Aborting.")
            return False

        # Process each file
        successful_files = 0
        for file_path in files:
            if self.process_file(Path(file_path)):
                successful_files += 1

        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {self.stats['files_processed']}/{len(files)}")
        print(f"Issues processed: {self.stats['issues_processed']}")
        print(f"MRs enriched: {self.stats['mrs_enriched']}")
        print(f"MRs from cache: {self.stats['mrs_cached']}")
        print(f"Cache entries: {len(self.mr_cache)}")

        # Calculate cache efficiency
        total_mr_requests = self.stats['mrs_enriched'] + self.stats['mrs_cached']
        if total_mr_requests > 0:
            cache_rate = (self.stats['mrs_cached'] / total_mr_requests) * 100
            print(f"Cache hit rate: {cache_rate:.1f}%")

        if self.stats['errors']:
            print(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more errors")

        return successful_files == len(files)

    def process_single_file(self, filename: str) -> bool:
        """
        Process a single specific file.

        Args:
            filename (str): Name of the file to process

        Returns:
            bool: True if processing was successful
        """
        file_path = self.issue_data_path / filename

        if not file_path.exists():
            print(f" File not found: {file_path}")
            return False

        # Load release dates
        if not self.load_release_dates():
            print(" Failed to load release dates. Aborting.")
            return False

        return self.process_file(file_path)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Add version information to merge requests in issue data")
    parser.add_argument('--dir-path', help='Path to directory containing issue data files')
    parser.add_argument('--system', help='Subject system name (auto-detected if not provided)')
    parser.add_argument('--file', help='Process single file instead of all files')
    parser.add_argument('--pattern', default='issues_all_issues_batch_*.json',
                       help='File pattern for batch processing issues')

    args = parser.parse_args()

    # Validate path
    if not Path(args.dir_path).exists():
        print(f" Error: Path not found: {args.dir_path}")
        return 1

    # Create merger
    merger = MRVersionMerger(args.dir_path, args.system)

    # Process files
    if args.file:
        success = merger.process_single_file(args.file)
    else:
        success = merger.process_all_files(args.pattern)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
