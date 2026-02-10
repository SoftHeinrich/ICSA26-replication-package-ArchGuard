#!/usr/bin/env python3
"""
Generate Clean Classification Data

Uses issue-to-component mapping and smell-to-issue mapping to generate
classification data for smell detection. Loads issue descriptions from
batch issue files. Generates both dirty and clean versions:
- Dirty: every version-specific mapping treated as a separate sample
- Clean: deduplicated by issue, labeled smelly if any version has smells
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime

def load_json_file(file_path: str) -> dict:
    """Load and return JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_version_label(version: Optional[str]) -> Optional[str]:
    """
    Normalize version labels to align smell evolution data with enriched issues.

    - Strips the temporary `temp-` prefix used by Arcan evolution exports.
    """
    if not version:
        return version
    normalized = version.strip()
    if normalized.startswith("temp-"):
        normalized = normalized[len("temp-"):]
    return normalized


def load_dependency_change_mrs(dep_checklist_file: str) -> Set[int]:
    """Load MRs that have dependency changes from checklist file."""
    if not dep_checklist_file or not Path(dep_checklist_file).exists():
        print(f"Warning: Dependency checklist file not found: {dep_checklist_file}")
        return set()

    data = load_json_file(dep_checklist_file)
    dep_mrs = set()

    for commit_info in data.get('commits', {}).values():
        if commit_info.get('has_dependency_changes', False):
            mr_iid = commit_info.get('mr_iid')
            if mr_iid is not None:
                dep_mrs.add(mr_iid)

    print(f"Loaded {len(dep_mrs)} MRs with dependency changes")
    return dep_mrs


def get_issue_mrs(mapping_data: dict, issue_id: int) -> Set[int]:
    """Get all MR IIDs associated with an issue."""
    for issue in mapping_data.get('issues', []):
        if issue.get('issue_id') == issue_id:
            mrs = set()
            for mr in issue.get('mr_results', []):
                mr_iid = mr.get('mr_iid')
                if mr_iid is not None and mr.get('processing_status') == 'success':
                    mrs.add(mr_iid)
            return mrs
    return set()


def extract_smelly_issues_by_version(s2i_data: dict, mapping_data: dict = None,
                                     dep_change_mrs: Set[int] = None) -> Dict[str, Set[int]]:
    """
    Extract issues linked to smells by version.

    If dep_change_mrs is provided, only include issues that have at least
    one MR with dependency changes.

    Supports both old format (smell_to_issue_mappings) and new format (evolution_to_issue_mappings).
    """
    smelly_issues = defaultdict(set)

    def is_change_to_smelly(mapping: dict) -> bool:
        """Return True if a changed mapping represents newly smelly components."""
        cluster_changes = mapping.get("cluster_changes") or {}
        added = cluster_changes.get("added_clusters") or cluster_changes.get("added") or []
        if added:
            return True

        old_clusters = set((mapping.get("old_smell") or {}).get("clusters") or [])
        new_clusters = set((mapping.get("new_smell") or {}).get("clusters") or [])
        return bool(new_clusters - old_clusters)

    # Handle new format: evolution_to_issue_mappings
    if 'evolution_to_issue_mappings' in s2i_data:
        for evolution in s2i_data.get('evolution_to_issue_mappings', []):
            to_version = normalize_version_label(evolution.get('to_version'))
            if not to_version:
                continue

            # Only count smells that become smelly (added or changed-to-smelly).
            for mapping_type in ['added_smell_mappings', 'changed_smell_mappings']:
                for mapping in evolution.get(mapping_type, []):
                    if mapping_type == 'changed_smell_mappings' and not is_change_to_smelly(mapping):
                        continue
                    for issue_match in mapping.get('matching_issues', []):
                        issue_id = issue_match.get('issue_id')
                        if issue_id:
                            # Filter by dependency changes if requested
                            if dep_change_mrs is not None and mapping_data is not None:
                                issue_mrs = get_issue_mrs(mapping_data, issue_id)
                                # Only include if issue has at least one MR with dependency changes
                                if issue_mrs & dep_change_mrs:  # Set intersection
                                    smelly_issues[to_version].add(issue_id)
                            else:
                                smelly_issues[to_version].add(issue_id)

    # Handle old format: smell_to_issue_mappings (for backward compatibility)
    elif 'smell_to_issue_mappings' in s2i_data:
        for version, version_data in s2i_data.get('smell_to_issue_mappings', {}).items():
            normalized_version = normalize_version_label(version)
            if not normalized_version:
                continue
            for mapping in version_data.get('mappings', []):
                if mapping.get('issue_count', 0) > 0:
                    for issue_match in mapping.get('matching_issues', []):
                        issue_id = issue_match.get('issue_id')
                        if issue_id:
                            # Filter by dependency changes if requested
                            if dep_change_mrs is not None and mapping_data is not None:
                                issue_mrs = get_issue_mrs(mapping_data, issue_id)
                                # Only include if issue has at least one MR with dependency changes
                                if issue_mrs & dep_change_mrs:  # Set intersection
                                    smelly_issues[normalized_version].add(issue_id)
                            else:
                                smelly_issues[normalized_version].add(issue_id)

    return smelly_issues


def extract_issue_descriptions(issues_folder: str) -> Dict[int, Dict[str, str]]:
    """Extract issue titles and descriptions from batch issue files."""
    descriptions = {}
    issues_path = Path(issues_folder)

    if not issues_path.exists():
        print(f"Warning: Issues folder not found: {issues_folder}")
        return descriptions

    # Load all JSON files in the folder
    for json_file in sorted(issues_path.glob("*.json")):
        try:
            data = load_json_file(str(json_file))

            for issue in data.get('issues', []):
                issue_metadata = issue.get('issue_data', {}).get('issue_metadata', {})
                issue_id = issue_metadata.get('iid')
                title = issue_metadata.get('title', '')
                description = issue_metadata.get('description', '')

                if issue_id:
                    full_description = f"{title}\n{description}" if description else title
                    descriptions[issue_id] = {
                        'title': title,
                        'description': description,
                        'full_text': full_description
                    }

        except Exception as e:
            print(f"Warning: Error loading {json_file.name}: {e}")
            continue

    print(f"Loaded {len(descriptions)} issue descriptions from {issues_folder}")
    return descriptions


def extract_first_mr_timestamps(enriched_issues_folder: str) -> Dict[int, str]:
    """
    Extract the earliest commit timestamp from the first MR for each issue.

    This is used to filter discussions that occurred after the first MR was proposed,
    preventing temporal leakage in classification data.

    Args:
        enriched_issues_folder: Path to folder containing enriched_issues_*.json files

    Returns:
        Dict mapping issue_id -> earliest MR commit timestamp (ISO 8601 string)
    """
    first_mr_timestamps = {}
    enriched_path = Path(enriched_issues_folder)

    if not enriched_path.exists():
        print(f"Warning: Enriched issues folder not found: {enriched_issues_folder}")
        return first_mr_timestamps

    # Scan all enriched_issues_*.json files
    for json_file in sorted(enriched_path.glob("enriched_issues_*.json")):
        try:
            data = load_json_file(str(json_file))

            for issue in data.get('issues', []):
                issue_id = issue.get('summary', {}).get('issue_id')
                if not issue_id:
                    continue

                related_mrs = issue.get('related_mrs', [])
                if not related_mrs:
                    continue

                # Get first MR (they should be ordered, but we'll be safe)
                first_mr = related_mrs[0]
                commits = first_mr.get('commits', [])

                if not commits:
                    continue

                # Find earliest commit timestamp from first MR
                earliest_timestamp = None
                for commit in commits:
                    commit_meta = commit.get('commit_metadata', {})
                    # Try authored_date first (when developer created the commit)
                    timestamp = commit_meta.get('authored_date') or commit_meta.get('created_at')

                    if timestamp:
                        if earliest_timestamp is None:
                            earliest_timestamp = timestamp
                        else:
                            # Parse and compare timestamps
                            try:
                                current_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                earliest_dt = datetime.fromisoformat(earliest_timestamp.replace('Z', '+00:00'))
                                if current_dt < earliest_dt:
                                    earliest_timestamp = timestamp
                            except ValueError:
                                # If parsing fails, just keep the first one
                                pass

                if earliest_timestamp:
                    first_mr_timestamps[issue_id] = earliest_timestamp

        except Exception as e:
            print(f"Warning: Error loading {json_file.name}: {e}")
            continue

    print(f"Loaded first MR timestamps for {len(first_mr_timestamps)} issues")
    return first_mr_timestamps


def extract_issue_discussions(issues_folder: str, first_mr_timestamps: Optional[Dict[int, str]] = None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Extract issue discussions from batch discussion files.

    Looks for issues_all_discussions_batch_*.json files and combines
    all discussion threads into a single text for each issue.

    If first_mr_timestamps is provided, filters out discussions that occurred
    after the first MR was proposed (to prevent temporal leakage).

    Args:
        issues_folder: Path to folder containing discussion files
        first_mr_timestamps: Optional dict mapping issue_id -> first MR timestamp.
                           If provided, only discussions before this timestamp are included.

    Returns:
        Tuple of (filtered_discussions, unfiltered_discussions)
    """
    discussions: Dict[int, str] = {}
    raw_discussions: Dict[int, str] = {}
    issues_path = Path(issues_folder)

    if not issues_path.exists():
        print(f"Warning: Issues folder not found: {issues_folder}")
        return discussions

    # Load all discussion batch files
    discussion_files = sorted(issues_path.glob("issues_all_discussions_batch_*.json"))

    if not discussion_files:
        print(f"Warning: No discussion files found in {issues_folder}")
        return discussions

    filtered_count = 0
    total_discussions = 0

    for json_file in discussion_files:
        try:
            data = load_json_file(str(json_file))

            for issue_key, issue_discussion in data.get('discussions', {}).items():
                issue_id = int(issue_discussion.get('issue_id'))
                threads = issue_discussion.get('discussion_threads', [])

                # Get MR timestamp cutoff if available
                mr_timestamp = None
                if first_mr_timestamps and issue_id in first_mr_timestamps:
                    try:
                        mr_timestamp = datetime.fromisoformat(
                            first_mr_timestamps[issue_id].replace('Z', '+00:00')
                        )
                    except (ValueError, AttributeError):
                        mr_timestamp = None

                # Combine all discussion texts (with optional filtering)
                discussion_texts = []
                raw_discussion_texts = []
                for thread in threads:
                    total_discussions += 1
                    text = thread.get('text', '')
                    author = thread.get('author_username', 'unknown')

                    if not text:
                        continue

                    raw_discussion_texts.append(f"[{author}]: {text}")

                    # Filter by MR timestamp if available
                    if mr_timestamp:
                        discussion_created = thread.get('created_at')
                        if discussion_created:
                            try:
                                discussion_dt = datetime.fromisoformat(
                                    discussion_created.replace('Z', '+00:00')
                                )
                                # Skip discussions created after first MR
                                if discussion_dt >= mr_timestamp:
                                    filtered_count += 1
                                    continue
                            except (ValueError, AttributeError):
                                # If timestamp parsing fails, include the discussion
                                pass

                    discussion_texts.append(f"[{author}]: {text}")

                # Join all discussion threads
                if discussion_texts:
                    combined_discussion = "\n\n".join(discussion_texts)
                    discussions[issue_id] = combined_discussion
                if raw_discussion_texts:
                    raw_discussions[issue_id] = "\n\n".join(raw_discussion_texts)

        except Exception as e:
            print(f"Warning: Error loading {json_file.name}: {e}")
            continue

    print(f"Loaded {len(discussions)} filtered issue discussions from {issues_folder}")
    if first_mr_timestamps:
        print(f"  Filtered out {filtered_count} discussions that occurred after first MR (out of {total_discussions} total)")
    print(f"Loaded {len(raw_discussions)} unfiltered issue discussions from {issues_folder}")
    return discussions, raw_discussions


def extract_issues_by_version_from_mapping(enriched_issues_folder: str) -> Dict[str, Set[int]]:
    """
    Extract all issues by version from enriched issues files.

    Scans all enriched issues files and includes only issues that have
    at least one related MR. Groups issues by MR version.

    Args:
        enriched_issues_folder: Path to folder containing enriched_issues_*.json files

    Returns:
        Dict mapping version -> set of issue IDs that have related MRs in that version
    """
    issues_by_version = defaultdict(set)
    enriched_path = Path(enriched_issues_folder)

    if not enriched_path.exists():
        print(f"Warning: Enriched issues folder not found: {enriched_issues_folder}")
        return issues_by_version

    # Scan all enriched_issues_*.json files
    for json_file in sorted(enriched_path.glob("enriched_issues_*.json")):
        data = load_json_file(str(json_file))

        for issue in data.get('issues', []):
            issue_id = issue.get('summary', {}).get('issue_id')
            if not issue_id:
                continue

            related_mrs = issue.get('related_mrs', [])
            # Only include issues with at least one related MR
            if len(related_mrs) > 0:
                # Add issue to all versions it appears in via related MRs
                for mr in related_mrs:
                    raw_version = mr.get('mr_metadata', {}).get('mr_version')
                    mr_version = normalize_version_label(raw_version)
                    if mr_version:
                        issues_by_version[mr_version].add(issue_id)


    print(f"Loaded {sum(len(issues) for issues in issues_by_version.values())} issue-version pairs from enriched issues")
    print(f"Unique issues with related MRs: {len(set().union(*issues_by_version.values()) if issues_by_version else set())}")

    return issues_by_version


def generate_dirty_classification_data(s2i_file: str, mapping_file: str,
                                      issues_folder: str,
                                      dep_checklist_file: str = None) -> List[Dict]:
    """Generate dirty classification data (every mapping as smelly sample)."""
    s2i_data = load_json_file(s2i_file)
    mapping_data = load_json_file(mapping_file)
    descriptions = extract_issue_descriptions(issues_folder)

    # Extract first MR timestamps to filter discussions (prevent temporal leakage)
    first_mr_timestamps = extract_first_mr_timestamps(issues_folder)
    discussions_filtered, discussions_unfiltered = extract_issue_discussions(issues_folder, first_mr_timestamps)

    # Load dependency change MRs if provided
    dep_change_mrs = None
    if dep_checklist_file:
        dep_change_mrs = load_dependency_change_mrs(dep_checklist_file)

    data = []
    smelly_issues = extract_smelly_issues_by_version(s2i_data, mapping_data, dep_change_mrs)
    all_issues = extract_issues_by_version_from_mapping(issues_folder)

    for version in sorted(all_issues.keys()):
        version_smelly = smelly_issues.get(version, set())
        version_issues = all_issues[version]

        for issue_id in version_issues:
            if issue_id in descriptions:
                label = "smell" if issue_id in version_smelly else "no_smell"
                issue_data = descriptions[issue_id]

                data.append({
                    'id': issue_id,
                    'title': issue_data['title'],
                    'description': issue_data['full_text'],
                    'discussion': discussions_filtered.get(issue_id, ''),
                    'discussion_unfiltered': discussions_unfiltered.get(issue_id, ''),
                    'label': label,
                    'version': version
                })

    return data


def generate_clean_classification_data(s2i_file: str, mapping_file: str,
                                       issues_folder: str,
                                       dep_checklist_file: str = None) -> List[Dict]:
    """Generate clean classification data (issue is smelly if any record is smelly)."""
    s2i_data = load_json_file(s2i_file)
    mapping_data = load_json_file(mapping_file)
    descriptions = extract_issue_descriptions(issues_folder)

    # Extract first MR timestamps to filter discussions (prevent temporal leakage)
    first_mr_timestamps = extract_first_mr_timestamps(issues_folder)
    discussions_filtered, discussions_unfiltered = extract_issue_discussions(issues_folder, first_mr_timestamps)

    # Load dependency change MRs if provided
    dep_change_mrs = None
    if dep_checklist_file:
        dep_change_mrs = load_dependency_change_mrs(dep_checklist_file)

    # Get all smelly issues across all versions
    smelly_by_version = extract_smelly_issues_by_version(s2i_data, mapping_data, dep_change_mrs)
    all_issues_by_version = extract_issues_by_version_from_mapping(issues_folder)

    # Invert the mapping: issue_id -> set of versions
    issue_versions = defaultdict(set)
    for version, issue_ids in all_issues_by_version.items():
        for issue_id in issue_ids:
            issue_versions[issue_id].add(version)

    all_smelly_issues = set()
    for version, issues in smelly_by_version.items():
        for issue_id in issues:
            if issue_id in issue_versions and version in issue_versions[issue_id]:
                all_smelly_issues.add(issue_id)

    # Generate deduplicated data
    data = []
    processed_issues = set()

    for issue_id, versions in issue_versions.items():
        if issue_id in processed_issues or issue_id not in descriptions:
            continue

        label = "smell" if issue_id in all_smelly_issues else "no_smell"
        issue_data = descriptions[issue_id]

        # Use all versions this issue appears in
        version_str = ','.join(sorted(versions))

        data.append({
            'id': issue_id,
            'title': issue_data['title'],
            'description': issue_data['full_text'],
                'discussion': discussions_filtered.get(issue_id, ''),
                'discussion_unfiltered': discussions_unfiltered.get(issue_id, ''),
            'label': label,
            'version': version_str
        })

        processed_issues.add(issue_id)

    return data


def save_classification_data(data: List[Dict], output_file: str):
    """Save classification data to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Count smell and no_smell issues
    smell_count = sum(1 for record in data if record['label'] == 'smell')
    no_smell_count = sum(1 for record in data if record['label'] == 'no_smell')

    print(f"Classification data saved to {output_file}")
    print(f"  - smell: {smell_count}")
    print(f"  - no_smell: {no_smell_count}")


def print_summary(data: List[Dict], title: str):
    """Print summary of classification data."""
    if not data:
        print(f"{title}: No data")
        return

    label_counts = defaultdict(int)
    version_stats = defaultdict(lambda: {'smell': 0, 'no_smell': 0, 'total': 0})

    for record in data:
        label = record['label']
        version = record['version']

        label_counts[label] += 1
        version_stats[version][label] += 1
        version_stats[version]['total'] += 1

    print(f"\n=== {title} ===")
    print(f"Total records: {len(data)}")
    print(f"With smell: {label_counts['smell']} ({label_counts['smell']/len(data)*100:.1f}%)")
    print(f"Without smell: {label_counts['no_smell']} ({label_counts['no_smell']/len(data)*100:.1f}%)")
    print(f"Unique versions: {len(version_stats)}")

    print(f"\nBreakdown by version:")
    for version in sorted(version_stats.keys()):
        stats = version_stats[version]
        smell_pct = (stats['smell'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {version}: {stats['total']} total, "
              f"{stats['smell']} smell ({smell_pct:.1f}%), "
              f"{stats['no_smell']} no_smell")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate clean classification data")
    parser.add_argument('--s2i', required=True,
                       help='Path to smell-to-issue mapping file')
    parser.add_argument('--mapping', required=True,
                       help='Path to issue-to-component mapping file')
    parser.add_argument('--issues', required=True,
                       help='Path to issues folder (e.g., data-smell/issues/inkscape_with_discussion)')
    parser.add_argument('--dep-checklist',
                       help='Path to dependency change checklist (from batch_dependency_analyzer.py)')
    parser.add_argument('--output-dirty', default='data/issues/inkscape/classification_data_dirty.json',
                       help='Output file for dirty data')
    parser.add_argument('--output-clean', default='data/issues/inkscape/classification_data_clean.json',
                       help='Output file for clean data')

    args = parser.parse_args()

    # Generate dirty classification data
    print("Generating dirty classification data...")
    if args.dep_checklist:
        print(f"Filtering smelly issues by dependency changes from: {args.dep_checklist}")
    dirty_data = generate_dirty_classification_data(
        args.s2i, args.mapping, args.issues, args.dep_checklist
    )
    save_classification_data(dirty_data, args.output_dirty)
    print_summary(dirty_data, "Dirty Classification Data")

    # Generate clean classification data
    print("\nGenerating clean classification data...")
    clean_data = generate_clean_classification_data(
        args.s2i, args.mapping, args.issues, args.dep_checklist
    )
    save_classification_data(clean_data, args.output_clean)
    print_summary(clean_data, "Clean Classification Data")

    # Show examples
    print(f"\n=== Example Records (Dirty) ===")
    for i, record in enumerate(dirty_data[:3]):
        print(f"{i+1}. ID: {record['id']}, Label: {record['label']}, Version: {record['version']}")
        print(f"   Title: {record['title'][:80]}...")
        print(f"   Description: {record['description'][:100]}...")

    print(f"\n=== Example Records (Clean) ===")
    for i, record in enumerate(clean_data[:3]):
        print(f"{i+1}. ID: {record['id']}, Label: {record['label']}, Version: {record['version']}")
        print(f"   Title: {record['title'][:80]}...")
        print(f"   Description: {record['description'][:100]}...")


if __name__ == "__main__":
    main()
