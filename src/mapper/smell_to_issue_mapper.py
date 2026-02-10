#!/usr/bin/env python3
"""
Smell Evolution to Issue Mapper

Analyzes smell evolution and maps issues to changed/removed smells.
Follows similar logic to evolution_decision_combiner and analyze_smell_evolution.
Can reuse existing evolution output or generate new analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.issues.config import ConfigManager, Algorithm
from src.issues.utils import Logger, PathResolver
from src.issues.base import SmellProcessorBase, ProcessingResult, ConfigurableCLIBase, DataMatcher


class SmellEvolutionToIssueMapper(SmellProcessorBase):
    """Maps issues to changed and removed smells from evolution analysis."""

    def __init__(self, config_manager: ConfigManager = None):
        super().__init__(config_manager)
        self.path_resolver = PathResolver()
        self.data_matcher = DataMatcher()

    def find_evolution_file(self, system_name: str, algorithm: str) -> Optional[Path]:
        """Find evolution file for system and algorithm."""
        evolution_dir = self.config.paths.smells_dir / system_name / "evolution"
        pattern = f"{system_name}_{algorithm}_evolution.json"
        evolution_file = evolution_dir / pattern

        if evolution_file.exists():
            self.logger.info(f"Found evolution file: {evolution_file}")
            return evolution_file

        self.logger.warning(f"No evolution file found: {evolution_file}")
        return None

    def load_evolution_data(self, evolution_file: Path) -> Dict[str, Any]:
        """Load evolution data from file."""
        with open(evolution_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_issue_mapping_file(self, system_name: str, algorithm: str) -> Optional[Path]:
        """Find issue mapping file for system and algorithm."""
        mapping_dir = self.config.paths.project_root / "data" / "mapping" / system_name
        pattern = f"mapping_{system_name}_{algorithm.lower()}.json"
        mapping_file = mapping_dir / pattern

        if mapping_file.exists():
            self.logger.info(f"Found mapping file: {mapping_file}")
            return mapping_file

        self.logger.warning(f"No mapping file found: {mapping_file}")
        return None

    def load_issue_mapping(self, mapping_file: Path) -> Dict[str, Any]:
        """Load issue mapping data."""
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_matching_issues_for_smell(self, smell_clusters: List[str], version: str,
                                      issue_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find issues that match the smell's clusters for the given version."""
        matching_issues = []

        if not smell_clusters:
            return matching_issues

        # Normalize cluster names
        normalized_clusters = set()
        for cluster in smell_clusters:
            variants = self.path_resolver.normalize_path_separators(cluster)
            normalized_clusters.update(variants)

        # Check each issue
        for issue in issue_mapping.get('issues', []):
            issue_id = issue.get('issue_id')

            # Check each MR in the issue
            for mr in issue.get('mr_results', []):
                mr_version = mr.get('mr_version')
                if mr_version != version:
                    continue

                # Check component overlap
                affected_components = set(mr.get('affected_components', []))
                overlap = normalized_clusters.intersection(affected_components)

                if overlap:
                    matching_issues.append({
                        'issue_id': issue_id,
                        'mr_iid': mr.get('mr_iid'),
                        'mr_version': mr_version,
                        'overlapping_components': list(overlap),
                        'overlap_count': len(overlap),
                        'total_smell_clusters': len(smell_clusters),
                        'overlap_ratio': len(overlap) / len(smell_clusters) if smell_clusters else 0
                    })

        return matching_issues

    def process_added_smells(self, evolution_step: Dict[str, Any],
                            issue_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process added smells and find matching issues."""
        results = []
        to_version = evolution_step.get('to_version')

        for added_smell_info in evolution_step.get('added_smells', []):
            new_smell = added_smell_info.get('new_smell', {})
            smell_clusters = new_smell.get('clusters', [])

            # Find issues in the version where smell was added that touched these clusters
            matching_issues = self.find_matching_issues_for_smell(
                smell_clusters, to_version, issue_mapping
            )

            results.append({
                'evolution_type': 'added',
                'from_version': evolution_step.get('from_version'),
                'to_version': to_version,
                'smell': new_smell,
                'reason': added_smell_info.get('reason'),
                'cluster_history': added_smell_info.get('cluster_history'),
                'matching_issues': matching_issues,
                'issue_count': len(matching_issues)
            })

        return results

    def process_removed_smells(self, evolution_step: Dict[str, Any],
                              issue_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process removed smells and find matching issues."""
        results = []
        to_version = evolution_step.get('to_version')

        for removed_smell_info in evolution_step.get('removed_smells', []):
            old_smell = removed_smell_info.get('old_smell', {})
            smell_clusters = old_smell.get('clusters', [])

            # Find issues in the removal version that touched these clusters
            matching_issues = self.find_matching_issues_for_smell(
                smell_clusters, to_version, issue_mapping
            )

            results.append({
                'evolution_type': 'removed',
                'from_version': evolution_step.get('from_version'),
                'to_version': to_version,
                'smell': old_smell,
                'reason': removed_smell_info.get('reason'),
                'matching_issues': matching_issues,
                'issue_count': len(matching_issues)
            })

        return results

    def process_changed_smells(self, evolution_step: Dict[str, Any],
                              issue_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process changed smells and find matching issues."""
        results = []
        to_version = evolution_step.get('to_version')

        for changed_smell_info in evolution_step.get('changed_smells', []):
            new_smell = changed_smell_info.get('new_smell', {})
            old_smell = changed_smell_info.get('old_smell', {})
            cluster_changes = changed_smell_info.get('cluster_changes', {})

            # Find issues based on changed clusters
            added_clusters = cluster_changes.get('added_clusters', [])
            removed_clusters = cluster_changes.get('removed_clusters', [])
            all_changed_clusters = list(set(added_clusters + removed_clusters))

            # Find issues in the target version that touched these clusters
            matching_issues = self.find_matching_issues_for_smell(
                all_changed_clusters, to_version, issue_mapping
            )

            results.append({
                'evolution_type': 'changed',
                'from_version': evolution_step.get('from_version'),
                'to_version': to_version,
                'old_smell': old_smell,
                'new_smell': new_smell,
                'cluster_changes': cluster_changes,
                'matching_issues': matching_issues,
                'issue_count': len(matching_issues)
            })

        return results

    def map_evolution_to_issues(self, system_name: str, algorithm: str,
                               evolution_file: Optional[Path] = None) -> ProcessingResult:
        """Main function to map smell evolution to issues."""
        description = f"smell evolution to issue mapping for {system_name} ({algorithm})"
        self._start_processing(description)

        # Find or use provided evolution file
        if evolution_file is None:
            evolution_file = self.find_evolution_file(system_name, algorithm)
            if not evolution_file:
                return ProcessingResult(
                    success=False,
                    message=f"No evolution file found for {system_name} ({algorithm})"
                )

        # Load evolution data
        evolution_data = self.load_evolution_data(evolution_file)

        # Find and load issue mapping
        mapping_file = self.find_issue_mapping_file(system_name, algorithm)
        if not mapping_file:
            raise Exception()

        issue_mapping = self.load_issue_mapping(mapping_file)

        results = {
            'system': system_name,
            'algorithm': algorithm,
            'evolution_file': str(evolution_file.name),
            'mapping_file': str(mapping_file.name),
            'evolution_to_issue_mappings': [],
            'summary': {
                'total_evolution_steps': 0,
                'added_smells_with_issues': 0,
                'removed_smells_with_issues': 0,
                'changed_smells_with_issues': 0,
                'total_added_smells': 0,
                'total_removed_smells': 0,
                'total_changed_smells': 0,
                'total_issue_mappings': 0
            }
        }

        # Process each evolution step
        for evolution_step in evolution_data.get('evolution_steps', []):
            from_version = evolution_step.get('from_version')
            to_version = evolution_step.get('to_version')

            self.logger.info(f"Processing evolution step: {from_version} → {to_version}")

            # Process added smells
            added_mappings = self.process_added_smells(evolution_step, issue_mapping)

            # Process removed smells
            removed_mappings = self.process_removed_smells(evolution_step, issue_mapping)

            # Process changed smells
            changed_mappings = self.process_changed_smells(evolution_step, issue_mapping)

            step_result = {
                'from_version': from_version,
                'to_version': to_version,
                'added_smell_mappings': added_mappings,
                'removed_smell_mappings': removed_mappings,
                'changed_smell_mappings': changed_mappings,
                'stats': {
                    'added_smells': len(added_mappings),
                    'added_smells_with_issues': sum(1 for m in added_mappings if m['issue_count'] > 0),
                    'removed_smells': len(removed_mappings),
                    'removed_smells_with_issues': sum(1 for m in removed_mappings if m['issue_count'] > 0),
                    'changed_smells': len(changed_mappings),
                    'changed_smells_with_issues': sum(1 for m in changed_mappings if m['issue_count'] > 0)
                }
            }

            results['evolution_to_issue_mappings'].append(step_result)

            # Update summary
            results['summary']['total_evolution_steps'] += 1
            results['summary']['total_added_smells'] += len(added_mappings)
            results['summary']['total_removed_smells'] += len(removed_mappings)
            results['summary']['total_changed_smells'] += len(changed_mappings)
            results['summary']['added_smells_with_issues'] += step_result['stats']['added_smells_with_issues']
            results['summary']['removed_smells_with_issues'] += step_result['stats']['removed_smells_with_issues']
            results['summary']['changed_smells_with_issues'] += step_result['stats']['changed_smells_with_issues']
            results['summary']['total_issue_mappings'] += sum(m['issue_count'] for m in added_mappings)
            results['summary']['total_issue_mappings'] += sum(m['issue_count'] for m in removed_mappings)
            results['summary']['total_issue_mappings'] += sum(m['issue_count'] for m in changed_mappings)

        processing_time = self._finish_processing(description)
        results['processing_time'] = processing_time

        return ProcessingResult(
            success=len(results['evolution_to_issue_mappings']) > 0,
            message=f"Mapped {results['summary']['total_issue_mappings']} issues to smell evolution",
            data=results,
            processing_time=processing_time
        )

    def process(self, system_name: str, algorithm: str = None, **kwargs) -> ProcessingResult:
        """Implementation of abstract process method."""
        evolution_file = kwargs.get('evolution_file')
        return self.map_evolution_to_issues(system_name, algorithm, evolution_file)

    def save_results(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save mapping results to JSON file."""
        if output_path is None:
            system_name = results.get('system', 'unknown')
            algorithm = results.get('algorithm', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.paths.project_root / "data" / "mapping" / system_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"evolution_to_issue_{system_name}_{algorithm}_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {output_path}")
        return output_path

    def print_summary(self, result: ProcessingResult) -> None:
        """Print summary of mapping results."""
        if not result.success or not result.data:
            print("No results to summarize")
            return

        data = result.data
        summary = data.get('summary', {})

        print(f"\nSmell Evolution to Issue Mapping Summary")
        print(f"System: {data.get('system')}")
        print(f"Algorithm: {data.get('algorithm')}")
        print(f"Evolution file: {data.get('evolution_file')}")
        print(f"Total evolution steps: {summary.get('total_evolution_steps')}")
        print(f"\nAdded smells:")
        print(f"  Total: {summary.get('total_added_smells')}")
        print(f"  With issues: {summary.get('added_smells_with_issues')}")
        print(f"\nRemoved smells:")
        print(f"  Total: {summary.get('total_removed_smells')}")
        print(f"  With issues: {summary.get('removed_smells_with_issues')}")
        print(f"\nChanged smells:")
        print(f"  Total: {summary.get('total_changed_smells')}")
        print(f"  With issues: {summary.get('changed_smells_with_issues')}")
        print(f"\nTotal issue mappings: {summary.get('total_issue_mappings')}")



class SmellEvolutionToIssueMapperCLI(ConfigurableCLIBase):
    """Command line interface for Smell Evolution to Issue Mapper."""

    def setup_arguments(self, parser) -> None:
        """Setup command line arguments."""
        parser.add_argument('--system_name', required=True,
                          help='System name (e.g., inkscape)')
        parser.add_argument('--algorithm', required=True,
                          choices=[a.value for a in Algorithm],
                          help='Clustering algorithm')
        parser.add_argument('--evolution-file', type=Path,
                          help='Path to evolution file (optional, will auto-detect if not provided)')
        parser.add_argument('--output', type=Path,
                          help='Output file path for results')

    def run(self, args) -> int:
        """Main execution method."""
        mapper = SmellEvolutionToIssueMapper(self.config)

        try:
            result = mapper.map_evolution_to_issues(
                args.system_name,
                args.algorithm,
                args.evolution_file
            )

            if not result.success:
                print(f"Failed to process {args.system_name}: {result.message}")
                return 1

            output_path = mapper.save_results(result.data, args.output)

            if not args.quiet:
                mapper.print_summary(result)
            print("Smell evolution to issue mapping completed!")
            print(f"Results saved to: {output_path}")

            return 0

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1


def main() -> int:
    """Main entry point."""
    cli = SmellEvolutionToIssueMapperCLI()
    return cli.execute()


if __name__ == "__main__":
    sys.exit(main())
