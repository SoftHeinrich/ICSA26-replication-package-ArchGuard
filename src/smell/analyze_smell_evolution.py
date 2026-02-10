import json
import glob
import os
import sys
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import re

from src.preprocessing.constants import SMELLS_ROOT, CLUSTERS_ROOT, time_print


class ClusterAligner:
    """Aligns clusters across versions based on content similarity."""
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
    
    def load_cluster_contents(self, clusters_path: str) -> Dict[str, Set[str]]:
        """Load cluster contents from RSF file."""
        cluster_contents = defaultdict(set)
        
        if not os.path.exists(clusters_path):
            return {}
            
        with open(clusters_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('contain '):
                    parts = line.split()
                    if len(parts) >= 3:
                        cluster_name = parts[1]
                        file_path = parts[2]
                        cluster_contents[cluster_name].add(file_path)
        
        return dict(cluster_contents)
    
    def calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0


class SmellEvolutionAnalyzer:
    """Analyzes architectural smell evolution across software versions."""
    
    def __init__(self, system_name: str, algorithm: str = "acdc"):
        self.system_name = system_name
        self.algorithm = algorithm
        self.aligner = ClusterAligner()
    
    def get_version_list(self) -> List[str]:
        """Get sorted list of versions for the system."""
        smells_pattern = f"{SMELLS_ROOT}/{self.system_name}/{self.algorithm}/*_smells.json"
        smell_files = glob.glob(smells_pattern)
        
        versions = []
        for file_path in smell_files:
            filename = os.path.basename(file_path)
            version = filename.replace(f"{self.system_name}-", "").replace("_smells.json", "")
            versions.append(version)
        
        # Sort versions naturally
        def version_key(v):
            parts = re.findall(r'\d+|\D+', v)
            return [int(x) if x.isdigit() else x for x in parts]
        
        return sorted(versions, key=version_key)
    
    def load_smells(self, version: str) -> List[Dict]:
        """Load smell data for a specific version as a list of complete smells."""
        smell_file = f"{SMELLS_ROOT}/{self.system_name}/{self.algorithm}/{self.system_name}-{version}_smells.json"
        
        if not os.path.exists(smell_file):
            return []
        
        with open(smell_file, 'r') as f:
            data = json.load(f)
        
        # Return list of smells with their complete information
        smells = []
        if 'smellcollection' in data:
            for i, smell in enumerate(data['smellcollection']):
                smells.append({
                    'id': f"{version}_smell_{i}",
                    'type': smell.get('type', 'unknown'),
                    'clusters': smell.get('clusters', []),
                    'topicNum': smell.get('topicNum', -1),
                    'version': version,
                    'cluster_count': len(smell.get('clusters', []))
                })
        
        return smells
    
    def load_cluster_contents(self, version: str) -> Dict[str, Set[str]]:
        """Load cluster contents for a specific version."""
        clusters_pattern = f"{CLUSTERS_ROOT}/{self.system_name}/{self.algorithm}/{version}/{self.system_name}-{version}_*.rsf"
        cluster_files = glob.glob(clusters_pattern)
        
        if not cluster_files:
            return {}
        
        return self.aligner.load_cluster_contents(cluster_files[0])
    
    def calculate_smell_similarity(self, smell1: Dict, smell2: Dict) -> float:
        """Calculate similarity between two smells based on their cluster content."""
        clusters1 = set(smell1.get('../../data/clusters', []))
        clusters2 = set(smell2.get('../../data/clusters', []))
        return self.aligner.calculate_jaccard_similarity(clusters1, clusters2)
    
    def create_smell_signature(self, smell: Dict) -> str:
        """Create a signature for a smell based on its clusters and type."""
        clusters = sorted(smell.get('../../data/clusters', []))
        smell_type = smell.get('type', 'unknown')
        # Create signature from sorted cluster list and type
        cluster_str = '|'.join(clusters)
        return f"{smell_type}::{cluster_str}"
    
    def align_smells_between_versions(self, smells_v1: List[Dict], smells_v2: List[Dict]) -> Tuple[Dict, Set[int]]:
        """Align smells between two versions based on cluster content similarity."""
        alignment = {}
        used_v2_smells = set()
        
        for smell1 in smells_v1:
            best_match = None
            best_similarity = 0.0
            
            for i, smell2 in enumerate(smells_v2):
                if i in used_v2_smells:
                    continue
                    
                # Only align same type smells
                if smell1['type'] != smell2['type']:
                    continue
                    
                similarity = self.calculate_smell_similarity(smell1, smell2)
                if similarity > best_similarity and similarity >= self.aligner.similarity_threshold:
                    best_similarity = similarity
                    best_match = i
            
            if best_match is not None:
                alignment[smell1['id']] = {
                    'old_smell': smell1,
                    'new_smell': smells_v2[best_match],
                    'similarity': best_similarity
                }
                used_v2_smells.add(best_match)
            else:
                alignment[smell1['id']] = {
                    'old_smell': smell1,
                    'new_smell': None,
                    'similarity': 0.0
                }
        
        return alignment, used_v2_smells
    
    def analyze_cluster_history(self, smell_clusters: List[str], v1: str) -> Dict:
        """Analyze the history of clusters to understand if they existed before but weren't smelly."""
        # Load cluster contents from v1 to check if clusters existed
        clusters_v1 = self.load_cluster_contents(v1)
        
        # Load smells from v1 to check if clusters were smelly
        smells_v1 = self.load_smells(v1)
        smelly_clusters_v1 = set()
        for smell in smells_v1:
            smelly_clusters_v1.update(smell['clusters'])
        
        existed_but_not_smelly = []
        did_not_exist = []
        was_smelly = []
        
        for cluster in smell_clusters:
            if cluster in smelly_clusters_v1:
                was_smelly.append(cluster)
            elif cluster in clusters_v1:
                existed_but_not_smelly.append(cluster)
            else:
                did_not_exist.append(cluster)
        
        return {
            'existed_but_not_smelly': existed_but_not_smelly,
            'did_not_exist': did_not_exist,
            'was_previously_smelly': was_smelly
        }
    
    def generate_detailed_reason(self, smell: Dict, history: Dict) -> str:
        """Generate detailed reason for smell addition based on cluster history."""
        total_clusters = len(smell['clusters'])
        existed_count = len(history['existed_but_not_smelly'])
        new_count = len(history['did_not_exist'])
        was_smelly_count = len(history['was_previously_smelly'])
        
        reasons = []
        
        if new_count > 0:
            reasons.append(f"{new_count} completely new cluster(s)")
        
        if existed_count > 0:
            reasons.append(f"{existed_count} cluster(s) existed before but were not smelly")
        
        if len(reasons) == 0:
            # All clusters were previously smelly in different independent smells
            if was_smelly_count > 0:
                reason_str = f"{was_smelly_count} cluster(s) were previously in different independent smells"
            else:
                reason_str = "new smell with unknown cluster history"
        else:
            reason_str = ", ".join(reasons)
            # Add note about previously smelly clusters if they exist (for completeness)
            if was_smelly_count > 0:
                reason_str += f" (plus {was_smelly_count} cluster(s) from previous independent smells)"
        
        if total_clusters == 1:
            return f"new {smell['type'].upper()} smell: {reason_str}"
        else:
            return f"new {smell['type'].upper()} smell with {total_clusters} clusters: {reason_str}"
    
    def analyze_type_evolution(self, type_smells_v1: List[Dict], type_smells_v2: List[Dict], v1: str, v2: str, smell_type: str) -> Dict:
        """Analyze evolution for a specific smell type independently."""
        # Align smells within this type only
        alignment, used_v2_smells = self.align_smells_between_versions(type_smells_v1, type_smells_v2)
        
        # Analyze smell changes within this type
        added_smells = []
        removed_smells = []
        remained_smells = []
        changed_smells = []
        
        # Process aligned smells
        for smell_id, align_info in alignment.items():
            old_smell = align_info['old_smell']
            new_smell = align_info['new_smell']
            similarity = align_info['similarity']
            
            if new_smell is None:
                # Smell disappeared
                removed_smells.append({
                    'old_smell': old_smell,
                    'reason': f'{old_smell["type"].upper()} smell disappeared (no similar smell found in {v2})'
                })
            else:
                # Compare smell signatures
                old_signature = self.create_smell_signature(old_smell)
                new_signature = self.create_smell_signature(new_smell)
                
                if old_signature == new_signature:
                    # Smell remained exactly the same
                    remained_smells.append({
                        'old_smell': old_smell,
                        'new_smell': new_smell,
                        'similarity': similarity
                    })
                else:
                    # Smell changed (different clusters but same type)
                    old_clusters = set(old_smell['clusters'])
                    new_clusters = set(new_smell['clusters'])
                    
                    changed_smells.append({
                        'old_smell': old_smell,
                        'new_smell': new_smell,
                        'similarity': similarity,
                        'cluster_changes': {
                            'added_clusters': sorted(list(new_clusters - old_clusters)),
                            'removed_clusters': sorted(list(old_clusters - new_clusters)),
                            'common_clusters': sorted(list(old_clusters & new_clusters))
                        }
                    })
        
        # Find new smells within this type
        for i, smell in enumerate(type_smells_v2):
            if i not in used_v2_smells:
                # Analyze cluster history for detailed reason (only consider clusters from same type)
                history = self.analyze_cluster_history_for_type(smell['clusters'], v1, smell_type)
                
                # Only consider it a "new smell" if it contains clusters that are genuinely new to being smelly within this type
                genuinely_new_cluster_count = len(history['existed_but_not_smelly']) + len(history['did_not_exist'])
                
                if genuinely_new_cluster_count > 0:
                    # This smell contains at least some clusters that are genuinely new to being smelly within this type
                    detailed_reason = self.generate_detailed_reason(smell, history)
                    
                    added_smells.append({
                        'new_smell': smell,
                        'reason': detailed_reason,
                        'cluster_history': history
                    })
        
        return {
            'added_smells': added_smells,
            'removed_smells': removed_smells,
            'remained_smells': remained_smells,
            'changed_smells': changed_smells
        }
    
    def analyze_cluster_history_for_type(self, smell_clusters: List[str], v1: str, smell_type: str) -> Dict:
        """Analyze the history of clusters considering only the same smell type."""
        # Load cluster contents from v1 to check if clusters existed
        clusters_v1 = self.load_cluster_contents(v1)
        
        # Load smells from v1 and filter by the same type
        smells_v1 = self.load_smells(v1)
        smelly_clusters_v1_same_type = set()
        for smell in smells_v1:
            if smell['type'] == smell_type:  # Only consider same type
                smelly_clusters_v1_same_type.update(smell['clusters'])
        
        existed_but_not_smelly = []
        did_not_exist = []
        was_smelly = []
        
        for cluster in smell_clusters:
            if cluster in smelly_clusters_v1_same_type:
                was_smelly.append(cluster)
            elif cluster in clusters_v1:
                existed_but_not_smelly.append(cluster)
            else:
                did_not_exist.append(cluster)
        
        return {
            'existed_but_not_smelly': existed_but_not_smelly,
            'did_not_exist': did_not_exist,
            'was_previously_smelly': was_smelly
        }
    
    def analyze_evolution_between_versions(self, v1: str, v2: str) -> Dict:
        """Analyze smell evolution between two consecutive versions."""
        time_print(f"Analyzing evolution from {v1} to {v2}")
        
        # Load smell data
        smells_v1 = self.load_smells(v1)
        smells_v2 = self.load_smells(v2)
        
        # Group smells by type for independent analysis
        smells_v1_by_type = {}
        smells_v2_by_type = {}
        
        for smell in smells_v1:
            smell_type = smell['type']
            if smell_type not in smells_v1_by_type:
                smells_v1_by_type[smell_type] = []
            smells_v1_by_type[smell_type].append(smell)
            
        for smell in smells_v2:
            smell_type = smell['type']
            if smell_type not in smells_v2_by_type:
                smells_v2_by_type[smell_type] = []
            smells_v2_by_type[smell_type].append(smell)
        
        # Analyze each smell type independently
        all_added_smells = []
        all_removed_smells = []
        all_remained_smells = []
        all_changed_smells = []
        
        # Get all smell types present in either version
        all_types = set(smells_v1_by_type.keys()) | set(smells_v2_by_type.keys())
        
        for smell_type in all_types:
            type_smells_v1 = smells_v1_by_type.get(smell_type, [])
            type_smells_v2 = smells_v2_by_type.get(smell_type, [])
            
            # Analyze this type independently
            type_result = self.analyze_type_evolution(type_smells_v1, type_smells_v2, v1, v2, smell_type)
            
            all_added_smells.extend(type_result['added_smells'])
            all_removed_smells.extend(type_result['removed_smells']) 
            all_remained_smells.extend(type_result['remained_smells'])
            all_changed_smells.extend(type_result['changed_smells'])
        
        # Calculate alignment statistics from type-based results
        total_aligned = len(all_remained_smells) + len(all_changed_smells)
        total_disappeared = len(all_removed_smells)
        
        return {
            'from_version': v1,
            'to_version': v2,
            'version_stats': {
                'total_smells_v1': len(smells_v1),
                'total_smells_v2': len(smells_v2),
                'aligned_smells': total_aligned,
                'disappeared_smells': total_disappeared,
                'new_smells': len(all_added_smells)
            },
            'added_smells': all_added_smells,
            'removed_smells': all_removed_smells,
            'remained_smells': all_remained_smells,
            'changed_smells': all_changed_smells
        }
    
    def analyze_full_evolution(self) -> Dict:
        """Analyze smell evolution across all versions of the system."""
        time_print(f"Starting smell evolution analysis for {self.system_name} using {self.algorithm}")
        
        versions = self.get_version_list()
        if len(versions) < 2:
            print(f"Need at least 2 versions for evolution analysis. Found: {versions}")
            return {}
        
        evolution_data = {
            'system': self.system_name,
            'algorithm': self.algorithm,
            'versions': versions,
            'evolution_steps': []
        }
        
        # Analyze evolution between consecutive versions
        for i in range(len(versions) - 1):
            v1, v2 = versions[i], versions[i + 1]
            step_data = self.analyze_evolution_between_versions(v1, v2)
            evolution_data['evolution_steps'].append(step_data)
        
        # Generate summary statistics
        total_added = sum(len(step['added_smells']) for step in evolution_data['evolution_steps'])
        total_removed = sum(len(step['removed_smells']) for step in evolution_data['evolution_steps'])
        total_remained = sum(len(step['remained_smells']) for step in evolution_data['evolution_steps'])
        total_changed = sum(len(step['changed_smells']) for step in evolution_data['evolution_steps'])
        
        evolution_data['summary'] = {
            'total_added_smells': total_added,
            'total_removed_smells': total_removed,
            'total_remained_smells': total_remained,
            'total_changed_smells': total_changed,
            'total_evolution_steps': len(evolution_data['evolution_steps'])
        }
        
        return evolution_data
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save evolution analysis results to JSON file."""
        if output_path is None:
            output_dir = f"{SMELLS_ROOT}/{self.system_name}/evolution"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{self.system_name}_{self.algorithm}_evolution.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        time_print(f"Evolution analysis saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print a summary of the evolution analysis."""
        if not results:
            print("No evolution data to summarize.")
            return
        
        print(f"\n=== Smell Evolution Summary for {results['system']} ({results['algorithm']}) ===")
        print(f"Versions analyzed: {' -> '.join(results['versions'])}")
        print(f"Evolution steps: {results['summary']['total_evolution_steps']}")
        print(f"Total added smells: {results['summary']['total_added_smells']}")
        print(f"Total removed smells: {results['summary']['total_removed_smells']}")
        print(f"Total remained smells: {results['summary']['total_remained_smells']}")
        print(f"Total changed smells: {results['summary']['total_changed_smells']}")
        
        print(f"\n=== Step-by-step Evolution ===")
        for step in results['evolution_steps']:
            print(f"\n{step['from_version']} -> {step['to_version']}:")
            print(f"  Added: {len(step['added_smells'])} smells")
            print(f"  Removed: {len(step['removed_smells'])} smells") 
            print(f"  Remained: {len(step['remained_smells'])} smells")
            print(f"  Changed: {len(step['changed_smells'])} smells")
            print(f"  Smell alignment: {step['version_stats']['aligned_smells']}/{step['version_stats']['total_smells_v1']} smells aligned")
    
    def get_smell_status_changes(self) -> Dict:
        """Return all clusters and versions where smell status has changed (new smells added or old smells deleted)."""
        time_print(f"Analyzing smell status changes for {self.system_name} using {self.algorithm}")
        
        versions = self.get_version_list()
        if len(versions) < 2:
            return {
                'system': self.system_name,
                'algorithm': self.algorithm,
                'status_changes': [],
                'summary': {
                    'total_versions': len(versions),
                    'total_status_changes': 0,
                    'smell_additions': 0,
                    'smell_deletions': 0
                }
            }
        
        status_changes = []
        total_additions = 0
        total_deletions = 0
        
        # Analyze evolution between consecutive versions
        for i in range(len(versions) - 1):
            v1, v2 = versions[i], versions[i + 1]
            step_data = self.analyze_evolution_between_versions(v1, v2)
            
            # Process added smells (new smells)
            for added in step_data['added_smells']:
                smell = added['new_smell']
                status_changes.append({
                    'change_type': 'smell_added',
                    'from_version': v1,
                    'to_version': v2,
                    'smell_type': smell['type'],
                    'affected_clusters': smell['clusters'],
                    'cluster_count': len(smell['clusters']),
                    'reason': added['reason'],
                    'smell_id': smell['id'],
                    'cluster_history': added.get('cluster_history', {})
                })
                total_additions += 1
            
            # Process removed smells (deleted smells)
            for removed in step_data['removed_smells']:
                smell = removed['old_smell']
                status_changes.append({
                    'change_type': 'smell_deleted',
                    'from_version': v1,
                    'to_version': v2,
                    'smell_type': smell['type'],
                    'affected_clusters': smell['clusters'],
                    'cluster_count': len(smell['clusters']),
                    'reason': removed['reason'],
                    'smell_id': smell['id']
                })
                total_deletions += 1
            
            # Process changed smells that involve cluster additions/removals
            for changed in step_data['changed_smells']:
                cluster_changes = changed['cluster_changes']
                
                # If clusters were added to the smell
                if cluster_changes['added_clusters']:
                    status_changes.append({
                        'change_type': 'clusters_added_to_smell',
                        'from_version': v1,
                        'to_version': v2,
                        'smell_type': changed['old_smell']['type'],
                        'affected_clusters': cluster_changes['added_clusters'],
                        'cluster_count': len(cluster_changes['added_clusters']),
                        'reason': f"Clusters added to existing {changed['old_smell']['type']} smell",
                        'old_smell_id': changed['old_smell']['id'],
                        'new_smell_id': changed['new_smell']['id'],
                        'similarity': changed['similarity'],
                        'total_old_clusters': changed['old_smell']['clusters'],
                        'total_new_clusters': changed['new_smell']['clusters']
                    })
                
                # If clusters were removed from the smell
                if cluster_changes['removed_clusters']:
                    status_changes.append({
                        'change_type': 'clusters_removed_from_smell',
                        'from_version': v1,
                        'to_version': v2,
                        'smell_type': changed['old_smell']['type'],
                        'affected_clusters': cluster_changes['removed_clusters'],
                        'cluster_count': len(cluster_changes['removed_clusters']),
                        'reason': f"Clusters removed from existing {changed['old_smell']['type']} smell",
                        'old_smell_id': changed['old_smell']['id'],
                        'new_smell_id': changed['new_smell']['id'],
                        'similarity': changed['similarity'],
                        'total_old_clusters': changed['old_smell']['clusters'],
                        'total_new_clusters': changed['new_smell']['clusters']
                    })
        
        return {
            'system': self.system_name,
            'algorithm': self.algorithm,
            'versions': versions,
            'status_changes': status_changes,
            'summary': {
                'total_versions': len(versions),
                'total_status_changes': len(status_changes),
                'smell_additions': total_additions,
                'smell_deletions': total_deletions,
                'cluster_modifications': len(status_changes) - total_additions - total_deletions
            }
        }

    def print_detailed_analysis(self, results: Dict, max_details: int = 5):
        """Print detailed analysis with actual smell information."""
        if not results:
            print("No evolution data to analyze.")
            return
        
        print(f"\n=== Detailed Smell Evolution Analysis for {results['system']} ({results['algorithm']}) ===")
        
        for step in results['evolution_steps']:
            print(f"\n{'='*60}")
            print(f"Evolution from {step['from_version']} to {step['to_version']}")
            print(f"{'='*60}")
            
            # Added smells
            if step['added_smells']:
                print(f"\n--- ADDED SMELLS ({len(step['added_smells'])}) ---")
                for i, added in enumerate(step['added_smells']):
                    smell = added['new_smell']
                    print(f"{i+1}. {smell['type'].upper()} Smell (ID: {smell['id']})")
                    print(f"   Clusters ({smell['cluster_count']}): {smell['clusters']}")
                    print(f"   Reason: {added['reason']}")
                    if 'cluster_history' in added:
                        history = added['cluster_history']
                        if history['existed_but_not_smelly']:
                            print(f"   Previously non-smelly clusters: {history['existed_but_not_smelly']}")
                        if history['did_not_exist']:
                            print(f"   Completely new clusters: {history['did_not_exist']}")
                        if history['was_previously_smelly']:
                            print(f"   Previously smelly clusters: {history['was_previously_smelly']}")
            
            # Removed smells
            if step['removed_smells']:
                print(f"\n--- REMOVED SMELLS ({len(step['removed_smells'])}) ---")
                for i, removed in enumerate(step['removed_smells']):
                    smell = removed['old_smell']
                    print(f"{i+1}. {smell['type'].upper()} Smell (ID: {smell['id']})")
                    print(f"   Clusters ({smell['cluster_count']}): {smell['clusters']}")
                    print(f"   Reason: {removed['reason']}")
            
            # Changed smells
            if step['changed_smells']:
                print(f"\n--- CHANGED SMELLS ({len(step['changed_smells'])}) ---")
                for i, changed in enumerate(step['changed_smells']):
                    old_smell = changed['old_smell']
                    new_smell = changed['new_smell']
                    changes = changed['cluster_changes']
                    print(f"{i+1}. {old_smell['type'].upper()} Smell Evolution (Similarity: {changed['similarity']:.3f})")
                    print(f"   Old ID: {old_smell['id']} ({old_smell['cluster_count']} clusters)")
                    print(f"   New ID: {new_smell['id']} ({new_smell['cluster_count']} clusters)")
                    if changes['added_clusters']:
                        print(f"   Added clusters ({len(changes['added_clusters'])}): {changes['added_clusters']}")
                    if changes['removed_clusters']:
                        print(f"   Removed clusters ({len(changes['removed_clusters'])}): {changes['removed_clusters']}")
                    print(f"   Common clusters ({len(changes['common_clusters'])}): {changes['common_clusters']}")
            
            # Remained smells
            if step['remained_smells']:
                print(f"\n--- REMAINED SMELLS ({len(step['remained_smells'])}) ---")
                smell_types = {}
                for remained in step['remained_smells']:
                    smell_type = remained['old_smell']['type']
                    smell_types[smell_type] = smell_types.get(smell_type, 0) + 1
                
                for smell_type, count in smell_types.items():
                    print(f"   {smell_type.upper()}: {count} smells")


def run_evolution_analysis(system_name: str, algorithm: str = "acdc", detailed: bool = False):
    """Run smell evolution analysis for a system and algorithm."""
    analyzer = SmellEvolutionAnalyzer(system_name, algorithm)
    results = analyzer.analyze_full_evolution()
    
    if results:
        analyzer.save_results(results)
        analyzer.print_summary(results)
        if detailed:
            analyzer.print_detailed_analysis(results)
    else:
        print(f"No evolution analysis results for {system_name} with {algorithm}")


def run_status_changes_analysis(system_name: str, algorithm: str = "acdc"):
    """Run smell status changes analysis for a system and algorithm."""
    analyzer = SmellEvolutionAnalyzer(system_name, algorithm)
    results = analyzer.get_smell_status_changes()
    
    if results and results['status_changes']:
        # Save results
        output_dir = f"{SMELLS_ROOT}/{system_name}/evolution"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{system_name}_{algorithm}_status_changes.json"
        
        with open(output_path, 'w') as f:

            json.dump(results, f, indent=2)
        
        time_print(f"Smell status changes saved to {output_path}")
        
        # Print summary
        print(f"\n=== Smell Status Changes Summary for {results['system']} ({results['algorithm']}) ===")
        print(f"Versions analyzed: {' -> '.join(results['versions'])}")
        print(f"Total status changes: {results['summary']['total_status_changes']}")
        print(f"Smell additions: {results['summary']['smell_additions']}")
        print(f"Smell deletions: {results['summary']['smell_deletions']}")
        print(f"Cluster modifications: {results['summary']['cluster_modifications']}")
        
        # Print detailed changes
        print(f"\n=== Detailed Status Changes ===")
        for change in results['status_changes']:
            print(f"\n{change['from_version']} -> {change['to_version']}:")
            print(f"  Change: {change['change_type']}")
            print(f"  Smell Type: {change['smell_type'].upper()}")
            print(f"  Affected Clusters ({change['cluster_count']}): {change['affected_clusters']}")
            print(f"  Reason: {change['reason']}")
            if 'smell_id' in change:
                print(f"  Smell ID: {change['smell_id']}")
            if 'old_smell_id' in change:
                print(f"  Old Smell ID: {change['old_smell_id']}")
                print(f"  New Smell ID: {change['new_smell_id']}")
                print(f"  Similarity: {change['similarity']:.3f}")
        
        return results
    else:
        print(f"No smell status changes found for {system_name} with {algorithm}")
        return {}


def run_all_algorithms(system_name: str, detailed: bool = False):
    """Run evolution analysis for all algorithms."""
    algorithms = ["acdc", "arc", "limbo", "pkg"]
    
    for algorithm in algorithms:
        time_print(f"Running evolution analysis for {system_name} with {algorithm}")
        try:
            run_evolution_analysis(system_name, algorithm, detailed)
        except Exception as e:
            print(f"Error analyzing {algorithm}: {e}")
        print()


def run_all_algorithms_status_changes(system_name: str):
    """Run smell status changes analysis for all algorithms."""
    algorithms = ["acdc", "arc", "limbo", "pkg"]
    
    for algorithm in algorithms:
        time_print(f"Running status changes analysis for {system_name} with {algorithm}")
        try:
            run_status_changes_analysis(system_name, algorithm)
        except Exception as e:
            print(f"Error analyzing {algorithm}: {e}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_smell_evolution.py <system_name> [algorithm] [--detailed] [--status-changes]")
        print("Available algorithms: acdc, arc, limbo, pkg, all")
        print("Use --detailed for detailed smell information")
        print("Use --status-changes to analyze only smell status changes (additions/deletions)")
        sys.exit(1)
    
    system_name = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "all"
    detailed = "--detailed" in sys.argv
    status_changes_only = "--status-changes" in sys.argv
    
    if status_changes_only:
        if algorithm == "all":
            run_all_algorithms_status_changes(system_name)
        else:
            run_status_changes_analysis(system_name, algorithm)
    else:
        if algorithm == "all":
            run_all_algorithms(system_name, detailed)
        else:
            run_evolution_analysis(system_name, algorithm, detailed)