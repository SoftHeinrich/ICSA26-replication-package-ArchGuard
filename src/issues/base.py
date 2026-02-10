#!/usr/bin/env python3
"""
Base classes and interfaces for issue mapping tools.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.issues.config import ConfigManager
from src.issues.utils import Logger, JSONProcessor, FileHandler, ProgressTracker, DataValidator


@dataclass
class ProcessingResult:
    """Standard result format for processing operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    processing_time: Optional[str] = None


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.logger = Logger(self.__class__.__name__)
        self.start_time = None
        self.progress = None
    
    def _start_processing(self, description: str) -> None:
        """Initialize processing session."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {description}")
        self.progress = ProgressTracker(self.logger)
    
    def _finish_processing(self, description: str) -> str:
        """Finalize processing session and return duration."""
        if self.start_time:
            duration = str(datetime.now() - self.start_time)
            self.logger.info(f"Completed {description} in {duration}")
            if self.progress:
                self.progress.finish()
            return duration
        return "Unknown"
    
    @abstractmethod
    def process(self, *args, **kwargs) -> ProcessingResult:
        """Main processing method to be implemented by subclasses."""
        pass


class DecisionMapperBase(BaseProcessor):
    """Base class for decision mapping operations."""
    
    def __init__(self, config_manager: ConfigManager = None):
        super().__init__(config_manager)
        self.file_handler = FileHandler()
        self.json_processor = JSONProcessor()
        self.validator = DataValidator()
    
    def find_input_files(self, system_name: str, pattern: str, 
                        search_dir: Optional[Path] = None) -> List[Path]:
        """Find input files for processing."""
        if search_dir is None:
            system_config = self.config.get_system_config(system_name)
            search_dir = self.config.paths.issues_dir / system_config.name
        
        if not search_dir.exists():
            self.logger.warning(f"Directory not found: {search_dir}")
            return []
        
        files = self.file_handler.find_files_by_pattern(search_dir, pattern)
        self.logger.info(f"Found {len(files)} files matching {pattern} in {search_dir}")
        
        return files
    
    def save_results(self, results: Dict[str, Any],
                    output_path: Optional[Path] = None) -> Path:
        """Save mapping results to JSON file."""
        if output_path is None:
            system_name = results.get('system', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__.lower()}_{system_name}_{timestamp}.json"
            output_path = self.config.paths.issues_dir / system_name / filename
        else:
            # If output_path is a directory, generate a filename within it
            if output_path.is_dir():
                system_name = results.get('system', 'unknown')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.__class__.__name__.lower()}_{system_name}_{timestamp}.json"
                output_path = output_path / filename

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.json_processor.save_json(results, output_path)
        self.logger.info(f"Results saved to: {output_path}")
        return output_path


class SmellProcessorBase(BaseProcessor):
    """Base class for smell processing operations."""
    
    def __init__(self, config_manager: ConfigManager = None):
        super().__init__(config_manager)
        self.file_handler = FileHandler()
        self.json_processor = JSONProcessor()
        self.validator = DataValidator()
    
    def find_smell_files(self, system_name: str, algorithm: str = None) -> List[Path]:
        """Find smell files for a given system and optional algorithm."""
        smell_dir = self.config.paths.smells_dir / system_name
        
        if not smell_dir.exists():
            self.logger.warning(f"Smell directory not found: {smell_dir}")
            return []
        
        pattern = f"{algorithm}/*_smells.json" if algorithm else "*/*_smells.json"
        files = self.file_handler.find_files_in_subdirs(smell_dir, "*_smells.json")
        
        if algorithm:
            # Filter by algorithm
            files = [f for f in files if f.parent.name.lower() == algorithm.lower()]
        
        self.logger.info(f"Found {len(files)} smell files for {system_name}")
        return sorted(files)
    
    def load_smell_data(self, smell_file: Path) -> List[Dict[str, Any]]:
        """Load smell data from a JSON file."""
        try:
            data = self.json_processor.load_json(smell_file)
            
            # Extract smells from the structure
            if 'smellcollection' in data:
                return data['smellcollection']
            elif isinstance(data, list):
                return data
            else:
                return [data]
        except Exception as e:
            self.logger.error(f"Error loading smell file {smell_file}: {e}")
            return []


class StreamingProcessorMixin:
    """Mixin for streaming JSON processing capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_threshold_mb = 100  # Default threshold for streaming
    
    def should_use_streaming(self, file_path: Path) -> bool:
        """Determine if file should use streaming processing."""
        file_size_mb = self.file_handler.get_file_size_mb(file_path)
        return file_size_mb > self.streaming_threshold_mb
    
    def process_file_streaming(self, file_path: Path, 
                             json_path: str = "decisions.item") -> ProcessingResult:
        """Process large JSON files using streaming."""
        try:
            with self.json_processor.stream_json_items(file_path, json_path) as items:
                processed_count = 0
                
                for item in items:
                    if self.validator.validate_decision_record(item):
                        self._process_item(item)
                        processed_count += 1
                        
                        if self.progress:
                            self.progress.update()
                
                return ProcessingResult(
                    success=True,
                    message=f"Successfully processed {processed_count} items from {file_path.name}",
                    stats={"processed_items": processed_count, 
                           "file_size_mb": self.file_handler.get_file_size_mb(file_path)}
                )
                
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Error processing {file_path.name}: {e}",
                errors=[str(e)]
            )
    
    @abstractmethod
    def _process_item(self, item: Dict[str, Any]) -> None:
        """Process a single item (to be implemented by subclasses)."""
        pass


class EvolutionProcessorBase(BaseProcessor):
    """Base class for evolution data processing."""
    
    def __init__(self, config_manager: ConfigManager = None):
        super().__init__(config_manager)
        self.json_processor = JSONProcessor()
    
    def load_evolution_data(self, evolution_file: Path) -> Optional[Dict[str, Any]]:
        """Load evolution data from JSON file."""
        try:
            data = self.json_processor.load_json(evolution_file)
            self.logger.info(f"Loaded evolution data from {evolution_file.name}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading evolution file {evolution_file}: {e}")
            return None
    
    def find_evolution_files(self, system_name: str, algorithm: str = None) -> List[Path]:
        """Find evolution files for a system."""
        search_dirs = [
            self.config.paths.project_root,
            Path.cwd(),
            self.config.paths.data_smell_dir
        ]
        
        patterns = []
        if algorithm:
            patterns.append(f"*{system_name}*{algorithm}*evolution*.json")
            patterns.append(f"*{system_name}*{algorithm}*improved*.json")
        else:
            patterns.append(f"*{system_name}*evolution*.json")
            patterns.append(f"*{system_name}*improved*.json")
        
        found_files = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for pattern in patterns:
                files = self.file_handler.find_files_by_pattern(search_dir, pattern)
                found_files.extend(files)
            
            if found_files:
                break
        
        return list(set(found_files))  # Remove duplicates


class AnalysisBase(BaseProcessor):
    """Base class for analysis operations."""
    
    def __init__(self, config_manager: ConfigManager = None):
        super().__init__(config_manager)
        self.json_processor = JSONProcessor()
        self.results = {}
    
    def add_analysis_result(self, key: str, result: Any) -> None:
        """Add a result to the analysis."""
        self.results[key] = result
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of analysis results."""
        return {
            'total_analyses': len(self.results),
            'analysis_keys': list(self.results.keys()),
            'generated_at': datetime.now().isoformat()
        }


class ConfigurableCLIBase(ABC):
    """Base class for CLI applications with standardized argument handling."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.logger = Logger(self.__class__.__name__)
    
    @abstractmethod
    def setup_arguments(self, parser) -> None:
        """Setup command line arguments (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def run(self, args) -> int:
        """Main execution method (to be implemented by subclasses)."""
        pass
    
    def execute(self) -> int:
        """Standard execution flow for CLI applications."""
        import argparse
        
        parser = argparse.ArgumentParser(
            description=self.__class__.__doc__ or "Command line tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Common arguments
        parser.add_argument('--config-dir', type=Path,
                           help='Configuration directory path')
        parser.add_argument('--quiet', action='store_true',
                           help='Suppress detailed output')
        parser.add_argument('--debug', action='store_true',
                           help='Enable debug logging')
        
        # Subclass-specific arguments
        self.setup_arguments(parser)
        
        args = parser.parse_args()
        
        # Configure logging level
        if args.debug:
            self.logger = Logger(self.__class__.__name__, level=10)  # DEBUG
        elif args.quiet:
            self.logger = Logger(self.__class__.__name__, level=30)  # WARNING
        
        try:
            return self.run(args)
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1


class DataMatcher:
    """Utility class for matching data across different structures."""
    
    @staticmethod
    def normalize_clusters(clusters: List[str]) -> Set[str]:
        """Normalize cluster names for comparison."""
        normalized = set()
        for cluster in clusters:
            normalized.add(cluster)
            normalized.add(cluster.replace('/', '\\'))
            normalized.add(cluster.replace('\\', '/'))
        return normalized
    
    @staticmethod
    def calculate_cluster_overlap(clusters1: List[str], clusters2: List[str]) -> Tuple[float, int, Set[str]]:
        """Calculate overlap ratio and details between two cluster sets."""
        set1 = DataMatcher.normalize_clusters(clusters1)
        set2 = DataMatcher.normalize_clusters(clusters2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0:
            return 0.0, 0, set()
        
        overlap_ratio = len(intersection) / len(union)
        return overlap_ratio, len(intersection), intersection
    
    @staticmethod
    def find_best_cluster_match(target_clusters: List[str], 
                               candidate_mappings: List[Dict[str, Any]],
                               min_overlap: float = 0.5) -> Optional[Dict[str, Any]]:
        """Find the best cluster match from candidates."""
        best_match = None
        best_overlap = 0.0
        
        for mapping in candidate_mappings:
            mapping_clusters = mapping.get('clusters', [])
            overlap_ratio, overlap_count, overlapping = DataMatcher.calculate_cluster_overlap(
                target_clusters, mapping_clusters
            )
            
            if overlap_ratio >= min_overlap and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match = mapping.copy()
                best_match.update({
                    'overlap_ratio': overlap_ratio,
                    'overlap_count': overlap_count,
                    'overlapping_clusters': list(overlapping)
                })
        
        return best_match