#!/usr/bin/env python3
"""
Shared utilities for issue mapping tools.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from contextlib import contextmanager
import ijson


class Logger:
    """Enhanced logger with timestamp formatting."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)


class VersionNormalizer:
    """Utilities for version string handling."""
    
    @staticmethod
    def normalize_version(version: str) -> str:
        """
        Normalize version strings, treating minor versions as next release.
        
        Examples:
        - 1.2.1 -> 1.3.0
        - 1.4.2 -> 1.5.0
        - 1.0 -> 1.0.0
        - 1.2 -> 1.2.0
        """
        if not version:
            return version
        
        # Remove common prefixes and suffixes
        version = version.strip().lower()
        version = re.sub(r'^v\.?', '', version)  # Remove 'v' or 'v.'
        version = re.sub(r'[+\-].*$', '', version)  # Remove suffixes like '+devel', '-alpha'
        
        # Extract version parts
        parts = version.split('.')
        
        try:
            if len(parts) >= 3:
                # For versions like 1.2.1, increment minor version
                major = int(parts[0])
                minor = int(parts[1])
                patch = int(parts[2])
                
                if patch > 0:
                    # Increment minor version and reset patch
                    minor += 1
                    return f"{major}.{minor}.0"
                else:
                    return f"{major}.{minor}.{patch}"
            
            elif len(parts) == 2:
                # For versions like 1.2, add .0
                major = int(parts[0])
                minor = int(parts[1])
                return f"{major}.{minor}.0"
            
            elif len(parts) == 1:
                # For versions like 1, add .0.0
                major = int(parts[0])
                return f"{major}.0.0"
                
        except ValueError:
            # If parsing fails, return original
            return version
        
        return version
    
    @staticmethod
    def get_next_major_version(version: str) -> Optional[str]:
        """Calculate the next major version from current version."""
        match = re.match(r'(\d+)\.(\d+)', version)
        if not match:
            return None
        
        major = int(match.group(1))
        minor = int(match.group(2))
        
        # Calculate next minor version
        next_minor = minor + 1
        
        return f"{major}.{next_minor}"


class FileHandler:
    """Utilities for file operations."""
    
    @staticmethod
    def find_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
        """Find files matching a glob pattern."""
        if not directory.exists():
            return []
        
        return sorted(directory.glob(pattern))
    
    @staticmethod
    def find_files_in_subdirs(directory: Path, pattern: str) -> List[Path]:
        """Find files matching pattern in subdirectories."""
        if not directory.exists():
            return []
        
        return sorted(directory.rglob(pattern))
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)


class JSONProcessor:
    """Enhanced JSON processing utilities."""
    
    @staticmethod
    def load_json(file_path: Path, encoding: str = 'utf-8') -> Any:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise IOError(f"Error reading {file_path}: {e}")
    
    @staticmethod
    def save_json(data: Any, file_path: Path, encoding: str = 'utf-8', 
                  indent: int = 2) -> None:
        """Save data to JSON file."""
        FileHandler.ensure_directory(file_path.parent)
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    @contextmanager
    def stream_json_items(file_path: Path, json_path: str):
        """Stream JSON items from large files."""
        with open(file_path, 'rb') as file:
            try:
                items = ijson.items(file, json_path)
                yield items
            except Exception as e:
                # Fallback to full JSON loading
                file.seek(0)
                data = json.load(file)
                
                # Try to find items in different structures
                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    for key in ['issues', 'decisions', 'data', 'items', 'smellcollection']:
                        if key in data and isinstance(data[key], list):
                            items = data[key]
                            break
                
                yield iter(items)


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_decision_record(decision: Dict[str, Any]) -> bool:
        """Validate that a record looks like a decision."""
        required_fields = ['id', 'number', 'issue_id']
        return any(field in decision for field in required_fields)
    
    @staticmethod
    def validate_smell_record(smell: Dict[str, Any]) -> bool:
        """Validate that a record looks like a smell."""
        required_fields = ['type', 'clusters']
        return all(field in smell for field in required_fields)
    
    @staticmethod
    def extract_decision_id(decision: Dict[str, Any]) -> Optional[str]:
        """Extract decision ID from various formats."""
        for field in ['id', 'number', 'issue_id']:
            if field in decision and decision[field] is not None:
                return str(decision[field])
        return None


class PathResolver:
    """Path resolution utilities."""
    
    @staticmethod
    def resolve_algorithm_and_version(file_path: Path) -> tuple[str, str]:
        """Extract algorithm and version from file path."""
        # Path structure: .../smells/system/algorithm/system-version_smells.json
        algorithm = file_path.parent.name
        filename = file_path.stem
        
        # Extract version from filename (e.g., inkscape-1.0_smells -> 1.0)
        match = re.search(r'-(\d+\.\d+(?:\.\d+)?)_smells$', filename)
        version = match.group(1) if match else "unknown"
        
        return algorithm, version
    
    @staticmethod
    def normalize_path_separators(path: str) -> List[str]:
        """Return path variants with different separators."""
        return [
            path,
            path.replace('/', '\\'),
            path.replace('\\', '/')
        ]


class ProgressTracker:
    """Progress tracking utility."""
    
    def __init__(self, logger: Logger, interval: int = 1000):
        self.logger = logger
        self.interval = interval
        self.count = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter."""
        self.count += increment
        if self.count % self.interval == 0:
            elapsed = datetime.now() - self.start_time
            rate = self.count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            self.logger.info(f"Processed {self.count} items ({rate:.1f} items/sec)")
    
    def finish(self) -> None:
        """Log final progress."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Completed processing {self.count} items in {elapsed}")


class ResultsSummarizer:
    """Utility for generating result summaries."""
    
    @staticmethod
    def summarize_mapping_results(results: Dict[str, Any], mapping_type: str) -> str:
        """Generate a summary of mapping results."""
        if not results or 'file_results' not in results:
            return "No results to summarize"
        
        lines = []
        lines.append(f"=== {mapping_type.title()} Mapping Summary ===")
        lines.append(f"System: {results.get('system', 'Unknown')}")
        lines.append(f"Processing time: {results.get('processing_time', 'Unknown')}")
        lines.append(f"Files processed: {len(results.get('file_results', {}))}")
        
        file_results = results.get('file_results', {})
        
        for file_key, file_data in file_results.items():
            lines.append(f"\n--- {file_key} ---")
            
            file_info = file_data.get('file_info', {})
            mapping = file_data.get('mapping', {})
            summary = mapping.get('summary', {})
            
            lines.append(f"  File: {file_info.get('file', 'Unknown')}")
            lines.append(f"  Items processed: {file_info.get('processed_items', 0)}")
            lines.append(f"  File size: {file_info.get('file_size_mb', 0):.2f} MB")
            
            if summary:
                lines.append(f"  Mapping Statistics:")
                lines.append(f"    Total decisions: {summary.get('total_decisions', 0)}")
                
                if mapping_type == 'version':
                    lines.append(f"    Decisions with versions: {summary.get('decisions_with_versions', 0)}")
                    lines.append(f"    Unique versions: {summary.get('unique_versions', 0)}")
                elif mapping_type == 'cluster':
                    lines.append(f"    Decisions with clusters: {summary.get('decisions_with_clusters', 0)}")
                    lines.append(f"    Unique clusters: {summary.get('unique_clusters', 0)}")
                
                lines.append(f"    Total mappings: {summary.get('total_mappings', 0)}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def summarize_smell_mapping_results(results: Dict[str, Any]) -> str:
        """Generate a summary of smell-to-decision mapping results."""
        if not results:
            return "No results to summarize"
        
        summary = results.get('summary', {})
        
        lines = []
        lines.append("=== Smell to Decision Mapping Summary ===")
        lines.append(f"System: {results.get('system', 'Unknown')}")
        lines.append(f"Processing time: {results.get('processing_time', 'Unknown')}")
        lines.append(f"Algorithms processed: {', '.join(summary.get('algorithms_processed', []))}")
        
        lines.append("\nStatistics:")
        lines.append(f"  Total smell files: {summary.get('total_smell_files', 0)}")
        lines.append(f"  Total smells: {summary.get('total_smells', 0)}")
        lines.append(f"  Smells with decisions: {summary.get('smells_with_decisions', 0)}")
        lines.append(f"  Total decision mappings: {summary.get('total_decision_mappings', 0)}")
        
        return '\n'.join(lines)