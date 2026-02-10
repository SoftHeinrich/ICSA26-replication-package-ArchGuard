#!/usr/bin/env python3
"""
Configuration and constants for issue mapping tools.
"""

import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class MappingType(Enum):
    """Types of decision mappings."""
    VERSION = "version"
    CLUSTER = "cluster"


class Algorithm(Enum):
    """Clustering algorithms."""
    ACDC = "acdc"
    ARC = "arc"
    PKG = "pkg"
    LIMBO = "limbo"


@dataclass(frozen=True)
class PathConfig:
    """Path configuration for the mapping tools."""
    project_root: Path
    data_smell_dir: Path
    issues_dir: Path
    smells_dir: Path
    subject_systems_dir: Path
    facts_dir: Path
    clusters_dir: Path
    
    @classmethod
    def from_project_root(cls, project_root: Path) -> 'PathConfig':
        """Create path configuration from project root."""
        data_smell = project_root / "data-smell"
        return cls(
            project_root=project_root,
            data_smell_dir=data_smell,
            issues_dir=data_smell / "issues",
            smells_dir=data_smell / "smells",
            subject_systems_dir=data_smell / "subject_systems",
            facts_dir=data_smell / "facts",
            clusters_dir=data_smell / "clusters"
        )


@dataclass
class ProcessingConfig:
    """Configuration for processing operations."""
    batch_size: int = 1000
    progress_interval: int = 1000
    max_file_size_mb: int = 1024
    streaming_threshold_mb: int = 100
    timeout_seconds: int = 300
    encoding: str = 'utf-8'


@dataclass
class SystemConfig:
    """System-specific configuration."""
    name: str
    aliases: List[str]
    version_pattern: str
    
    @property
    def all_names(self) -> List[str]:
        """Get all names including the primary name and aliases."""
        return [self.name] + self.aliases


class ConfigManager:
    """Manages configuration for mapping tools."""
    
    # System configurations
    SYSTEM_CONFIGS = {
        'inkscape': SystemConfig(
            name='inkscape',
            aliases=['inkscape-full'],
            version_pattern=r'(\d+\.\d+(?:\.\d+)?)'
        ),
        'opencv': SystemConfig(
            name='opencv',
            aliases=['opencv-non-patch'],
            version_pattern=r'(\d+\.\d+(?:\.\d+)?)'
        ),
        'wireshark': SystemConfig(
            name='wireshark',
            aliases=['wireshark-non-patch'],
            version_pattern=r'(\d+\.\d+(?:\.\d+)?)'
        )
    }
    
    # File patterns
    FILE_PATTERNS = {
        'recovar': '*recovar*.json',
        'smell': '*_smells.json',
        'decision_version_mapping': '*d2v*.json',
        'decision_cluster_mapping': '*d2c*.json',
        'evolution': '*_acdc_improved.json'
    }
    
    # JSON paths for streaming
    JSON_PATHS = {
        'decisions': 'decisions.item',
        'issues': 'issues.item',
        'items': 'item',
        'smellcollection': 'smellcollection.item'
    }
    
    def __init__(self, project_root: Path = None):
        """Initialize configuration manager."""
        if project_root is None:
            # Auto-detect project root
            current = Path(__file__).parent
            while current.parent != current:
                if (current / "CLAUDE.md").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path(__file__).parent.parent.parent
        
        self.paths = PathConfig.from_project_root(project_root)
        self.processing = ProcessingConfig()
    
    def get_system_config(self, system_name: str) -> SystemConfig:
        """Get system configuration, with fallback for aliases."""
        # Direct match
        if system_name in self.SYSTEM_CONFIGS:
            return self.SYSTEM_CONFIGS[system_name]
        
        # Check aliases
        for config in self.SYSTEM_CONFIGS.values():
            if system_name in config.aliases:
                return config
        
        # Create default config
        return SystemConfig(
            name=system_name,
            aliases=[],
            version_pattern=r'(\d+\.\d+(?:\.\d+)?)'
        )
    
    def get_decision_system_name(self, system_name: str) -> str:
        """Get the system name used in decision files."""
        config = self.get_system_config(system_name)
        # For backward compatibility, use the full name for decisions
        if system_name == 'inkscape':
            return 'inkscape-full'
        return config.name