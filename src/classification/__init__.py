"""
Simple classification module for smell detection.
Uses traditional ML with imbalance learning strategies.
"""

from .simple_ml import run_traditional_ml_experiments

# Version information
__version__ = "3.0.0-simple"
__author__ = "Smell Detection Team"

# Simple API
__all__ = [
    'run_traditional_ml_experiments',
    'run_cross_project_classification',
    '__version__',
    '__author__'
]


def __getattr__(name):
    if name == "run_cross_project_classification":
        from .cross_project import run_cross_project_classification
        return run_cross_project_classification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
