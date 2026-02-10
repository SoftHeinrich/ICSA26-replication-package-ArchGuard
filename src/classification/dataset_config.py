"""
Shared dataset configuration for classification modules.

This file contains constants used by both zero_shot_llm_classifier.py
and src/casestudy/zero_shot_all.py to ensure consistency.
"""

from pathlib import Path

# Dataset base directory (uses supervised_ml_all output structure)
# Note: Changed from timestamp to "run" for privacy
DATASET_BASE_DIR = Path("Output/supervised_ml_all/run")

# Classification data filename
CLASSIFICATION_DATA_FILENAME = "classification_data_clean.json"

# Projects and their available detectors
PROJECT_DETECTORS = {
    "inkscape": ["acdc"],  # inkscape only has ACDC
    "shepard": ["acdc", "arcan"],
    "stackgres": ["acdc", "arcan"],
}

# Mapping project names to directory names in classification folders
PROJECT_DIR_NAMES = {
    "inkscape": "inkscape",
    "stackgres": "ongresinc-stackgres",
    "shepard": "dlr-shepard-shepard",
}

# Dataset folder names: (project, detector) -> folder name
DATASET_FOLDERS = {
    ("inkscape", "acdc"): "inkscape_acdc",
    ("stackgres", "acdc"): "stackgres_acdc",
    ("stackgres", "arcan"): "stackgres_arcan",
    ("shepard", "acdc"): "shepard_acdc",
    ("shepard", "arcan"): "shepard_arcan",
}

# Prompt style mapping for detectors
DETECTOR_PROMPT_STYLES = {
    "acdc": "arcade",
    "arcan": "arcan",
}
