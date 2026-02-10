"""
Preprocessing pipeline wiring ARCADE fact extraction and clustering together.

Intended to run immediately before ``classification_pipeline`` for systems
that do not yet have facts/clusters generated locally (e.g., OpenRGB).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

from src.preprocessing.dependency_filter import remove_dependencies_by_keyword
from src.preprocessing.detect_smells import run_smell_detection
from src.preprocessing.extract_facts import extract_facts
from src.preprocessing.run_clustering import cluster_system
from src.smell.analyze_smell_evolution import run_all_algorithms as run_smell_evolution


@dataclass
class PreprocessingConfig:
    system_name: str
    language: str
    mallet_mode: str = "all"
    versions: Optional[List[str]] = None
    memory: str = "8"
    skip_facts: bool = False
    skip_clustering: bool = False
    skip_keyword_filter: bool = False
    skip_smell_detection: bool = False
    skip_smell_evolution: bool = False
    filter_keywords: List[str] = field(default_factory=lambda: ["test"])


def run_preprocessing_pipeline(config: PreprocessingConfig) -> None:
    """Run fact extraction followed by clustering for the requested system."""
    print(
        f"Preprocessing pipeline configuration:\n"
        f"  System: {config.system_name}\n"
        f"  Language: {config.language}\n"
        f"  Mallet mode: {config.mallet_mode}\n"
        f"  Selected versions: {config.versions or 'all'}\n"
        f"  Cluster memory (GB): {config.memory}\n"
        f"  Skip facts: {config.skip_facts}\n"
        f"  Skip clustering: {config.skip_clustering}\n"
        f"  Skip keyword filtering: {config.skip_keyword_filter}\n"
        f"  Skip smell detection: {config.skip_smell_detection}\n"
        f"  Skip smell evolution: {config.skip_smell_evolution}\n"
        f"  Filter keywords: {', '.join(config.filter_keywords) if config.filter_keywords else 'none'}"
    )

    if not config.skip_facts:
        extract_facts(
            system_name=config.system_name,
            language=config.language,
            mallet_mode=config.mallet_mode,
            selected_versions=config.versions,
        )
    else:
        print("\nSkipping fact extraction per --skip-facts.")

    if not config.skip_clustering:
        cluster_system(
            system_name=config.system_name,
            language=config.language,
            memory=config.memory,
        )
    else:
        print("\nSkipping clustering per --skip-clustering.")

    if not config.skip_keyword_filter:
        remove_dependencies_by_keyword(
            system_name=config.system_name,
            keywords=config.filter_keywords,
        )
    else:
        print("\nSkipping dependency keyword filtering per --skip-keyword-filter.")

    if not config.skip_smell_detection:
        run_smell_detection(config.system_name)
    else:
        print("\nSkipping smell detection per --skip-smell-detection.")

    if not config.skip_smell_evolution:
        run_smell_evolution(config.system_name)
    else:
        print("\nSkipping smell evolution per --skip-smell-evolution.")

    print("\nPreprocessing pipeline completed.")


def parse_args() -> PreprocessingConfig:
    parser = argparse.ArgumentParser(
        description="Generate ARCADE facts and clusters before running classification."
    )
    parser.add_argument(
        "--system",
        required=True,
        help="System slug (matches folders under data/facts and data/clusters).",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Programming language for Understand + clustering (e.g., c++, java).",
    )
    parser.add_argument(
        "--mallet-mode",
        choices=["all", "selected"],
        default="all",
        help="Whether to run Mallet across all versions or a selected subset.",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Specific versions to process when --mallet-mode=selected.",
    )
    parser.add_argument(
        "--memory",
        default="8",
        help="Heap size in GB for clustering algorithms that use Clusterer (default: 8).",
    )
    parser.add_argument(
        "--skip-facts",
        action="store_true",
        help="Skip fact extraction (requires existing data/facts/<system> artifacts).",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering (requires existing data/clusters/<system> artifacts).",
    )
    parser.add_argument(
        "--skip-keyword-filter",
        action="store_true",
        help="Skip dependency keyword filtering on *_deps.rsf files.",
    )
    parser.add_argument(
        "--skip-smell-detection",
        action="store_true",
        help="Skip ARCADE smell detection (requires existing data/smells/<system> artifacts).",
    )
    parser.add_argument(
        "--skip-smell-evolution",
        action="store_true",
        help="Skip smell evolution analysis.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        help="Keyword to remove from dependency RSF paths (case-insensitive). "
        "Provide multiple times for more than one keyword. Default: test.",
    )

    args = parser.parse_args()

    if args.mallet_mode == "selected" and not args.versions:
        parser.error("--versions must be provided when --mallet-mode=selected")

    keywords = args.keywords or ["test"]

    return PreprocessingConfig(
        system_name=args.system.strip(),
        language=args.language.strip(),
        mallet_mode=args.mallet_mode,
        versions=args.versions,
        memory=args.memory,
        skip_facts=args.skip_facts,
        skip_clustering=args.skip_clustering,
        skip_keyword_filter=args.skip_keyword_filter,
        skip_smell_detection=args.skip_smell_detection,
        skip_smell_evolution=args.skip_smell_evolution,
        filter_keywords=keywords,
    )


def main() -> None:
    config = parse_args()
    run_preprocessing_pipeline(config)


if __name__ == "__main__":
    main()
