#!/usr/bin/env python3
"""
Zero-Shot LLM Classification Wrapper for All Project-Detector Combinations

Runs zero-shot LLM classification across all combinations of:
  - Projects: inkscape, shepard, stackgres
  - Detectors: acdc (inkscape), acdc/arcan (shepard, stackgres)

Output folder structure:
  Output/zero_shot_runs/
    ├── {timestamp}/
    │   ├── summary.json
    │   ├── inkscape/
    │   │   └── acdc/
    │   │       ├── results_{model}_{text_source}.csv
    │   │       └── metrics_{model}_{text_source}.json
    │   ├── shepard/
    │   │   ├── acdc/
    │   │   └── arcan/
    │   └── stackgres/
    │       ├── acdc/
    │       └── arcan/

Usage:
    # Run all combinations
    python -m src.casestudy.zero_shot_all

    # Run specific projects
    python -m src.casestudy.zero_shot_all --projects inkscape shepard

    # Run specific detectors
    python -m src.casestudy.zero_shot_all --detectors acdc

    # Specify model and text source
    python -m src.casestudy.zero_shot_all --model gpt-4o-mini --text-source combined

    # Enable chain-of-thought
    python -m src.casestudy.zero_shot_all --use-cot

    # With parallel workers and rate limiting
    python -m src.casestudy.zero_shot_all --max-workers 8 --api-delay 0.5

    # Dry run to see what would be executed
    python -m src.casestudy.zero_shot_all --dry-run

    # Run from cache only (no API calls)
    python -m src.casestudy.zero_shot_all --cache-only
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Import shared constants from dataset_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.classification.dataset_config import (
    DATASET_BASE_DIR,
    CLASSIFICATION_DATA_FILENAME,
    PROJECT_DETECTORS,
    PROJECT_DIR_NAMES,
    DATASET_FOLDERS,
    DETECTOR_PROMPT_STYLES,
)

# Default settings
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TEXT_SOURCE = "combined"
DEFAULT_OUTPUT_BASE = Path("Output/zero_shot_runs")

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    project: str
    detector: str
    data_path: Path
    prompt_style: str

    @property
    def key(self) -> str:
        return f"{self.project}_{self.detector}"


@dataclass
class RunConfig:
    """Configuration for a single zero-shot run."""
    project: str
    detector: str
    data_path: Path
    prompt_style: str
    text_source: str
    model: str
    use_cot: bool
    output_dir: Path

    @property
    def run_name(self) -> str:
        cot_suffix = "_cot" if self.use_cot else ""
        return f"{self.project}_{self.detector}_{self.text_source}_{self.model}{cot_suffix}"


@dataclass
class RunResult:
    """Results from a single classification run."""
    project: str
    detector: str
    text_source: str
    model: str
    use_cot: bool
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_samples: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    output_dir: str = ""
    error: Optional[str] = None
    runtime_seconds: float = 0.0


# ============================================================================
# Dataset Discovery
# ============================================================================


def discover_datasets(
    projects: Optional[List[str]] = None,
    detectors: Optional[List[str]] = None,
) -> List[DatasetConfig]:
    """Discover all available datasets based on filters."""
    configs = []
    target_projects = projects or list(PROJECT_DETECTORS.keys())

    for project in target_projects:
        if project not in PROJECT_DETECTORS:
            LOGGER.warning("Unknown project: %s", project)
            continue

        available_detectors = PROJECT_DETECTORS[project]
        target_detectors = detectors or available_detectors

        for detector in target_detectors:
            if detector not in available_detectors:
                LOGGER.debug(
                    "Detector %s not available for project %s (available: %s)",
                    detector, project, available_detectors
                )
                continue

            # Find dataset path
            key = (project, detector)
            if key not in DATASET_FOLDERS:
                LOGGER.warning("No dataset config for %s/%s", project, detector)
                continue

            dataset_folder = DATASET_BASE_DIR / DATASET_FOLDERS[key]
            if not dataset_folder.exists():
                LOGGER.warning("Dataset folder not found: %s", dataset_folder)
                continue

            project_dir_name = PROJECT_DIR_NAMES.get(project, project)
            classification_dir = dataset_folder / "data" / "classification" / project_dir_name
            data_file = classification_dir / CLASSIFICATION_DATA_FILENAME

            if not data_file.exists():
                LOGGER.warning("Classification data not found: %s", data_file)
                continue

            prompt_style = DETECTOR_PROMPT_STYLES.get(detector, "arcade")

            configs.append(DatasetConfig(
                project=project,
                detector=detector,
                data_path=data_file,
                prompt_style=prompt_style,
            ))

    return configs


# ============================================================================
# Run Configuration Generation
# ============================================================================


def generate_run_configs(
    datasets: List[DatasetConfig],
    text_source: str,
    model: str,
    use_cot: bool,
    output_base: Path,
) -> List[RunConfig]:
    """Generate run configurations for all datasets."""
    configs = []

    for dataset in datasets:
        # Output structure: output_base/project/detector/
        output_dir = output_base / dataset.project / dataset.detector

        configs.append(RunConfig(
            project=dataset.project,
            detector=dataset.detector,
            data_path=dataset.data_path,
            prompt_style=dataset.prompt_style,
            text_source=text_source,
            model=model,
            use_cot=use_cot,
            output_dir=output_dir,
        ))

    return configs


# ============================================================================
# Execution
# ============================================================================


def run_zero_shot_classification(
    config: RunConfig,
    cache_only: bool = False,
    api_delay: float = 1.0,
    max_workers: int = 4,
) -> RunResult:
    """Execute a single zero-shot classification run."""
    import time
    start_time = time.time()

    try:
        # Import the classifier
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from classification.zero_shot_llm_classifier import run_zero_shot_classification as run_classifier

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Build output file name
        cot_suffix = "_cot" if config.use_cot else ""
        safe_model = config.model.replace(":", "_").replace("/", "_")
        output_file = config.output_dir / f"results_{safe_model}_{config.text_source}{cot_suffix}.csv"

        LOGGER.info(
            "Running: %s/%s [%s] model=%s cot=%s",
            config.project, config.detector, config.text_source,
            config.model, config.use_cot
        )
        LOGGER.info("  Data: %s", config.data_path)
        LOGGER.info("  Output: %s", output_file)

        # Run classification
        df, results = run_classifier(
            data_path=str(config.data_path),
            text_source=config.text_source,
            model=config.model,
            use_cot=config.use_cot,
            api_delay=api_delay,
            output_file=str(output_file),
            cache_only=cache_only,
            prompt_style=config.prompt_style,
            max_workers=max_workers,
        )

        runtime = time.time() - start_time

        return RunResult(
            project=config.project,
            detector=config.detector,
            text_source=config.text_source,
            model=config.model,
            use_cot=config.use_cot,
            accuracy=results.get("accuracy", 0.0),
            precision=results.get("precision", 0.0),
            recall=results.get("recall", 0.0),
            f1=results.get("f1", 0.0),
            total_samples=results.get("total_samples", 0),
            tp=results.get("tp", 0),
            fp=results.get("fp", 0),
            fn=results.get("fn", 0),
            tn=results.get("tn", 0),
            output_dir=str(config.output_dir),
            runtime_seconds=runtime,
        )

    except Exception as e:
        LOGGER.exception("Error in run %s: %s", config.run_name, e)
        return RunResult(
            project=config.project,
            detector=config.detector,
            text_source=config.text_source,
            model=config.model,
            use_cot=config.use_cot,
            error=str(e),
            runtime_seconds=time.time() - start_time,
        )


def run_all(
    projects: Optional[List[str]] = None,
    detectors: Optional[List[str]] = None,
    text_source: str = DEFAULT_TEXT_SOURCE,
    model: str = DEFAULT_MODEL,
    use_cot: bool = False,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    cache_only: bool = False,
    api_delay: float = 1.0,
    max_workers: int = 4,
) -> List[RunResult]:
    """
    Run zero-shot classification for all project-detector combinations.

    Args:
        projects: Filter to specific projects (default: all)
        detectors: Filter to specific detectors (default: all available per project)
        text_source: Text source for classification
        model: LLM model to use
        use_cot: Enable chain-of-thought prompting
        output_dir: Base output directory (default: Output/zero_shot_runs/{timestamp})
        dry_run: Print what would be run without executing
        cache_only: Only use cached responses, no API calls
        api_delay: Delay between API calls for rate limiting
        max_workers: Max concurrent API workers for parallel processing

    Returns:
        List of RunResult objects
    """
    # Discover available datasets
    datasets = discover_datasets(projects=projects, detectors=detectors)

    if not datasets:
        LOGGER.error("No datasets found matching the specified filters")
        return []

    LOGGER.info("Discovered %d dataset(s):", len(datasets))
    for ds in datasets:
        LOGGER.info("  - %s/%s [prompt=%s]: %s", ds.project, ds.detector, ds.prompt_style, ds.data_path)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = DEFAULT_OUTPUT_BASE / timestamp
    else:
        output_base = output_dir

    # Generate run configurations
    run_configs = generate_run_configs(
        datasets=datasets,
        text_source=text_source,
        model=model,
        use_cot=use_cot,
        output_base=output_base,
    )

    LOGGER.info("\nGenerated %d run configuration(s):", len(run_configs))
    for cfg in run_configs:
        LOGGER.info(
            "  - %s/%s: text=%s, model=%s, cot=%s, prompt=%s",
            cfg.project, cfg.detector, cfg.text_source, cfg.model, cfg.use_cot, cfg.prompt_style
        )

    if dry_run:
        LOGGER.info("\n[DRY RUN] Would execute %d runs. Exiting.", len(run_configs))
        return []

    # Execute runs
    results = []
    output_base.mkdir(parents=True, exist_ok=True)

    for i, config in enumerate(run_configs, 1):
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Run %d/%d: %s", i, len(run_configs), config.run_name)
        LOGGER.info("=" * 60)

        result = run_zero_shot_classification(
            config=config,
            cache_only=cache_only,
            api_delay=api_delay,
            max_workers=max_workers,
        )
        results.append(result)

        if result.error:
            LOGGER.error("Run failed: %s", result.error)
        else:
            LOGGER.info(
                "Completed: acc=%.3f, prec=%.3f, rec=%.3f, f1=%.3f",
                result.accuracy, result.precision, result.recall, result.f1
            )

    # Save summary
    summary_file = output_base / "summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "text_source": text_source,
        "use_cot": use_cot,
        "total_runs": len(results),
        "successful_runs": sum(1 for r in results if r.error is None),
        "results": [asdict(r) for r in results],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("\nSummary saved to: %s", summary_file)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Project':<12} {'Detector':<8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Status':<10}")
    print("-" * 80)

    for r in results:
        status = "OK" if r.error is None else "ERROR"
        print(
            f"{r.project:<12} {r.detector:<8} "
            f"{r.accuracy:>7.3f} {r.precision:>7.3f} {r.recall:>7.3f} {r.f1:>7.3f} "
            f"{status:<10}"
        )

    print("-" * 80)
    print(f"Output directory: {output_base}")

    return results


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot LLM classification across all project-detector combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all combinations with defaults
    python -m src.casestudy.zero_shot_all

    # Run only for inkscape and shepard
    python -m src.casestudy.zero_shot_all --projects inkscape shepard

    # Run only ACDC detector
    python -m src.casestudy.zero_shot_all --detectors acdc

    # Use a different model
    python -m src.casestudy.zero_shot_all --model gpt-4o

    # Enable chain-of-thought prompting
    python -m src.casestudy.zero_shot_all --use-cot

    # With parallel workers and rate limiting
    python -m src.casestudy.zero_shot_all --max-workers 8 --api-delay 0.5

    # Dry run to see what would execute
    python -m src.casestudy.zero_shot_all --dry-run
        """
    )

    parser.add_argument(
        "--projects", "-p",
        nargs="+",
        choices=list(PROJECT_DETECTORS.keys()),
        help="Projects to run (default: all)"
    )

    parser.add_argument(
        "--detectors", "-d",
        nargs="+",
        choices=["acdc", "arcan"],
        help="Detectors to run (default: all available per project)"
    )

    parser.add_argument(
        "--text-source", "-t",
        default=DEFAULT_TEXT_SOURCE,
        choices=["description", "combined", "combined_unfiltered"],
        help=f"Text source for classification (default: {DEFAULT_TEXT_SOURCE})"
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Enable chain-of-thought prompting"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (default: Output/zero_shot_runs/{timestamp})"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )

    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached responses, no API calls"
    )

    parser.add_argument(
        "--api-delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds for rate limiting (default: 1.0)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max concurrent API workers for parallel processing (default: 4)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Zero-Shot LLM Classification - All Combinations")
    print("=" * 60)

    run_all(
        projects=args.projects,
        detectors=args.detectors,
        text_source=args.text_source,
        model=args.model,
        use_cot=args.use_cot,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        cache_only=args.cache_only,
        api_delay=args.api_delay,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
