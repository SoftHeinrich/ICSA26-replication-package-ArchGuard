#!/usr/bin/env python3
"""


Runs finetuning for:
  - inkscape × (description, combined) × (acdc)        = 2 runs
  - stackgres × (description, combined) × (acdc, arcan) = 4 runs
  - shepard × (description, combined) × (acdc, arcan)   = 4 runs
  Total: 10 runs
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import shared dataset configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.classification.dataset_config import (
    DATASET_BASE_DIR,
    DATASET_FOLDERS,
    PROJECT_DETECTORS,
    CLASSIFICATION_DATA_FILENAME,
)

LOGGER = logging.getLogger(__name__)

# Text sources to evaluate (default to description only)
TEXT_SOURCES = ["description"]


@dataclass
class RunConfig:
    """Configuration for a single finetuning run."""
    project: str
    detector: str
    text_source: str
    data_path: Path
    run_name: str


@dataclass
class RunResult:
    """Results from a single finetuning run."""
    project: str
    detector: str
    text_source: str
    run_name: str
    issue_f1_mean: float
    issue_f1_std: float
    chunk_f1_mean: float
    chunk_f1_std: float
    n_folds: int
    run_dir: str
    success: bool
    error: Optional[str] = None


def discover_dataset_path(project: str, detector: str) -> Optional[Path]:
    """Discover the classification data path for a given project and detector."""
    key = (project, detector)
    if key not in DATASET_FOLDERS:
        LOGGER.warning("No dataset folder configured for %s/%s", project, detector)
        return None

    folder_name = DATASET_FOLDERS[key]

    # Get project directory name
    from classification.dataset_config import PROJECT_DIR_NAMES
    project_dir = PROJECT_DIR_NAMES.get(project, project)

    # Path structure: base_dir / folder_name / data / classification / project_dir / filename
    data_path = DATASET_BASE_DIR / folder_name / "data" / "classification" / project_dir / CLASSIFICATION_DATA_FILENAME

    if not data_path.exists():
        LOGGER.warning("Classification data not found: %s", data_path)
        return None

    return data_path


def build_run_configs(
    projects: Optional[list[str]] = None,
    text_sources: Optional[list[str]] = None,
    detectors: Optional[list[str]] = None,
) -> list[RunConfig]:
    """Build all run configurations based on filters."""
    configs = []
    target_projects = projects or list(PROJECT_DETECTORS.keys())
    target_sources = text_sources or TEXT_SOURCES

    for project in target_projects:
        if project not in PROJECT_DETECTORS:
            LOGGER.warning("Unknown project: %s", project)
            continue

        available_detectors = PROJECT_DETECTORS[project]
        target_detectors = detectors or available_detectors

        for detector in target_detectors:
            if detector not in available_detectors:
                continue

            data_path = discover_dataset_path(project, detector)
            if data_path is None:
                continue

            for text_source in target_sources:
                run_name = f"{project}_{detector}_{text_source}"
                configs.append(RunConfig(
                    project=project,
                    detector=detector,
                    text_source=text_source,
                    data_path=data_path,
                    run_name=run_name,
                ))

    return configs


def run_single_finetune(
    config: RunConfig,
    output_base: Path,
    model_name: str = "roberta-base",
    chunk_words: int = 256,
    max_length: int = 512,
    batch_size: int = 8,
    epochs: int = 15,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    early_stopping: bool = False,
    early_stopping_patience: int = 5,
    warmup_ratio: float = 0.06,
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 42,
    aggregation: str = "vote",
) -> RunResult:
    """
    Run a single finetuning configuration by calling roberta_finetune.py.

    This is a wrapper that invokes the actual training script as a subprocess.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Starting: %s", config.run_name)
    LOGGER.info("  Project: %s, Detector: %s, Text: %s",
                config.project, config.detector, config.text_source)
    LOGGER.info("  Data: %s", config.data_path)
    LOGGER.info("=" * 60)

    try:
        # Construct command to call roberta_finetune.py
        cmd = [
            sys.executable, "-m", "classification.roberta_finetune",
            str(config.data_path),
            "--text-source", config.text_source,
            "--model", model_name,
            "--chunk-words", str(chunk_words),
            "--max-length", str(max_length),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--weight-decay", str(weight_decay),
            "--warmup-ratio", str(warmup_ratio),
            "--n-splits", str(n_splits),
            "--n-repeats", str(n_repeats),
            "--seed", str(seed),
            "--issue-aggregation", aggregation,
            "--output-dir", str(output_base),
        ]

        # Add early stopping flags if enabled
        if early_stopping:
            cmd.extend(["--early-stopping", "--early-stopping-patience", str(early_stopping_patience)])

        LOGGER.info("Executing: %s", " ".join(cmd))

        # Run the subprocess
        result = subprocess.run(
            cmd,
            capture_output=False,  # Stream output to console
            text=True,
            check=True,
        )

        # Parse metrics from the output directory
        # The script creates: output_base / detector_folder / <dataset_stem>_<text_source>_cw<chunk_words>_<aggregation>_<model>/metrics.json
        # Detector folder is derived from path structure (matching roberta_finetune.py behavior)
        # Path: .../shepard_acdc/data/classification/dlr-shepard-shepard/classification_data_clean.json
        detector_folder = config.data_path.parent.parent.parent.parent.name
        if not detector_folder or detector_folder == "supervised_ml_all":
            detector_folder = config.data_path.parent.name or config.data_path.stem

        dataset_stem = config.data_path.stem
        safe_model = model_name.replace("/", "_").replace(":", "_")
        run_dir_name = f"{dataset_stem}_{config.text_source}_cw{chunk_words}_{aggregation}_{safe_model}"
        metrics_file = output_base / detector_folder / run_dir_name / "metrics.json"

        if not metrics_file.exists():
            LOGGER.warning("Metrics file not found: %s", metrics_file)
            return RunResult(
                project=config.project,
                detector=config.detector,
                text_source=config.text_source,
                run_name=config.run_name,
                issue_f1_mean=0.0,
                issue_f1_std=0.0,
                chunk_f1_mean=0.0,
                chunk_f1_std=0.0,
                n_folds=n_splits * n_repeats,
                run_dir=str(output_base / detector_folder / run_dir_name),
                success=False,
                error="Metrics file not found after training",
            )

        # Load metrics
        with metrics_file.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        return RunResult(
            project=config.project,
            detector=config.detector,
            text_source=config.text_source,
            run_name=config.run_name,
            issue_f1_mean=metrics.get("f1_mean", 0.0),  # Issue-level F1 for selected aggregation
            issue_f1_std=metrics.get("f1_std", 0.0),
            chunk_f1_mean=metrics.get("chunk_f1_mean", 0.0),
            chunk_f1_std=metrics.get("chunk_f1_std", 0.0),
            n_folds=metrics.get("total_folds", n_splits * n_repeats),
            run_dir=str(output_base / detector_folder / run_dir_name),
            success=True,
            error=None,
        )

    except subprocess.CalledProcessError as e:
        LOGGER.exception("Training failed for %s: %s", config.run_name, e)
        return RunResult(
            project=config.project,
            detector=config.detector,
            text_source=config.text_source,
            run_name=config.run_name,
            issue_f1_mean=0.0,
            issue_f1_std=0.0,
            chunk_f1_mean=0.0,
            chunk_f1_std=0.0,
            n_folds=n_splits * n_repeats,
            run_dir="",
            success=False,
            error=f"Subprocess failed: {e}",
        )
    except Exception as e:
        LOGGER.exception("Error in run %s: %s", config.run_name, e)
        return RunResult(
            project=config.project,
            detector=config.detector,
            text_source=config.text_source,
            run_name=config.run_name,
            issue_f1_mean=0.0,
            issue_f1_std=0.0,
            chunk_f1_mean=0.0,
            chunk_f1_std=0.0,
            n_folds=n_splits * n_repeats,
            run_dir="",
            success=False,
            error=str(e),
        )


def save_summary(results: list[RunResult], output_path: Path) -> None:
    """Save results summary to CSV and JSON."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "project": r.project,
            "detector": r.detector,
            "text_source": r.text_source,
            "run_name": r.run_name,
            "issue_f1_mean": r.issue_f1_mean,
            "issue_f1_std": r.issue_f1_std,
            "chunk_f1_mean": r.chunk_f1_mean,
            "chunk_f1_std": r.chunk_f1_std,
            "n_folds": r.n_folds,
            "run_dir": r.run_dir,
            "success": r.success,
            "error": r.error,
        })

    df = pd.DataFrame(rows)
    csv_path = output_path / "finetune_results_summary.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("Summary CSV saved to %s", csv_path)

    json_path = output_path / "finetune_results_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    LOGGER.info("Summary JSON saved to %s", json_path)

    print("\n" + "=" * 80)
    print("FINETUNING RESULTS SUMMARY")
    print("=" * 80)
    print(df[["project", "detector", "text_source", "issue_f1_mean", "issue_f1_std", "success"]].to_string())
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch RoBERTa fine-tuning wrapper across project/text-source/detector combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projects", nargs="+", default=None,
                        choices=["inkscape", "stackgres", "shepard"],
                        help="Filter to specific projects.")
    parser.add_argument("--text-sources", nargs="+", default=None,
                        choices=["description", "combined"],
                        help="Filter to specific text sources.")
    parser.add_argument("--detectors", nargs="+", default=None,
                        choices=["acdc", "arcan"],
                        help="Filter to specific detectors.")
    parser.add_argument("--output-dir", type=str, default="Output/hf_roberta_batch",
                        help="Base output directory for all runs.")
    parser.add_argument("--model", type=str, default="roberta-base",
                        help="Hugging Face model name.")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate for AdamW optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW regularization.")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping based on validation loss (disabled by default).")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience (default: 5).")
    parser.add_argument("--chunk-words", type=int, default=256,
                        help="Max words per chunk.")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds.")
    parser.add_argument("--n-repeats", type=int, default=2,
                        help="Number of CV repeats.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configurations without running.")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    # Build configurations
    configs = build_run_configs(
        projects=args.projects,
        text_sources=args.text_sources,
        detectors=args.detectors,
    )

    if not configs:
        LOGGER.error("No valid configurations found!")
        return

    LOGGER.info("Found %d configurations to run:", len(configs))
    for i, config in enumerate(configs, 1):
        LOGGER.info("  %d. %s (data: %s)", i, config.run_name, config.data_path)

    if args.dry_run:
        print("\nDRY RUN - would execute the following:")
        for config in configs:
            print(f"  - {config.run_name}: {config.data_path}")
        return

    # Setup output
    output_base = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_base / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", output_path)

    # Run all configurations
    results = []
    for i, config in enumerate(configs, 1):
        LOGGER.info("\n[%d/%d] Running: %s", i, len(configs), config.run_name)
        result = run_single_finetune(
            config=config,
            output_base=output_path,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            chunk_words=args.chunk_words,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
        )
        results.append(result)
        save_summary(results, output_path)

    LOGGER.info("\nAll runs complete!")
    LOGGER.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
