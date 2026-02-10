"""
High-level orchestration for rebuilding smell-classification assets end-to-end.

This pipeline wires together the existing component mapper, smell-to-issue mapper,
classification dataset generator, and traditional ML runner so we can regenerate
artifacts in a clean output directory without manual glue code.
"""

from __future__ import annotations

import argparse
import json
import shutil

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.issues.config import Algorithm, ConfigManager
from src.mapper.component_mapper import ComponentMapper
from src.mapper.smell_to_issue_mapper import SmellEvolutionToIssueMapper
from src.preprocessing.gen_classification_data import (
    generate_clean_classification_data,
    generate_dirty_classification_data,
    print_summary,
    save_classification_data,
)
from src.preprocessing.detect_smells import (
    SmellDetectionError,
    run_smell_detection,
)
from src.smell.analyze_smell_evolution import run_evolution_analysis
from src.commit.batch_dependency_analyzer import BatchDependencyAnalyzer
from src.version.mr_version_merger import MRVersionMerger
from src.utils.system_paths import resolve_system_subdir


@dataclass
class PipelineConfig:
    """Configuration bundle for the classification preparation pipeline."""

    system_name: str
    algorithm: str
    issue_dir: Path
    evolution_file: Path
    output_root: Path
    text_source: str = "combined"
    max_workers: int = 4
    run_smell: bool = True
    run_ml: bool = True
    source_issue_dir: Optional[Path] = None
    prepared_issue_dir: Optional[Path] = None
    use_cv: bool = False
    cv_folds: int = 5
    cv_repeats: int = 2


def _ensure_directory(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def prepare_issue_data(config: PipelineConfig) -> Path:
    """
    Copy raw issue artifacts into a workspace directory and enrich them with MR version info.

    Returns:
        Path to the prepared issue directory that subsequent steps should consume.
    """
    print("\n=== STEP 0/5: Issue Dataset Preparation ===")

    source_dir = config.source_issue_dir or config.issue_dir
    if source_dir is None:
        raise ValueError("Issue data source directory is not configured.")
    if not source_dir.exists():
        raise FileNotFoundError(f"Issue data source directory not found: {source_dir}")

    work_dir = config.output_root / "work" / "issues" / config.system_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(source_dir, work_dir)
    print(f"Copied issue artifacts to workspace: {work_dir}")

    merger = MRVersionMerger(str(work_dir), config.system_name)
    if not merger.process_all_files():
        raise RuntimeError("MR version enrichment failed; check MRVersionMerger logs for details.")

    enriched_count = len(list(work_dir.glob("enriched_*.json")))
    if enriched_count == 0:
        print("Warning: MR version enrichment completed but no enriched_* files were produced.")
    else:
        print(f"Enriched merge request metadata for {enriched_count} batch files.")

    config.prepared_issue_dir = work_dir
    return work_dir


def align_clean_dataset_with_gold(
    generated_clean: list[dict],
    gold_path: Path,
) -> list[dict]:
    """Project generated clean dataset onto the gold label set if available."""
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    generated_index = {entry["id"]: entry for entry in generated_clean}

    aligned: list[dict] = []
    missing_ids: list[int] = []
    for record in gold_data:
        source = generated_index.get(record["id"])
        if source is None:
            missing_ids.append(record["id"])
            continue

        aligned_record = dict(record)
        if source:
            for key, value in source.items():
                aligned_record.setdefault(key, value)
        aligned.append(aligned_record)

    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} gold issues were not present in generated data. "
            f"Example IDs: {missing_ids[:10]}"
        )

    print(f"Aligned clean dataset to gold template with {len(aligned)} records.")
    return aligned


def build_dependency_checklist(config: PipelineConfig) -> Path:
    """Create dependency checklist JSON for the issue directory."""
    print("\n=== STEP 1/5: Dependency Checklist Generation ===")
    analyzer = BatchDependencyAnalyzer()
    commits = analyzer.process_directory(str(config.issue_dir), pattern="issues_all_diffs_batch_*.json")
    if not commits:
        raise RuntimeError("No commits found while building dependency checklist.")

    analyzer.generate_checklist(commits, only_with_changes=True)
    output_data = analyzer.get_output_data()
    if not output_data:
        raise RuntimeError("Checklist generation did not return any data.")

    checklist_dir = config.output_root / "data" / "issues" / config.system_name
    _ensure_directory(checklist_dir)
    checklist_path = checklist_dir / f"{config.system_name}_commit_dependency_checklist.json"

    with checklist_path.open("w", encoding="utf-8") as handle:
        import json

        json.dump(output_data, handle, indent=2)

    print(f"Dependency checklist saved to: {checklist_path}")
    return checklist_path


def run_component_mapping(config: PipelineConfig, data_smell_root: Path) -> Path:
    """Execute component mapping and persist output under the pipeline root."""
    print("\n=== STEP 2/5: Component Mapping ===")
    mapper = ComponentMapper(
        issue_dir=config.issue_dir,
        subject_system=config.system_name,
        data_root=data_smell_root,
        algorithm=config.algorithm,
    )

    results = mapper.process_all_issues(max_workers=config.max_workers)
    if "error" in results:
        raise RuntimeError(f"Component mapping failed: {results['error']}")

    # Reuse mapper's save logic but redirect to our pipeline output tree.
    original_data_root = mapper.data_root
    mapper.data_root = config.output_root / "data"
    mapping_path = mapper.save_results(results)
    mapper.print_summary(results)
    mapper.data_root = original_data_root

    # Relocate to data/mapping/<system>/ for consistent structure.
    target_dir = config.output_root / "data" / "mapping" / config.system_name
    _ensure_directory(target_dir)
    target_path = target_dir / mapping_path.name
    if target_path.exists():
        target_path.unlink()
    mapping_path.rename(target_path)

    print(f"\nComponent mapping saved to: {target_path}")
    return target_path


class _PipelineSmellMapper(SmellEvolutionToIssueMapper):
    """Smell mapper variant that prefers a provided component mapping file."""

    def __init__(self, config_manager: Optional[ConfigManager], mapping_override: Path):
        super().__init__(config_manager)
        self._mapping_override = mapping_override

    def find_issue_mapping_file(self, system_name: str, algorithm: str) -> Optional[Path]:
        """Use the injected mapping file when available."""
        if self._mapping_override and self._mapping_override.exists():
            return self._mapping_override
        return super().find_issue_mapping_file(system_name, algorithm)


def run_smell_mapping(
    config: PipelineConfig,
    mapping_path: Path,
) -> Path:
    """Generate smell-to-issue mappings using the freshly built component map."""
    print("\n=== STEP 3/5: Smell ↔ Issue Mapping ===")
    pipeline_config = ConfigManager()
    mapper = _PipelineSmellMapper(pipeline_config, mapping_path)

    result = mapper.map_evolution_to_issues(
        system_name=config.system_name,
        algorithm=config.algorithm,
        evolution_file=config.evolution_file,
    )

    if not result.success or not result.data:
        raise RuntimeError(f"Smell mapping failed: {result.message}")

    output_dir = config.output_root / "data" / "mapping" / config.system_name
    _ensure_directory(output_dir)
    smell_output_path = output_dir / f"evolution_to_issue_{config.system_name}_{config.algorithm}.json"
    mapper.save_results(result.data, smell_output_path)
    mapper.print_summary(result)

    print(f"\nSmell evolution mapping saved to: {smell_output_path}")
    return smell_output_path


def generate_classification_datasets(
    config: PipelineConfig,
    mapping_path: Path,
    smell_mapping_path: Path,
    dep_checklist_path: Path,
) -> Tuple[Path, Path]:
    """Create dirty and clean classification datasets under the pipeline root."""
    print("\n=== STEP 4/5: Classification Dataset Generation ===")
    classification_dir = config.output_root / "data" / "classification" / config.system_name
    _ensure_directory(classification_dir)

    dirty_data = generate_dirty_classification_data(
        str(smell_mapping_path),
        str(mapping_path),
        str(config.issue_dir),
        str(dep_checklist_path),
    )
    dirty_path = classification_dir / "classification_data_dirty.json"
    save_classification_data(dirty_data, str(dirty_path))
    print_summary(dirty_data, "Dirty Classification Data")

    clean_data = generate_clean_classification_data(
        str(smell_mapping_path),
        str(mapping_path),
        str(config.issue_dir),
        str(dep_checklist_path),
    )
    clean_raw_path = classification_dir / "classification_data_clean_raw.json"
    save_classification_data(clean_data, str(clean_raw_path))
    print_summary(clean_data, "Clean Classification Data (Raw)")

    gold_source_path = (
        Path("data")
        / "issues"
        / f"{config.system_name}_with_discussion"
        / "classification_clean_temp.json"
    )
    final_clean_path = classification_dir / "classification_data_clean.json"

    if gold_source_path.exists():
        aligned_data = align_clean_dataset_with_gold(clean_data, gold_source_path)
        final_clean_path.write_text(
            json.dumps(aligned_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Clean dataset aligned with gold reference and saved to: {final_clean_path}\n"
            f"Gold reference: {gold_source_path}"
        )
    else:
        print("Gold clean dataset not found; using raw clean data as final output.")
        final_clean_path.write_text(
            json.dumps(clean_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nClassification datasets written to: {classification_dir}")
    return dirty_path, final_clean_path


def run_ml_pipeline(
    config: PipelineConfig,
    clean_data_path: Path,
) -> Path:
    """Execute the traditional ML experiment on the generated clean dataset."""
    try:
        from src.classification.simple_ml import run_traditional_ml_experiments
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive path for missing deps
        raise ModuleNotFoundError(
            "sentence_transformers (or another ML dependency) is required to run the "
            "traditional ML baseline. Install the project's ML requirements or rerun "
            "the pipeline with --skip-ml."
        ) from exc

    print("\n=== STEP 5/5: Traditional ML Evaluation ===")
    reports_dir = config.output_root / "reports"
    _ensure_directory(reports_dir)
    results_path = reports_dir / f"{config.system_name}_{config.algorithm}_{config.text_source}_results.csv"

    run_traditional_ml_experiments(
        data_path=str(clean_data_path),
        output_file=str(results_path),
        text_source=config.text_source,
        use_cv=config.use_cv,
        cv_folds=config.cv_folds,
        cv_repeats=config.cv_repeats,
    )

    print(f"\nClassification results saved to: {results_path}")
    return results_path


def resolve_issue_dir(project_root: Path, system_name: str, override: Optional[str]) -> Path:
    """Choose a sensible default issue directory when none is supplied."""
    if override:
        return Path(override).expanduser().resolve()

    candidates = [
        project_root / "data" / "issues" / f"{system_name}_with_discussion",
        project_root / "data" / "issues" / system_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate issue directory for '{system_name}'. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def resolve_evolution_file(project_root: Path, system_name: str, algorithm: str, override: Optional[str]) -> Path:
    """Locate the smell evolution file."""
    if override:
        evolution_path = Path(override).expanduser().resolve()
    else:
        smell_root = resolve_system_subdir(project_root / "data" / "smells", system_name, create=True)
        evolution_path = smell_root / "evolution" / f"{system_name}_{algorithm}_evolution.json"

    return evolution_path


def default_output_root(project_root: Path, system_name: str, pipeline_label: str | None = None) -> Path:
    """Create a timestamped output directory that includes the system name and optional pipeline label."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [system_name]
    if pipeline_label:
        parts.append(pipeline_label)
    parts.append(timestamp)
    folder_name = "_".join(parts)
    root = project_root / "Output" / "pipeline_runs" / folder_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_mapper_data_root(project_root: Path) -> Path:
    """Find a data root that provides RSF cluster files for the component mapper."""
    candidates = [
        project_root / "data-smell",
        project_root / "data",
    ]
    for candidate in candidates:
        if (candidate / "clusters").exists():
            return candidate.resolve()
    candidate_list = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Unable to locate RSF clusters under any candidate data root: {candidate_list}"
    )


def ensure_smell_artifacts(config: PipelineConfig) -> Path:
    """Generate smell detection and evolution files when not already present."""
    smell_root = resolve_system_subdir(Path("data") / "smells", config.system_name, create=True)
    default_evolution = smell_root / "evolution" / f"{config.system_name}_{config.algorithm}_evolution.json"
    default_evolution.parent.mkdir(parents=True, exist_ok=True)

    target = config.evolution_file or default_evolution
    if target.exists():
        return target

    if not config.run_smell:
        raise FileNotFoundError(
            f"Smell evolution file missing and smell generation disabled: {target}"
        )

    try:
        run_smell_detection(config.system_name, [config.algorithm])
        run_evolution_analysis(config.system_name, config.algorithm)
    except SmellDetectionError as error:
        raise RuntimeError(f"Smell detection failed: {error}") from error

    if not target.exists() and default_evolution.exists():
        target = default_evolution

    if not target.exists():
        raise FileNotFoundError(
            f"Failed to generate smell evolution file for {config.system_name} "
            f"({config.algorithm}). Expected at {target}"
        )

    return target


def parse_args() -> PipelineConfig:
    """Parse CLI arguments into a PipelineConfig."""
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Rebuild smell classification datasets and ML results end-to-end.",
    )
    parser.add_argument("--system", required=True, help="Subject system name (e.g., inkscape)")
    parser.add_argument(
        "--algorithm",
        default=Algorithm.ACDC.value,
        choices=[alg.value for alg in Algorithm],
        help="Clustering algorithm to use for component mapping",
    )
    parser.add_argument("--issue-dir", help="Directory containing enriched issues and MR diffs",required=True)
    parser.add_argument("--evolution-file", help="Path to the smell evolution JSON",required=True)
    parser.add_argument(
        "--output-root",
        help="Root directory for regenerated artifacts (defaults to Output/pipeline_runs/<timestamp>)",
    )
    parser.add_argument(
        "--text-source",
        default="combined",
        choices=["description", "combined", "combined_unfiltered"],
        help="Text source for ML baseline: description, combined (filtered discussion), combined_unfiltered",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum parallel workers for component mapping",
    )
    parser.add_argument(
        "--skip-smell",
        action="store_true",
        help="Skip smell detection/evolution generation (expects evolution file to exist)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip the final traditional ML experiment (for environments without ML dependencies)",
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use repeated cross-validation instead of single train/test split for ML evaluation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds per repetition (default: 5)",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=2,
        help="Number of CV repetitions (default: 2, total trials = folds × repeats)",
    )

    args = parser.parse_args()

    system_name = args.system.strip().lower()
    algorithm = args.algorithm.strip().lower()

    issue_dir = resolve_issue_dir(project_root, system_name, args.issue_dir)
    evolution_file = resolve_evolution_file(project_root, system_name, algorithm, args.evolution_file)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(project_root, system_name, "arcade")
    )

    return PipelineConfig(
        system_name=system_name,
        algorithm=algorithm,
        issue_dir=issue_dir,
        evolution_file=evolution_file,
        output_root=output_root,
        text_source=args.text_source,
        max_workers=args.max_workers,
        run_smell=not args.skip_smell,
        run_ml=not args.skip_ml,
        source_issue_dir=issue_dir,
        use_cv=args.use_cv,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
    )


def main() -> None:
    """Entrypoint for CLI execution."""
    config = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_smell_root = resolve_mapper_data_root(project_root)

    try:
        prepared_issue_dir = prepare_issue_data(config)
    except Exception as error:
        raise SystemExit(f"Issue preparation failed: {error}") from error
    config.issue_dir = prepared_issue_dir

    try:
        config.evolution_file = ensure_smell_artifacts(config)
    except Exception as error:
        raise SystemExit(f"Smell preparation failed: {error}") from error

    print(f"Pipeline configuration:\n  System: {config.system_name}\n  Algorithm: {config.algorithm}\n"
          f"  Issue source: {config.source_issue_dir}\n  Prepared issue directory: {config.issue_dir}\n"
          f"  Evolution file: {config.evolution_file}\n"
          f"  Output root: {config.output_root}\n  Text source: {config.text_source}\n"
          f"  Data root (clusters): {data_smell_root}")

    dep_checklist_path = build_dependency_checklist(config)
    mapping_path = run_component_mapping(config, data_smell_root)
    smell_mapping_path = run_smell_mapping(config, mapping_path)
    _, clean_path = generate_classification_datasets(
        config,
        mapping_path,
        smell_mapping_path,
        dep_checklist_path,
    )

    if config.run_ml:
        run_ml_pipeline(config, clean_path)
    else:
        print("\nSkipping ML evaluation (requested via --skip-ml).")
        print(f"Clean dataset ready at: {clean_path}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
