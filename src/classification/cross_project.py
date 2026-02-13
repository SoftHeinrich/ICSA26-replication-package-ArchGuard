"""
Train on one or more datasets and validate on a different dataset.

This entrypoint reuses the traditional ML stack (TF-IDF, SBERT chunking, OpenAI
embeddings when present) and runs every available encoding/model/imbalance
combination by default. Training data can be provided as multiple JSON files or
pipeline run directories that contain classification outputs.
"""

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .simple_ml import (
    TEXT_SOURCE_COMBINED_FILTERED,
    VALID_TEXT_SOURCES,
    _clear_cuda_cache,
    _find_openai_embeddings_path,
    _load_sentence_transformer,
    _select_torch_device,
    apply_imbalance_strategy,
    calculate_top_k_positive_count,
    clean_text_for_tfidf,
    compute_balanced_sample_weights,
    configure_model_for_strategy,
    extract_texts_from_data,
    get_base_models,
    get_encoding_methods,
    get_imbalance_strategies,
    load_precomputed_embeddings,
    normalize_text_source,
    sbert_encode_with_chunking, TEXT_SOURCE_DESCRIPTION,
)

# Import centralized dataset configuration
from .dataset_config import (
    DATASET_BASE_DIR,
    DATASET_FOLDERS,
    PROJECT_DIR_NAMES,
    CLASSIFICATION_DATA_FILENAME,
)


CLASSIFICATION_FILENAMES = [
    "classification_data_clean.json",
    "classification_data_clean_raw.json",
    "classification_clean_temp.json",
    "classification_data_dirty.json",
]

# Leave-one-out combinations by detector
LEAVE_ONE_OUT_ACDC = [
    # (train_projects, eval_project)
    (["inkscape", "stackgres"], "shepard"),
    (["inkscape", "shepard"], "stackgres"),
    (["stackgres", "shepard"], "inkscape"),
]

LEAVE_ONE_OUT_ARCAN = [
    # (train_projects, eval_project)
    (["stackgres"], "shepard"),
    (["shepard"], "stackgres"),
]


VALID_DETECTORS = {"acdc", "arcan"}


@dataclass
class DatasetBundle:
    """Container for a loaded classification dataset."""

    path: Path
    texts: list[str]
    labels: list[str]
    detector: str  # "acdc" or "arcan"

    @property
    def dataset_id(self) -> str:
        return self.path.parent.name or self.path.stem


def _detect_detector(data_path: Path) -> str:
    """
    Detect whether a dataset was generated using ACDC or Arcan.

    Looks for smell mapping files in the pipeline run's data/mapping directory.
    - ACDC: evolution_to_issue_{system}_acdc.json
    - Arcan: evolution_to_issue_{system}_arcan.json or mapping_{system}_arcan.json
    """
    # Walk up to find pipeline run root (contains data/ directory)
    current = data_path.parent
    for _ in range(5):  # Limit search depth
        mapping_dir = current / "data" / "mapping"
        if mapping_dir.exists():
            # Check for Arcan-specific files first (more specific)
            arcan_files = list(mapping_dir.rglob("*_arcan.json"))
            if arcan_files:
                return "arcan"
            # Check for ACDC files
            acdc_files = list(mapping_dir.rglob("*_acdc.json"))
            if acdc_files:
                return "acdc"
        current = current.parent

    # Fallback: check path components for detector hints
    path_str = str(data_path).lower()
    if "_arcan" in path_str or "/arcan" in path_str:
        return "arcan"

    # Default to ACDC (original detector)
    return "acdc"


def _load_dataframe(data_path: Path) -> pd.DataFrame:
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    df = pd.DataFrame(payload)
    if "label" not in df.columns:
        raise ValueError(f"Dataset {data_path} is missing required 'label' column.")
    return df


def _resolve_classification_data_path(raw: str) -> Path:
    """Resolve a classification dataset path, accepting pipeline run directories."""
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return path

    if path.is_dir():
        for name in CLASSIFICATION_FILENAMES:
            matches = sorted(path.rglob(name))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"No classification data found under {path}. "
            f"Tried: {', '.join(CLASSIFICATION_FILENAMES)}"
        )

    raise FileNotFoundError(f"Path not found: {raw}")


def _load_bundle(path: Path, text_source: str) -> DatasetBundle:
    df = _load_dataframe(path)
    texts = extract_texts_from_data(df, text_source=text_source)
    labels = df["label"].tolist()
    detector = _detect_detector(path)
    return DatasetBundle(path=path, texts=texts, labels=labels, detector=detector)


def _default_output_path(train_bundles: Sequence[DatasetBundle], eval_bundle: DatasetBundle, text_source: str) -> Path:
    train_slug = "+".join(bundle.dataset_id for bundle in train_bundles)
    eval_slug = eval_bundle.dataset_id
    detector = train_bundles[0].detector  # All bundles have same detector (enforced)
    base = f"cross_project_{detector}_{train_slug}_to_{eval_slug}_{text_source}.csv"
    return DATASET_BASE_DIR.parents[2] / base


def _maybe_collect_openai_paths(paths: Iterable[Path], text_source: str) -> dict[Path, Path] | None:
    """Return map of dataset path -> embeddings path when all exist."""
    if os.getenv("DISABLE_OPENAI_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes"}:
        return None

    embeddings_paths: dict[Path, Path] = {}
    for dataset_path in paths:
        try:
            embeddings_paths[dataset_path] = _find_openai_embeddings_path(dataset_path, text_source)
        except FileNotFoundError as exc:
            print(f"Skipping OpenAI embeddings (missing for {dataset_path}): {exc}")
            return None
    return embeddings_paths


def _encode_for_method(
    method: str,
    train_bundles: Sequence[DatasetBundle],
    eval_bundle: DatasetBundle,
    text_source: str,
    openai_paths: dict[Path, Path] | None,
):
    """Encode train and eval texts with the requested method."""
    train_texts = [text for bundle in train_bundles for text in bundle.texts]
    eval_texts = list(eval_bundle.texts)

    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("Cleaning texts for TF-IDF encoding...")
        cleaned_train_texts = [clean_text_for_tfidf(text) for text in train_texts]
        cleaned_eval_texts = [clean_text_for_tfidf(text) for text in eval_texts]
        vectorizer = TfidfVectorizer(max_features=50000, stop_words="english", ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(cleaned_train_texts)
        X_eval = vectorizer.transform(cleaned_eval_texts)
        return X_train, X_eval

    if method in {"sbert_avg", "sbert_max", "sbert_tfidf_weighted"}:
        strategy_map = {"sbert_avg": "avg", "sbert_max": "max", "sbert_tfidf_weighted": "tfidf_weighted"}
        chunk_strategy = strategy_map[method]
        device = _select_torch_device()
        SentenceTransformer = _load_sentence_transformer()
        model = SentenceTransformer("paraphrase-mpnet-base-v2", device=device)

        tfidf_vectorizer = None
        if chunk_strategy == "tfidf_weighted":
            from sklearn.feature_extraction.text import TfidfVectorizer

            print("Cleaning texts for TF-IDF weighting...")
            cleaned_train_texts = [clean_text_for_tfidf(text) for text in train_texts]
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf_vectorizer.fit(cleaned_train_texts)

        X_train = sbert_encode_with_chunking(
            train_texts,
            model,
            chunk_strategy=chunk_strategy,
            max_words=512,
            tfidf_vectorizer=tfidf_vectorizer,
            batch_size=32,
            show_progress=True,
        )
        X_eval = sbert_encode_with_chunking(
            eval_texts,
            model,
            chunk_strategy=chunk_strategy,
            max_words=512,
            tfidf_vectorizer=tfidf_vectorizer,
            batch_size=32,
            show_progress=False,
        )
        del model
        import gc
        gc.collect()
        _clear_cuda_cache()
        return X_train, X_eval

    if method == "openai":
        if openai_paths is None:
            raise ValueError("OpenAI embeddings requested but not available for all datasets.")

        train_arrays: List[np.ndarray] = []
        for bundle in train_bundles:
            embeddings, _ = load_precomputed_embeddings(openai_paths[bundle.path])
            if embeddings.shape[0] != len(bundle.texts):
                raise ValueError(
                    f"Train embeddings count ({embeddings.shape[0]}) does not match samples ({len(bundle.texts)}) "
                    f"for {bundle.path}"
                )
            train_arrays.append(embeddings)
        X_train = np.vstack(train_arrays)

        eval_embeddings, _ = load_precomputed_embeddings(openai_paths[eval_bundle.path])
        if eval_embeddings.shape[0] != len(eval_bundle.texts):
            raise ValueError(
                f"Eval embeddings count ({eval_embeddings.shape[0]}) does not match samples ({len(eval_bundle.texts)}) "
                f"for {eval_bundle.path}"
            )

        train_dim = X_train.shape[1]
        if eval_embeddings.shape[1] != train_dim:
            raise ValueError(
                f"OpenAI embedding dimensions differ between train ({train_dim}) and eval ({eval_embeddings.shape[1]}). "
                "Regenerate embeddings with a consistent model."
            )
        return X_train, eval_embeddings

    raise ValueError(f"Unknown encoding method: {method}")


def _compute_random_baseline(y_train: np.ndarray, y_eval: np.ndarray, n_iterations: int = 100):
    """
    Compute random baseline metrics using stratified random prediction.

    Uses DummyClassifier with 'stratified' strategy - predicts based on
    training label distribution.

    Args:
        y_train: Training labels (used to learn class distribution)
        y_eval: Evaluation labels (used for scoring)
        n_iterations: Number of iterations to average over (reduces variance)

    Returns:
        dict with baseline metrics and predictions from last iteration
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Run multiple iterations to get stable estimates
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    for seed in range(n_iterations):
        baseline = DummyClassifier(strategy='stratified', random_state=seed)
        baseline.fit(np.zeros((len(y_train), 1)), y_train)  # X doesn't matter for stratified
        y_pred = baseline.predict(np.zeros((len(y_eval), 1)))

        accuracies.append(accuracy_score(y_eval, y_pred))
        p, r, f, _ = precision_recall_fscore_support(y_eval, y_pred, labels=[1], zero_division=0)
        precisions.append(float(p[0]))
        recalls.append(float(r[0]))
        f1_scores.append(float(f[0]))

    # Get predictions from a fixed seed for McNemar test
    baseline_fixed = DummyClassifier(strategy='stratified', random_state=42)
    baseline_fixed.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_baseline = baseline_fixed.predict(np.zeros((len(y_eval), 1)))

    return {
        "baseline_accuracy": float(np.mean(accuracies)),
        "baseline_accuracy_std": float(np.std(accuracies)),
        "baseline_precision": float(np.mean(precisions)),
        "baseline_recall": float(np.mean(recalls)),
        "baseline_f1": float(np.mean(f1_scores)),
        "baseline_f1_std": float(np.std(f1_scores)),
        "baseline_predictions": y_pred_baseline,
    }


def _compute_mcnemar_pvalue(y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_baseline: np.ndarray) -> float:
    """
    Compute McNemar's test p-value comparing model vs baseline.

    McNemar's test compares two classifiers on the same test set by looking at
    the contingency table of correct/incorrect predictions.

    Args:
        y_true: Ground truth labels
        y_pred_model: Model predictions
        y_pred_baseline: Baseline predictions

    Returns:
        p-value (lower = model is significantly better than baseline)
    """
    from statsmodels.stats.contingency_tables import mcnemar

    # Build contingency table
    model_correct = y_pred_model == y_true
    baseline_correct = y_pred_baseline == y_true

    # Contingency table:
    #                    Baseline Correct  Baseline Wrong
    # Model Correct          n00              n01
    # Model Wrong            n10              n11
    n00 = np.sum(model_correct & baseline_correct)
    n01 = np.sum(model_correct & ~baseline_correct)
    n10 = np.sum(~model_correct & baseline_correct)
    n11 = np.sum(~model_correct & ~baseline_correct)

    table = [[n00, n01], [n10, n11]]

    # McNemar's test (exact=True for small samples, exact=False uses chi-square)
    # Use exact test when discordant pairs < 25
    discordant = n01 + n10
    if discordant == 0:
        return 1.0

    try:
        result = mcnemar(table, exact=(discordant < 25))
        return float(result.pvalue)
    except Exception:
        return 1.0


def _evaluate_model(model, X_eval, y_eval, baseline_metrics: dict | None = None):
    """
    Evaluate model with metrics consistent with simple_ml.py (binary positive class focus).

    Args:
        model: Trained classifier
        X_eval: Evaluation features
        y_eval: Evaluation labels
        baseline_metrics: Dict from _compute_random_baseline() for comparison

    Returns:
        Dict with model metrics using _mean/_std naming convention (consistent with simple_ml.py)
    """
    y_pred = model.predict(X_eval)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_eval, y_pred)

    # Use labels=[1] for binary positive class metrics (consistent with simple_ml.py)
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_eval, y_pred, labels=[1], zero_division=0
    )
    precision = float(precision_arr[0])
    recall = float(recall_arr[0])
    f1 = float(f1_arr[0])

    # Baseline comparison metrics
    baseline_comparison = {}
    if baseline_metrics is not None:
        baseline_f1 = baseline_metrics["baseline_f1"]
        f1_improvement = f1 - baseline_f1

        # McNemar's test for statistical significance
        # Only consider significant if BOTH improvement > 0 AND p < 0.05
        y_pred_baseline = baseline_metrics["baseline_predictions"]
        mcnemar_pvalue = _compute_mcnemar_pvalue(y_eval, y_pred, y_pred_baseline)
        significant = (f1_improvement > 0) and (mcnemar_pvalue < 0.05)

        baseline_comparison = {
            "random_f1_mean": baseline_f1,
            "random_f1_std": baseline_metrics["baseline_f1_std"],
            "f1_improvement": f1_improvement,
            "mcnemar_pvalue": mcnemar_pvalue,
            "significantly_better_than_random": significant,
        }

    # Use _mean/_std naming convention to match simple_ml.py schema
    # For single-run results, std is 0 (no variance across trials)
    return {
        "cv_folds": 0,
        "cv_repeats": 0,
        "total_trials": 1,
        "accuracy_mean": float(accuracy),
        "accuracy_std": 0.0,
        "precision_mean": precision,
        "precision_std": 0.0,
        "recall_mean": recall,
        "recall_std": 0.0,
        "f1_mean": f1,
        "f1_std": 0.0,
        **baseline_comparison,
    }


def _create_strategy_with_seed(strategy_name: str, seed: int):
    """Create a fresh imbalance strategy object with specified random seed."""
    try:
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
        from imblearn.combine import SMOTEENN, SMOTETomek
        IMBLEARN_AVAILABLE = True
    except ImportError:
        IMBLEARN_AVAILABLE = False
        return None

    if not IMBLEARN_AVAILABLE:
        return None

    strategy_map = {
        'random_undersample': lambda s: RandomUnderSampler(random_state=s),
        'smote': lambda s: SMOTE(random_state=s),
        'borderline_smote': lambda s: BorderlineSMOTE(random_state=s),
        'adasyn': lambda s: ADASYN(random_state=s),
    }

    if strategy_name in strategy_map:
        return strategy_map[strategy_name](seed)
    return None


def _evaluate_cross_project_repeated(
    model,
    model_name: str,
    strategy_name: str,
    strategy,
    X_train,
    y_train,
    X_eval,
    y_eval,
    n_folds: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
):
    """
    Evaluate cross-project model using repeated runs with different random seeds.

    Since train/test split is fixed (different projects), we repeat with different
    random seeds for model initialization and imbalance resampling to get variance estimates.

    Args:
        model: sklearn-compatible model
        model_name: Name of the model
        strategy_name: Name of imbalance strategy
        strategy: Strategy object
        X_train: Training features (encoded)
        y_train: Training labels
        X_eval: Evaluation features (encoded)
        y_eval: Evaluation labels
        n_folds: Number of trials per repetition (default: 5)
        n_repeats: Number of repetitions (default: 2)
        random_state: Base random seed

    Returns:
        Dict with repeated CV results including significance testing
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from scipy.stats import wilcoxon

    total_trials = n_folds * n_repeats

    # Storage for all metrics per trial
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    random_f1_scores = []

    print(f"  Strategy: {strategy_name} - Running {n_repeats} × {n_folds}-fold ({total_trials} trials)")

    for trial in range(total_trials):
        seed = random_state + trial

        # Clone and configure model with trial-specific seed
        trial_model = clone(model)
        trial_model = configure_model_for_strategy(trial_model, model_name, strategy_name)

        # Set random state on model if available
        if hasattr(trial_model, 'random_state'):
            trial_model.random_state = seed

        # Create fresh strategy with trial-specific seed
        trial_strategy = _create_strategy_with_seed(strategy_name, seed)

        # Apply imbalance strategy
        X_train_proc, y_train_proc = apply_imbalance_strategy(
            X_train, y_train, strategy_name, trial_strategy if trial_strategy else strategy
        )

        # Compute sample weights if needed
        sample_weight = None
        if strategy_name == 'class_weight':
            sample_weight = compute_balanced_sample_weights(y_train_proc)

        fit_kwargs = {}
        if sample_weight is not None and 'sample_weight' in inspect.signature(trial_model.fit).parameters:
            fit_kwargs['sample_weight'] = sample_weight

        # Scale if needed
        if model_name in {"logistic_regression", "svm", "knn", "feedforward_nn"}:
            with_mean = not hasattr(X_train_proc, "toarray")
            scaler = StandardScaler(with_mean=with_mean)
            X_train_scaled = scaler.fit_transform(X_train_proc)
            X_eval_scaled = scaler.transform(X_eval)
            trial_model.fit(X_train_scaled, y_train_proc, **fit_kwargs)
            y_pred = trial_model.predict(X_eval_scaled)
        else:
            trial_model.fit(X_train_proc, y_train_proc, **fit_kwargs)
            y_pred = trial_model.predict(X_eval)

        # Compute metrics
        accuracy = accuracy_score(y_eval, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_eval, y_pred, labels=[1], zero_division=0)
        precision = float(precision[0]) if hasattr(precision, '__len__') else float(precision)
        recall = float(recall[0]) if hasattr(recall, '__len__') else float(recall)
        f1 = float(f1[0]) if hasattr(f1, '__len__') else float(f1)

        accuracy_scores.append(float(accuracy))
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Random baseline with same seed
        random_baseline = DummyClassifier(strategy='stratified', random_state=seed)
        random_baseline.fit(np.zeros((len(y_train), 1)), y_train)
        y_pred_random = random_baseline.predict(np.zeros((len(y_eval), 1)))
        _, _, random_f1, _ = precision_recall_fscore_support(y_eval, y_pred_random, labels=[1], zero_division=0)
        random_f1 = float(random_f1[0]) if hasattr(random_f1, '__len__') else float(random_f1)
        random_f1_scores.append(random_f1)

        if (trial + 1) % max(1, total_trials // 5) == 0:
            print(f"    Trial {trial + 1}/{total_trials}: F1={f1:.3f} (random={random_f1:.3f})")

    # Aggregate results
    accuracy_array = np.array(accuracy_scores)
    precision_array = np.array(precision_scores)
    recall_array = np.array(recall_scores)
    f1_array = np.array(f1_scores)
    random_f1_array = np.array(random_f1_scores)

    accuracy_mean = float(np.mean(accuracy_array))
    accuracy_std = float(np.std(accuracy_array))
    precision_mean = float(np.mean(precision_array))
    precision_std = float(np.std(precision_array))
    recall_mean = float(np.mean(recall_array))
    recall_std = float(np.std(recall_array))
    f1_mean = float(np.mean(f1_array))
    f1_std = float(np.std(f1_array))
    random_f1_mean = float(np.mean(random_f1_array))
    random_f1_std = float(np.std(random_f1_array))
    f1_improvement = f1_mean - random_f1_mean

    # Statistical significance testing
    # Only consider significant if BOTH improvement > 0 AND p < 0.05
    if np.allclose(f1_array, random_f1_array):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = wilcoxon(f1_array, random_f1_array, alternative='greater')
        except ValueError:
            wilcoxon_p = 1.0

    significant = (f1_improvement > 0) and (wilcoxon_p < 0.05)

    # Compute effect size (Cohen's d)
    mean_diff = f1_mean - random_f1_mean
    pooled_std = np.sqrt((np.std(f1_array, ddof=1)**2 + np.std(random_f1_array, ddof=1)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    print(f"    Results: Acc={accuracy_mean:.3f}±{accuracy_std:.3f}, P={precision_mean:.3f}±{precision_std:.3f}, "
          f"R={recall_mean:.3f}±{recall_std:.3f}, F1={f1_mean:.3f}±{f1_std:.3f}")
    print(f"    Random F1: {random_f1_mean:.3f}±{random_f1_std:.3f}, Improvement: +{f1_improvement:.3f}, "
          f"p={wilcoxon_p:.4f}, Cohen's d={cohens_d:.3f} {'✓' if significant else '✗'}")

    return {
        'cv_folds': n_folds,
        'cv_repeats': n_repeats,
        'total_trials': total_trials,
        'accuracy_mean': accuracy_mean,
        'accuracy_std': accuracy_std,
        'precision_mean': precision_mean,
        'precision_std': precision_std,
        'recall_mean': recall_mean,
        'recall_std': recall_std,
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'random_f1_mean': random_f1_mean,
        'random_f1_std': random_f1_std,
        'f1_improvement': f1_improvement,
        'wilcoxon_p_value': float(wilcoxon_p),
        'cohens_d': float(cohens_d),
        'significantly_better_than_random': bool(significant),
    }


def run_cross_project_classification(
    train_data: Sequence[Path],
    eval_data: Path,
    text_source: str = TEXT_SOURCE_COMBINED_FILTERED,
    encodings: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
    output_path: Path | None = None,
    use_cv: bool = False,
    cv_folds: int = 5,
    cv_repeats: int = 2,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Train on one or more datasets and evaluate on another.

    When no encodings/models/strategies are provided, all available combinations
    are executed.

    Args:
        train_data: Paths to training datasets
        eval_data: Path to evaluation dataset
        text_source: Text fields to use
        encodings: Encoding methods to use
        models: ML models to use
        strategies: Imbalance strategies to use
        output_path: Output file path
        use_cv: Use repeated evaluation with significance testing (default: False)
        cv_folds: Number of trials per repetition (default: 5)
        cv_repeats: Number of repetitions (default: 2, total trials = folds × repeats)
        dry_run: Preview mode - show what would be run without executing (default: False)
    """
    resolved_text_source = normalize_text_source(text_source)
    train_bundles = [_load_bundle(path, resolved_text_source) for path in train_data]
    eval_bundle = _load_bundle(eval_data, resolved_text_source)

    # Enforce same detector for all datasets (ACDC with ACDC, Arcan with Arcan)
    all_detectors = {bundle.detector for bundle in train_bundles} | {eval_bundle.detector}
    if len(all_detectors) > 1:
        detector_info = [f"{b.dataset_id}={b.detector}" for b in train_bundles]
        detector_info.append(f"{eval_bundle.dataset_id}={eval_bundle.detector}")
        raise ValueError(
            f"Cross-project evaluation requires all datasets to use the same detector. "
            f"Found mixed detectors: {', '.join(detector_info)}. "
            f"ACDC-based datasets must be evaluated against ACDC-based datasets, "
            f"and Arcan-based datasets against Arcan-based datasets."
        )
    detector = train_bundles[0].detector

    print(f"Training datasets ({len(train_bundles)}):")
    for bundle in train_bundles:
        print(f"  - {bundle.dataset_id}: {bundle.path} ({len(bundle.labels)} samples)")
    print(f"Eval dataset: {eval_bundle.dataset_id}: {eval_bundle.path} ({len(eval_bundle.labels)} samples)")
    print(f"Detector: {detector.upper()}")
    print(f"Text source: {resolved_text_source}")
    if use_cv:
        total_trials = cv_folds * cv_repeats
        print(f"Evaluation: Repeated CV ({cv_repeats} × {cv_folds}-fold = {total_trials} trials)")
    else:
        print("Evaluation: single run")

    # Fit label encoder on combined labels to keep mappings consistent
    label_encoder = LabelEncoder()
    label_encoder.fit([label for bundle in train_bundles for label in bundle.labels] + eval_bundle.labels)
    y_train = label_encoder.transform([label for bundle in train_bundles for label in bundle.labels])
    y_eval = label_encoder.transform(eval_bundle.labels)

    # Compute random baseline (based on training label distribution)
    print("Computing random baseline...")
    baseline_metrics = _compute_random_baseline(y_train, y_eval, n_iterations=100)
    print(f"Random baseline: F1={baseline_metrics['baseline_f1']:.3f} (±{baseline_metrics['baseline_f1_std']:.3f}), "
          f"Acc={baseline_metrics['baseline_accuracy']:.3f}")

    encoding_methods = dict(get_encoding_methods())
    requested_encodings = set(encodings) if encodings else None

    openai_paths = _maybe_collect_openai_paths([bundle.path for bundle in train_bundles] + [eval_bundle.path], resolved_text_source)
    if openai_paths and (requested_encodings is None or "openai" in requested_encodings):
        encoding_methods = {"openai": "OpenAI", **encoding_methods}
    elif requested_encodings and "openai" in requested_encodings and not openai_paths:
        print("Requested OpenAI embeddings but could not locate them; skipping.")

    if requested_encodings:
        encoding_methods = {name: desc for name, desc in encoding_methods.items() if name in requested_encodings}
        if not encoding_methods:
            raise ValueError(f"No encodings left after filtering. Requested: {requested_encodings}")

    imbalance_strategies = get_imbalance_strategies()
    if strategies:
        imbalance_strategies = {k: v for k, v in imbalance_strategies.items() if k in set(strategies)}
        if not imbalance_strategies:
            raise ValueError(f"No imbalance strategies left after filtering. Requested: {strategies}")

    # Dry run: preview what would be executed
    if dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE - Preview only, no execution")
        print("=" * 70)
        total_runs = len(encoding_methods) * sum(len(get_base_models(enc)) for enc in encoding_methods) * len(imbalance_strategies)
        total_runs_filtered = 0
        for encoding_name in encoding_methods:
            available_models = get_base_models(encoding_name)
            if models:
                available_models = {k: v for k, v in available_models.items() if k in set(models)}
            total_runs_filtered += len(available_models) * len(imbalance_strategies)

        print(f"\nConfiguration:")
        print(f"  Encodings: {list(encoding_methods.keys())}")
        print(f"  Imbalance strategies: {list(imbalance_strategies.keys())}")
        print(f"  Total combinations: {total_runs_filtered}")
        if use_cv:
            total_trials = cv_folds * cv_repeats
            print(f"  Evaluation: Repeated CV ({cv_repeats} × {cv_folds}-fold = {total_trials} trials per combination)")
            print(f"  Total trials: {total_runs_filtered * total_trials}")
        else:
            print(f"  Evaluation: Single run per combination")

        print(f"\nDatasets:")
        for bundle in train_bundles:
            print(f"  Train: {bundle.dataset_id} ({len(bundle.labels)} samples)")
        print(f"  Eval:  {eval_bundle.dataset_id} ({len(eval_bundle.labels)} samples)")

        print(f"\nOutput would be saved to: {output_path or _default_output_path(train_bundles, eval_bundle, resolved_text_source)}")
        print("\n" + "=" * 70)
        print("Dry run complete. Use without --dry-run to execute.")
        print("=" * 70)
        return pd.DataFrame()

    results = []

    for encoding_name, encoding_desc in encoding_methods.items():
        print(f"\n=== Encoding: {encoding_desc} ===")
        X_train, X_eval = _encode_for_method(
            encoding_name, train_bundles, eval_bundle, resolved_text_source, openai_paths
        )

        available_models = get_base_models(encoding_name)
        if models:
            available_models = {k: v for k, v in available_models.items() if k in set(models)}
            if not available_models:
                raise ValueError(f"No models available for encoding '{encoding_name}' after filtering {models}")

        for model_name, base_model in available_models.items():
            if use_cv:
                # Repeated CV mode
                print(f"\n--- {model_name}: Repeated CV ({cv_repeats} × {cv_folds}-fold) ---")
                for strategy_name, strategy in imbalance_strategies.items():
                    cv_metrics = _evaluate_cross_project_repeated(
                        model=base_model,
                        model_name=model_name,
                        strategy_name=strategy_name,
                        strategy=strategy,
                        X_train=X_train,
                        y_train=y_train,
                        X_eval=X_eval,
                        y_eval=y_eval,
                        n_folds=cv_folds,
                        n_repeats=cv_repeats,
                        random_state=42,
                    )

                    summary = {
                        "encoding": encoding_name,
                        "model": model_name,
                        "strategy": strategy_name,
                        "detector": detector,
                        "text_source": resolved_text_source,
                        "train_datasets": [bundle.dataset_id for bundle in train_bundles],
                        "eval_dataset": eval_bundle.dataset_id,
                        "train_size": int(len(y_train)),
                        "eval_size": int(len(y_eval)),
                        "label_classes": label_encoder.classes_.tolist(),
                        **cv_metrics,
                    }
                    results.append(summary)
            else:
                # Single run mode
                for strategy_name, strategy in imbalance_strategies.items():
                    model = clone(base_model)
                    model = configure_model_for_strategy(model, model_name, strategy_name)

                    X_train_proc, y_train_proc = apply_imbalance_strategy(X_train, y_train, strategy_name, strategy)

                    sample_weight = None
                    if strategy_name == "class_weight":
                        sample_weight = compute_balanced_sample_weights(y_train_proc)

                    fit_kwargs = {}
                    if sample_weight is not None and "sample_weight" in inspect.signature(model.fit).parameters:
                        fit_kwargs["sample_weight"] = sample_weight

                    if model_name in {"logistic_regression", "svm", "knn", "feedforward_nn"}:
                        with_mean = not hasattr(X_train_proc, "toarray")
                        scaler = StandardScaler(with_mean=with_mean)
                        X_train_scaled = scaler.fit_transform(X_train_proc)
                        X_eval_scaled = scaler.transform(X_eval)
                        model.fit(X_train_scaled, y_train_proc, **fit_kwargs)
                        metrics = _evaluate_model(model, X_eval_scaled, y_eval, baseline_metrics)
                    else:
                        model.fit(X_train_proc, y_train_proc, **fit_kwargs)
                        metrics = _evaluate_model(model, X_eval, y_eval, baseline_metrics)

                    summary = {
                        "encoding": encoding_name,
                        "model": model_name,
                        "strategy": strategy_name,
                        "detector": detector,
                        "text_source": resolved_text_source,
                        "train_datasets": [bundle.dataset_id for bundle in train_bundles],
                        "eval_dataset": eval_bundle.dataset_id,
                        "train_size": int(len(y_train)),
                        "eval_size": int(len(y_eval)),
                        "label_classes": label_encoder.classes_.tolist(),
                        **metrics,
                    }
                    results.append(summary)

                    sig_marker = "*" if summary.get("significantly_better_than_random", False) else ""
                    pval = summary.get("mcnemar_pvalue", 1.0)
                    f1_imp = summary.get("f1_improvement", 0.0)
                    print(
                        f"[{encoding_name} | {model_name} | {strategy_name}] "
                        f"F1={summary['f1_mean']:.3f} (Δ{f1_imp:+.3f}) "
                        f"p={pval:.4f}{sig_marker}"
                    )

    results_df = pd.DataFrame(results)

    if output_path is None:
        output_path = _default_output_path(train_bundles, eval_bundle, resolved_text_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep list-like fields readable in CSV (metadata fields only)
    list_fields = ["train_datasets", "label_classes"]
    for field in list_fields:
        if field in results_df.columns:
            results_df[field] = results_df[field].apply(lambda v: json.dumps(v))

    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    else:
        results_df.to_csv(output_path, index=False)

    print(f"\nSaved cross-project metrics to {output_path}")

    # Generate simple summary (only for CSV outputs)
    if output_path.suffix.lower() == ".csv":
        generate_simple_summary(output_path)

    return results_df


def _resolve_project_dataset_path(project: str, detector: str) -> Path:
    """
    Resolve dataset path for a project/detector combination.

    Uses centralized configuration from dataset_config.py.
    """
    key = (project, detector)
    if key not in DATASET_FOLDERS:
        raise FileNotFoundError(
            f"No dataset configuration for {project}/{detector}. "
            f"Available: {list(DATASET_FOLDERS.keys())}"
        )

    folder_name = DATASET_FOLDERS[key]
    project_dir_name = PROJECT_DIR_NAMES.get(project, project)
    classification_dir = DATASET_BASE_DIR / folder_name / "data" / "classification" / project_dir_name

    # Try primary filename first
    data_file = classification_dir / CLASSIFICATION_DATA_FILENAME
    if data_file.exists():
        return data_file

    # Fallback to other known filenames
    for filename in CLASSIFICATION_FILENAMES:
        candidate = classification_dir / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Classification data not found in {classification_dir}. "
        f"Tried: {CLASSIFICATION_DATA_FILENAME} and {CLASSIFICATION_FILENAMES}"
    )


def run_leave_one_out(
    detector: str = "all",
    text_source: str = TEXT_SOURCE_COMBINED_FILTERED,
    encodings: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
    output_dir: Path | None = None,
    use_cv: bool = False,
    cv_folds: int = 5,
    cv_repeats: int = 2,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run leave-one-out cross-project evaluation.

    For ACDC: 3 combinations (train on 2, eval on 1)
    For Arcan: 2 combinations (train on 1, eval on 1)

    Args:
        detector: "acdc", "arcan", or "all" (default)
        text_source: Text fields to use
        encodings: Encoding methods to use
        models: ML models to use
        strategies: Imbalance strategies to use
        output_dir: Directory for output files
        use_cv: Use repeated evaluation with significance testing (default: False)
        cv_folds: Number of trials per repetition (default: 5)
        cv_repeats: Number of repetitions (default: 2, total trials = folds × repeats)
        dry_run: Preview mode - show what would be run without executing (default: False)

    Returns:
        Combined DataFrame with all results
    """
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DATASET_BASE_DIR.parents[2] / "cross_project_loo" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    combinations = []
    if detector in ("acdc", "all"):
        for train_projects, eval_project in LEAVE_ONE_OUT_ACDC:
            combinations.append(("acdc", train_projects, eval_project))
    if detector in ("arcan", "all"):
        for train_projects, eval_project in LEAVE_ONE_OUT_ARCAN:
            combinations.append(("arcan", train_projects, eval_project))

    print(f"Running leave-one-out cross-project evaluation")
    print(f"Detector(s): {detector}")
    print(f"Total combinations: {len(combinations)}")
    if use_cv:
        total_trials = cv_folds * cv_repeats
        print(f"Evaluation: Repeated CV ({cv_repeats} × {cv_folds}-fold = {total_trials} trials)")
    else:
        print("Evaluation: single run")

    if dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE - Preview only")
        print("=" * 70)
        for det, train_projects, eval_project in combinations:
            print(f"\n[{det.upper()}] Train: {'+'.join(train_projects)} → Eval: {eval_project}")
        print("\n" + "=" * 70)
        print("Dry run complete. Use without --dry-run to execute.")
        print("=" * 70)
        return pd.DataFrame()

    print("=" * 70)

    all_results = []

    for det, train_projects, eval_project in combinations:
        print(f"\n{'=' * 70}")
        print(f"[{det.upper()}] Train: {'+'.join(train_projects)} → Eval: {eval_project}")
        print("=" * 70)

        try:
            train_paths = [_resolve_project_dataset_path(p, det) for p in train_projects]
            eval_path = _resolve_project_dataset_path(eval_project, det)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        train_slug = "+".join(train_projects)
        output_file = output_dir / f"cross_{det}_{train_slug}_to_{eval_project}_{text_source}.csv"

        try:
            results_df = run_cross_project_classification(
                train_data=train_paths,
                eval_data=eval_path,
                text_source=text_source,
                encodings=encodings,
                models=models,
                strategies=strategies,
                output_path=output_file,
                use_cv=use_cv,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
            )
            all_results.append(results_df)
        except Exception as e:
            print(f"Error in combination: {e}")
            continue

    if not all_results:
        print("No successful runs!")
        return pd.DataFrame()

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / f"cross_project_loo_{detector}_combined.csv"
    combined_df.to_csv(combined_path, index=False)

    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(combined_df)}")
    print(f"Combined results saved to: {combined_path}")

    # Generate simple summary
    generate_simple_summary(combined_path)

    # Summary statistics
    if "f1_mean" in combined_df.columns:
        print(f"\nF1 Statistics by Detector:")
        for det_name in combined_df["detector"].unique():
            det_df = combined_df[combined_df["detector"] == det_name]
            print(f"  {det_name.upper()}: mean={det_df['f1_mean'].mean():.3f}, "
                  f"std={det_df['f1_mean'].std():.3f}, "
                  f"min={det_df['f1_mean'].min():.3f}, max={det_df['f1_mean'].max():.3f}")

    # Baseline comparison statistics
    if "significantly_better_than_random" in combined_df.columns:
        sig_count = combined_df["significantly_better_than_random"].sum()
        total_count = len(combined_df)
        sig_pct = (sig_count / total_count) * 100 if total_count > 0 else 0
        print(f"\nSignificance vs Random Baseline (McNemar p<0.05):")
        print(f"  Significant improvements: {sig_count}/{total_count} ({sig_pct:.1f}%)")

        if "f1_improvement" in combined_df.columns:
            mean_imp = combined_df["f1_improvement"].mean()
            print(f"  Mean F1 improvement over baseline: {mean_imp:+.3f}")

        # By detector
        for det_name in combined_df["detector"].unique():
            det_df = combined_df[combined_df["detector"] == det_name]
            det_sig = det_df["significantly_better_than_random"].sum()
            det_total = len(det_df)
            det_pct = (det_sig / det_total) * 100 if det_total > 0 else 0
            det_mean_imp = det_df["f1_improvement"].mean() if "f1_improvement" in det_df.columns else 0
            print(f"  {det_name.upper()}: {det_sig}/{det_total} significant ({det_pct:.1f}%), "
                  f"mean Δ={det_mean_imp:+.3f}")

    print("=" * 70)
    return combined_df


def generate_simple_summary(results_csv_path: Path, output_csv_path: Path | None = None) -> pd.DataFrame:
    """
    Generate a simplified summary CSV with only essential eval metrics.

    Args:
        results_csv_path: Path to the detailed results CSV
        output_csv_path: Path to save the simplified summary CSV (defaults to input_path with _summary suffix)

    Returns:
        DataFrame with simplified summary
    """
    # Load the detailed results
    df = pd.read_csv(results_csv_path)

    # Select only essential columns
    essential_cols = [
        'encoding',
        'model',
        'strategy',
        'detector',
        'text_source',
        'train_datasets',
        'eval_dataset',
        'precision_mean',
        'recall_mean',
        'f1_mean'
    ]

    # Filter to only columns that exist
    available_cols = [col for col in essential_cols if col in df.columns]
    summary_df = df[available_cols].copy()

    # Round metrics to 3 decimal places
    metric_cols = ['precision_mean', 'recall_mean', 'f1_mean']
    for col in metric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)

    # Determine output path
    if output_csv_path is None:
        input_path = Path(results_csv_path)
        output_csv_path = input_path.parent / f"{input_path.stem}_summary{input_path.suffix}"

    # Save summary
    summary_df.to_csv(output_csv_path, index=False)
    print(f"\nSimple summary saved to: {output_csv_path}")

    return summary_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-project classification evaluation. "
            "By default, runs leave-one-out evaluation across all configured datasets. "
            "Use --train-data and --eval-data for custom train/eval splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--leave-one-out", "--loo",
        action="store_true",
        default=True,
        dest="leave_one_out",
        help="Run leave-one-out cross-project evaluation (default mode).",
    )
    mode_group.add_argument(
        "--detector",
        choices=["acdc", "arcan", "all"],
        default="all",
        help="Which detector to use for leave-one-out: 'acdc' (3 combos), 'arcan' (2 combos), or 'all' (5 combos).",
    )

    # Custom mode (overrides leave-one-out)
    custom_group = parser.add_argument_group("Custom Train/Eval (overrides leave-one-out)")
    custom_group.add_argument(
        "--train-data",
        "-t",
        action="append",
        help="Path or directory containing classification data. Can be repeated for multiple training datasets.",
    )
    custom_group.add_argument(
        "--eval-data",
        "-e",
        help="Path or directory containing the evaluation classification data.",
    )

    # Common options
    common_group = parser.add_argument_group("Common Options")
    common_group.add_argument(
        "--text-source",
        default=TEXT_SOURCE_DESCRIPTION,
        choices=sorted(VALID_TEXT_SOURCES),
        help="Which text fields to use when building documents.",
    )
    common_group.add_argument(
        "--encoding",
        action="append",
        choices=sorted(set(get_encoding_methods().keys()) | {"openai"}),
        help="Restrict to specific encodings. Defaults to all available.",
    )
    common_group.add_argument(
        "--model",
        action="append",
        help="Restrict to specific model names. Defaults to all models for each encoding.",
    )
    common_group.add_argument(
        "--strategy",
        action="append",
        choices=sorted(get_imbalance_strategies().keys()),
        help="Restrict imbalance strategies. Defaults to all strategies.",
    )
    common_group.add_argument(
        "--output",
        help="Output path (file for custom mode, directory for leave-one-out mode).",
    )

    # CV options
    cv_group = parser.add_argument_group("Cross-Validation Options")
    cv_group.add_argument(
        "--use-cv",
        action="store_true",
        help="Use repeated evaluation with significance testing instead of single run.",
    )
    cv_group.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of bootstrap/repeated trials per repetition (default: 5).",
    )
    cv_group.add_argument(
        "--cv-repeats",
        type=int,
        default=2,
        help="Number of repetitions (default: 2, total trials = folds × repeats).",
    )

    # Dry run option
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually executing (preview mode).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Check if custom train/eval mode
    if args.train_data and args.eval_data:
        # Custom mode: specific train/eval datasets
        print("Running in custom train/eval mode")
        train_paths = [_resolve_classification_data_path(path) for path in args.train_data]
        eval_path = _resolve_classification_data_path(args.eval_data)
        output_path = Path(args.output).expanduser().resolve() if args.output else None

        run_cross_project_classification(
            train_data=train_paths,
            eval_data=eval_path,
            text_source=args.text_source,
            encodings=args.encoding,
            models=args.model,
            strategies=args.strategy,
            output_path=output_path,
            use_cv=args.use_cv,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            dry_run=args.dry_run,
        )
    elif args.train_data or args.eval_data:
        # Partial specification - error
        raise ValueError(
            "Both --train-data and --eval-data must be specified for custom mode. "
            "Omit both to use leave-one-out mode."
        )
    else:
        # Default: leave-one-out mode
        output_dir = Path(args.output).expanduser().resolve() if args.output else None

        run_leave_one_out(
            detector=args.detector,
            text_source=args.text_source,
            encodings=args.encoding,
            models=args.model,
            strategies=args.strategy,
            output_dir=output_dir,
            use_cv=args.use_cv,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
