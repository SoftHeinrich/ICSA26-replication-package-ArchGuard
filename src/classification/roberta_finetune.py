#!/usr/bin/env python3
"""
RoBERTa fine-tuning for architectural smell classification with 2x5-fold CV.

Uses repeated stratified k-fold cross-validation (2 repeats × 5 folds = 10 trials)
to evaluate model performance, matching the supervised_ml_all evaluation approach.
Long issues are chunked into sub-issues before training to avoid truncation and to
squeeze more signal out of lengthy reports. Issue labels are recovered from chunk
predictions via configurable aggregation (`vote` by default; also supports `mean`,
`max`, `first`).

Results include mean ± std across all CV folds for robust performance estimates.

Example:
    python -m src.classification.roberta_finetune \\
      --text-source combined \\
      --model roberta-base \\
      --issue-aggregation vote \\
      --chunk-words 256 \\
      --n-splits 5 \\
      --n-repeats 2 \\
      --output-dir Output/hf_roberta
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from scipy.stats import wilcoxon
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from .simple_ml import (
    TEXT_SOURCE_DESCRIPTION,
    VALID_TEXT_SOURCES,
    chunk_text,
    extract_texts_from_data,
    normalize_text_source,
)
from .dataset_config import DATASET_BASE_DIR

LOGGER = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class IssueExample:
    """One full issue (before chunking)."""

    issue_id: str
    label: str
    text: str


@dataclass
class ChunkExample:
    """A chunked sub-issue produced from a long report."""

    issue_id: str
    chunk_id: str
    label: str
    text: str


@dataclass
class RunPaths:
    """Filesystem locations used for a single dataset run."""

    run_dir: Path
    checkpoint_dir: Path
    issue_predictions_file: Path
    chunk_predictions_file: Path
    chunk_logits_file: Path
    metrics_file: Path
    config_file: Path


class RobertaChunkDataset(Dataset):
    """Lightweight Hugging Face dataset wrapper over chunked issue texts."""

    def __init__(self, chunks: list[ChunkExample], tokenizer, label_to_id: dict[str, int], max_length: int):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        example = self.chunks[idx]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.label_to_id[example.label]
        return encoded


class EpochEvalCallback(TrainerCallback):
    """
    Callback to evaluate on test set at the end of each epoch.

    Saves predictions and metrics for each epoch so users can select
    the best performing epoch later.
    """

    def __init__(self, test_dataset, test_chunks, label_to_id, id_to_label,
                 aggregation_strategies, fold_num, run_paths):
        self.test_dataset = test_dataset
        self.test_chunks = test_chunks
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.aggregation_strategies = aggregation_strategies
        self.fold_num = fold_num
        self.run_paths = run_paths
        self.epoch_results = []  # Store results for all epochs
        self.trainer = None  # Will be set after trainer creation

    def set_trainer(self, trainer):
        """Set the trainer reference after initialization."""
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate on test set at the end of each epoch (starting from epoch 4)."""
        if self.trainer is None:
            LOGGER.warning("Trainer not set in callback, skipping epoch evaluation")
            return

        epoch = int(state.epoch)  # Current epoch number

        # Skip evaluation for first 3 epochs
        if epoch <= 3:
            LOGGER.info(f"Skipping evaluation for epoch {epoch} (evaluating starts after epoch 3)")
            return

        LOGGER.info(f"Evaluating epoch {epoch} on test set...")

        # Run prediction on test set
        predictions = self.trainer.predict(self.test_dataset)
        logits = predictions.predictions

        # Compute chunk-level metrics
        chunk_metrics = _compute_chunk_metrics(predictions, self.id_to_label, self.label_to_id)

        # Compute issue-level metrics for all aggregation strategies
        epoch_result = {
            "epoch": epoch,
            "fold": self.fold_num,
            "chunk_accuracy": chunk_metrics["chunk_accuracy"],
            "chunk_precision": chunk_metrics["chunk_precision"],
            "chunk_recall": chunk_metrics["chunk_recall"],
            "chunk_f1": chunk_metrics["chunk_f1"],
        }

        # Add issue-level metrics for each aggregation strategy
        for agg in self.aggregation_strategies:
            issue_rows, issue_metrics = _aggregate_issue_predictions(
                self.test_chunks, logits, self.label_to_id, self.id_to_label, agg
            )
            epoch_result[f"issue_accuracy_{agg}"] = issue_metrics["issue_accuracy"]
            epoch_result[f"issue_precision_{agg}"] = issue_metrics["issue_precision"]
            epoch_result[f"issue_recall_{agg}"] = issue_metrics["issue_recall"]
            epoch_result[f"issue_f1_{agg}"] = issue_metrics["issue_f1"]

        self.epoch_results.append(epoch_result)

        # Log epoch results
        LOGGER.info(f"Epoch {epoch} Results:")
        LOGGER.info(f"  Chunk  - Acc: {chunk_metrics['chunk_accuracy']:.3f}, "
                   f"F1: {chunk_metrics['chunk_f1']:.3f}")
        for agg in self.aggregation_strategies:
            LOGGER.info(f"  Issue ({agg}) - F1: {epoch_result[f'issue_f1_{agg}']:.3f}")

    def get_results(self):
        """Return all epoch results."""
        return self.epoch_results


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataframe(data_path: Path) -> pd.DataFrame:
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    df = pd.DataFrame(payload)
    if "label" not in df.columns:
        raise ValueError(f"Dataset {data_path} is missing required 'label' column.")
    return df


def _resolve_data_paths(target: str) -> list[Path]:
    """
    Accept either a single JSON file or a directory containing classification datasets.
    Mirrors the zero-shot toolchain but avoids the OpenAI dependency.
    """
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if path.is_file():
        return [path]

    patterns = [
        "classification_data_clean*.json",
        "classification_clean*.json",
        "classification*clean*.json",
        "classification*.json",
    ]

    discovered: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in path.rglob(pattern):
            if candidate.is_file() and candidate not in seen:
                discovered.append(candidate)
                seen.add(candidate)

    if not discovered:
        raise ValueError(f"No classification datasets found under directory: {path}")

    LOGGER.info("Discovered %d classification dataset(s) under %s", len(discovered), path)
    return discovered


def _derive_run_paths(
    data_path: Path,
    model_name: str,
    text_source: str,
    chunk_words: int,
    aggregation: str,
    base_dir: Path | None,
) -> RunPaths:
    """Create deterministic output paths when users do not provide them explicitly."""
    # Extract detector-specific folder name from path structure
    # Path: .../shepard_acdc/data/classification/dlr-shepard-shepard/classification_data_clean.json
    # We want: shepard_acdc (4 levels up from data_path)
    detector_folder = data_path.parent.parent.parent.parent.name

    # Fallback to project name if detector folder pattern not found
    if not detector_folder or detector_folder == "supervised_ml_all":
        detector_folder = data_path.parent.name or data_path.stem

    safe_model = model_name.replace("/", "_").replace(":", "_")
    # Use detector-specific folder to avoid overwrites between acdc/arcan results
    base = (base_dir / detector_folder) if base_dir else (Path("Output") / "hf_roberta" / detector_folder)
    run_name = f"{data_path.stem}_{text_source}_cw{chunk_words}_{aggregation}_{safe_model}"
    run_dir = (base / run_name).resolve()
    checkpoint_dir = run_dir / "checkpoints"

    return RunPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        issue_predictions_file=run_dir / "issue_predictions.csv",
        chunk_predictions_file=run_dir / "chunk_predictions.csv",
        chunk_logits_file=run_dir / "chunk_logits.npz",
        metrics_file=run_dir / "metrics.json",
        config_file=run_dir / "run_config.json",
    )


def _build_issue_examples(df: pd.DataFrame, text_source: str) -> list[IssueExample]:
    normalized_source = normalize_text_source(text_source)
    texts = extract_texts_from_data(df, text_source=normalized_source)
    examples: list[IssueExample] = []

    for idx, text in enumerate(texts):
        raw_label = df.iloc[idx]["label"]
        if pd.isna(raw_label):
            continue

        label = str(raw_label).strip().lower()
        issue_id = str(df.iloc[idx].get("id", idx))
        cleaned_text = text.strip()

        if not cleaned_text:
            continue

        examples.append(IssueExample(issue_id=issue_id, label=label, text=cleaned_text))

    return examples


def _chunk_issues(issues: Iterable[IssueExample], chunk_words: int, max_chunks: int | None = None) -> list[ChunkExample]:
    """Legacy word-based chunking. Use _chunk_issues_by_tokens for token-based chunking."""
    chunked: list[ChunkExample] = []
    for issue in issues:
        pieces = chunk_text(issue.text, max_words=chunk_words)
        if max_chunks:
            pieces = pieces[:max_chunks]

        for idx, piece in enumerate(pieces):
            chunked.append(
                ChunkExample(
                    issue_id=issue.issue_id,
                    chunk_id=f"{issue.issue_id}_chunk{idx + 1}",
                    label=issue.label,
                    text=piece,
                )
            )
    return chunked


def _chunk_issues_by_tokens(
    issues: Iterable[IssueExample],
    tokenizer,
    max_length: int = 512,
    max_chunks: int | None = None,
    overlap: int = 0,
) -> list[ChunkExample]:
    """
    Chunk issues by token count after tokenization.

    This avoids truncation issues by chunking at the token level,
    ensuring each chunk fits exactly within max_length tokens.

    Args:
        issues: List of IssueExample objects
        tokenizer: HuggingFace tokenizer
        max_length: Maximum tokens per chunk (default 512, RoBERTa max)
        max_chunks: Optional cap on chunks per issue
        overlap: Number of overlapping tokens between chunks (default 0)

    Returns:
        List of ChunkExample objects
    """
    chunked: list[ChunkExample] = []

    # Account for special tokens [CLS] and [SEP] plus tokenization variance
    # Use -3 instead of -2 to provide buffer for decode/re-encode token count differences
    effective_max_length = max_length - 3

    for issue in issues:
        # Tokenize without truncation to get all tokens
        encoded = tokenizer(
            issue.text,
            truncation=False,
            add_special_tokens=False,  # We'll handle special tokens later
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"]

        # Split into chunks
        chunks_token_ids = []
        start = 0
        while start < len(token_ids):
            end = start + effective_max_length
            chunk_ids = token_ids[start:end]
            chunks_token_ids.append(chunk_ids)

            if overlap > 0 and end < len(token_ids):
                start = end - overlap
            else:
                start = end

        # Apply max_chunks limit
        if max_chunks:
            chunks_token_ids = chunks_token_ids[:max_chunks]

        # Decode chunks back to text
        for idx, chunk_ids in enumerate(chunks_token_ids):
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunked.append(
                ChunkExample(
                    issue_id=issue.issue_id,
                    chunk_id=f"{issue.issue_id}_chunk{idx + 1}",
                    label=issue.label,
                    text=chunk_text,
                )
            )

    return chunked


def _undersample_issues(issues: list[IssueExample], seed: int = 42) -> list[IssueExample]:
    """
    Apply random undersampling at the issue level to balance classes.

    This ensures we have equal representation of each class before chunking,
    matching the approach used in simple_ml.py for traditional ML models.
    """
    if not issues:
        return issues

    # Create arrays for undersampling
    issue_indices = np.arange(len(issues))
    labels = np.array([issue.label for issue in issues])

    # Get class distribution before
    unique, counts = np.unique(labels, return_counts=True)
    dist_before = dict(zip(unique, counts))
    LOGGER.info("Class distribution before undersampling: %s", dist_before)

    # Apply random undersampling
    rus = RandomUnderSampler(random_state=seed)
    indices_resampled, _ = rus.fit_resample(issue_indices.reshape(-1, 1), labels)
    indices_resampled = indices_resampled.flatten()

    # Get resampled issues
    resampled_issues = [issues[i] for i in indices_resampled]

    # Get class distribution after
    labels_after = np.array([issue.label for issue in resampled_issues])
    unique_after, counts_after = np.unique(labels_after, return_counts=True)
    dist_after = dict(zip(unique_after, counts_after))
    LOGGER.info("Class distribution after undersampling: %s (total: %d -> %d)",
                dist_after, len(issues), len(resampled_issues))

    return resampled_issues


def _create_cv_splitter(n_splits: int = 5, n_repeats: int = 2, seed: int = 42) -> RepeatedStratifiedKFold:
    """Create a repeated stratified k-fold cross-validator."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def _filter_chunks_by_issue_ids(chunks: Iterable[ChunkExample], allowed_ids: set[str]) -> list[ChunkExample]:
    return [chunk for chunk in chunks if chunk.issue_id in allowed_ids]


def _pick_positive_label(label_to_id: dict[str, int]) -> int | None:
    for candidate in ("smell", "positive", "1", "yes", "true"):
        if candidate in label_to_id:
            return label_to_id[candidate]
    return None


def _compute_chunk_metrics(eval_pred, id_to_label: dict[int, str], label_to_id: dict[str, int]) -> dict[str, float]:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=1)

    positive_id = _pick_positive_label(label_to_id)
    if positive_id is None:
        # Fallback to the first label id to avoid crashing on unexpected labels
        positive_id = sorted(set(label_to_id.values()))[0]

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[positive_id], zero_division=0
    )

    return {
        "chunk_accuracy": accuracy_score(labels, preds),
        "chunk_precision": float(precision[0]),
        "chunk_recall": float(recall[0]),
        "chunk_f1": float(f1[0]),
        "chunk_support_pos": int(support[0]),
    }


def _aggregate_issue_predictions(
    chunks: Sequence[ChunkExample],
    logits: np.ndarray,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    aggregation: str,
) -> tuple[list[dict], dict[str, float]]:
    """
    Merge chunk-level predictions back to their parent issues with configurable aggregation.
    """
    if logits.size == 0 or not chunks:
        return [], {}

    logits_max = np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits - logits_max) / np.exp(logits - logits_max).sum(axis=1, keepdims=True)
    chunk_preds = np.argmax(probs, axis=1)
    grouped: dict[str, dict[str, list]] = {}

    for chunk, prob, pred_id in zip(chunks, probs, chunk_preds):
        bucket = grouped.setdefault(
            chunk.issue_id,
            {"true_label": chunk.label, "probs": [], "chunk_ids": [], "pred_ids": []},
        )
        bucket["probs"].append(prob)
        bucket["chunk_ids"].append(chunk.chunk_id)
        bucket["pred_ids"].append(int(pred_id))

    issue_rows: list[dict] = []
    true_ids: list[int] = []
    pred_ids: list[int] = []

    for issue_id, payload in grouped.items():
        probs_stack = np.vstack(payload["probs"])

        if aggregation == "mean":
            pred_id = int(np.argmax(np.mean(probs_stack, axis=0)))
        elif aggregation == "max":
            pred_id = int(np.argmax(np.max(probs_stack, axis=0)))
        elif aggregation == "vote":
            votes = np.bincount(payload["pred_ids"], minlength=len(label_to_id))
            top_votes = np.flatnonzero(votes == votes.max())
            if len(top_votes) == 1:
                pred_id = int(top_votes[0])
            else:
                avg_probs = np.mean(probs_stack, axis=0)
                pred_id = int(top_votes[np.argmax(avg_probs[top_votes])])
        elif aggregation == "first":
            pred_id = payload["pred_ids"][0]
        else:
            raise ValueError(f"Unsupported aggregation strategy: {aggregation}")

        pred_label = id_to_label[pred_id]
        true_label = payload["true_label"]

        issue_rows.append(
            {
                "issue_id": issue_id,
                "true_label": true_label,
                "predicted_label": pred_label,
                "chunk_count": len(payload["chunk_ids"]),
            }
        )

        pred_ids.append(pred_id)
        true_ids.append(label_to_id[true_label])

    positive_id = _pick_positive_label(label_to_id)
    if positive_id is None:
        positive_id = sorted(set(label_to_id.values()))[0]

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=[positive_id], zero_division=0
    )

    metrics = {
        "issue_accuracy": accuracy_score(true_ids, pred_ids),
        "issue_precision": float(precision[0]),
        "issue_recall": float(recall[0]),
        "issue_f1": float(f1[0]),
        "issue_support_pos": int(support[0]),
    }

    return issue_rows, metrics


def _save_artifacts(
    paths: RunPaths,
    args: argparse.Namespace,
    chunk_rows: list[dict],
    issue_rows: list[dict],
    metrics: dict,
    chunk_arrays: dict | None = None,
) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    with paths.config_file.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    if chunk_rows:
        chunk_df = pd.DataFrame(chunk_rows)
        chunk_df.to_csv(paths.chunk_predictions_file, index=False)
        LOGGER.info("Chunk predictions saved to %s", paths.chunk_predictions_file)

    if chunk_arrays:
        np.savez(paths.chunk_logits_file, **chunk_arrays)
        LOGGER.info("Chunk logits/probabilities saved to %s", paths.chunk_logits_file)

    if issue_rows:
        issue_df = pd.DataFrame(issue_rows)
        issue_df.to_csv(paths.issue_predictions_file, index=False)
        LOGGER.info("Issue-level predictions saved to %s", paths.issue_predictions_file)

    with paths.metrics_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Metrics written to %s", paths.metrics_file)


def _run_single_dataset(data_path: Path, args: argparse.Namespace) -> None:
    LOGGER.info("Loading dataset from %s", data_path)
    df = _load_dataframe(data_path)
    issues = _build_issue_examples(df, args.text_source)
    if not issues:
        LOGGER.warning("No valid issues found in %s; skipping", data_path)
        return

    label_space = sorted({issue.label for issue in issues})
    label_to_id = {label: idx for idx, label in enumerate(label_space)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Create CV splitter (2x5-fold by default)
    n_splits = getattr(args, 'n_splits', 5)
    n_repeats = getattr(args, 'n_repeats', 2)
    cv_splitter = _create_cv_splitter(n_splits=n_splits, n_repeats=n_repeats, seed=args.seed)

    # Prepare issue-level arrays for CV splitting
    issue_ids_array = np.array([issue.issue_id for issue in issues])
    labels_array = np.array([label_to_id[issue.label] for issue in issues])

    total_folds = n_splits * n_repeats
    LOGGER.info("Running %d × %d-fold CV (%d total folds)", n_repeats, n_splits, total_folds)

    # Initialize tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Aggregation strategies to evaluate
    aggregation_strategies = ["vote", "mean", "max", "first"]

    # Storage for CV results (matching simple_ml.py format)
    fold_metrics = {
        "chunk_accuracy": [], "chunk_precision": [], "chunk_recall": [], "chunk_f1": [],
        "random_issue_f1": [],  # Random baseline F1 scores for significance testing
    }
    # Per-aggregation-strategy metrics
    for agg in aggregation_strategies:
        fold_metrics[f"issue_accuracy_{agg}"] = []
        fold_metrics[f"issue_precision_{agg}"] = []
        fold_metrics[f"issue_recall_{agg}"] = []
        fold_metrics[f"issue_f1_{agg}"] = []

    all_chunk_rows = []
    all_issue_rows = {agg: [] for agg in aggregation_strategies}
    all_epoch_results = []  # Store epoch-wise results across all folds

    # Run CV folds
    fold_num = 0
    for train_idx, test_idx in cv_splitter.split(issue_ids_array, labels_array):
        fold_num += 1
        LOGGER.info("\n=== Fold %d/%d ===", fold_num, total_folds)

        # Get train/test issue IDs for this fold
        train_issue_ids = set(issue_ids_array[train_idx])
        test_issue_ids = set(issue_ids_array[test_idx])

        # Split issues by fold
        train_issues = [issue for issue in issues if issue.issue_id in train_issue_ids]
        test_issues = [issue for issue in issues if issue.issue_id in test_issue_ids]

        # Apply random undersampling to training issues (matching simple_ml.py approach)
        train_issues_original_count = len(train_issues)
        train_issues = _undersample_issues(train_issues, seed=args.seed + fold_num)

        # Chunk issues by tokens (avoids truncation)
        train_chunks = _chunk_issues_by_tokens(
            train_issues, tokenizer, max_length=args.max_length,
            max_chunks=args.max_chunks_per_issue
        )
        test_chunks = _chunk_issues_by_tokens(
            test_issues, tokenizer, max_length=args.max_length,
            max_chunks=args.max_chunks_per_issue
        )

        LOGGER.info("Train: %d issues (undersampled from %d), %d chunks | Test: %d issues, %d chunks",
                    len(train_issues), train_issues_original_count, len(train_chunks),
                    len(test_issues), len(test_chunks))

        # Split training data into train/val if early stopping enabled
        val_dataset = None
        if args.early_stopping:
            # Use 10% of training data for validation
            val_size = max(1, int(len(train_chunks) * 0.1))
            import random
            random.seed(args.seed + fold_num)
            indices = list(range(len(train_chunks)))
            random.shuffle(indices)
            val_indices = set(indices[:val_size])
            train_chunks_split = [chunk for i, chunk in enumerate(train_chunks) if i not in val_indices]
            val_chunks = [chunk for i, chunk in enumerate(train_chunks) if i in val_indices]
            LOGGER.info("Split training into %d train chunks, %d val chunks for early stopping",
                       len(train_chunks_split), len(val_chunks))
            train_dataset = RobertaChunkDataset(train_chunks_split, tokenizer, label_to_id, args.max_length)
            val_dataset = RobertaChunkDataset(val_chunks, tokenizer, label_to_id, args.max_length)
        else:
            train_dataset = RobertaChunkDataset(train_chunks, tokenizer, label_to_id, args.max_length)

        # Create test dataset
        test_dataset = RobertaChunkDataset(test_chunks, tokenizer, label_to_id, args.max_length)

        # Initialize fresh model for this fold
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=len(label_space),
            id2label=id_to_label,
            label2id=label_to_id,
        )

        # Setup paths for this fold
        run_paths = _derive_run_paths(
            data_path,
            args.model,
            args.text_source,
            args.chunk_words,
            args.issue_aggregation,
            Path(args.output_dir) if args.output_dir else None,
        )
        fold_checkpoint_dir = run_paths.checkpoint_dir / f"fold_{fold_num}"
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments with early stopping support
        eval_steps = max(10, len(train_dataset) // (args.batch_size * 4))  # Evaluate 4 times per epoch
        training_args = TrainingArguments(
            output_dir=str(fold_checkpoint_dir),
            eval_strategy="steps" if args.early_stopping else "no",
            eval_steps=eval_steps if args.early_stopping else None,
            save_strategy="steps" if args.early_stopping else "no",
            save_steps=eval_steps if args.early_stopping else None,
            load_best_model_at_end=args.early_stopping,
            metric_for_best_model="eval_loss" if args.early_stopping else None,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=int(args.warmup_ratio * args.epochs * (len(train_dataset) // args.batch_size)),
            logging_steps=25,
            save_total_limit=1,
            seed=args.seed + fold_num,  # Different seed per fold
            report_to=[],
        )

        # Create trainer with optional early stopping and epoch evaluation
        callbacks = []
        if args.early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

        # Add epoch evaluation callback to evaluate on test set every epoch
        epoch_eval_callback = EpochEvalCallback(
            test_dataset=test_dataset,
            test_chunks=test_chunks,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            aggregation_strategies=aggregation_strategies,
            fold_num=fold_num,
            run_paths=run_paths,
        )
        callbacks.append(epoch_eval_callback)

        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if args.early_stopping else None,
            compute_metrics=lambda pred: _compute_chunk_metrics(pred, id_to_label, label_to_id),
            callbacks=callbacks,
        )

        # Set trainer reference in epoch eval callback
        epoch_eval_callback.set_trainer(trainer)

        # Train
        LOGGER.info("Training fold %d...", fold_num)
        trainer.train()

        # Collect epoch-wise results from callback
        epoch_results = epoch_eval_callback.get_results()
        all_epoch_results.extend(epoch_results)

        # Evaluate on test set
        LOGGER.info("Evaluating fold %d...", fold_num)
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions

        # Compute chunk-level metrics
        chunk_metrics = _compute_chunk_metrics(predictions, id_to_label, label_to_id)
        fold_metrics["chunk_accuracy"].append(chunk_metrics["chunk_accuracy"])
        fold_metrics["chunk_precision"].append(chunk_metrics["chunk_precision"])
        fold_metrics["chunk_recall"].append(chunk_metrics["chunk_recall"])
        fold_metrics["chunk_f1"].append(chunk_metrics["chunk_f1"])

        # Compute issue-level metrics for ALL aggregation strategies (no extra cost)
        for agg in aggregation_strategies:
            issue_rows, issue_metrics = _aggregate_issue_predictions(
                test_chunks, logits, label_to_id, id_to_label, agg
            )
            fold_metrics[f"issue_accuracy_{agg}"].append(issue_metrics["issue_accuracy"])
            fold_metrics[f"issue_precision_{agg}"].append(issue_metrics["issue_precision"])
            fold_metrics[f"issue_recall_{agg}"].append(issue_metrics["issue_recall"])
            fold_metrics[f"issue_f1_{agg}"].append(issue_metrics["issue_f1"])

            # Store predictions with fold number
            for row in issue_rows:
                row["fold"] = fold_num
                row["aggregation"] = agg
                all_issue_rows[agg].append(row)

        # Compute random baseline for significance testing (matching simple_ml.py)
        # Use issue-level labels for random baseline comparison
        test_issue_labels = [label_to_id[issue.label] for issue in test_issues]
        train_issue_labels = [label_to_id[issue.label] for issue in train_issues]

        random_baseline = DummyClassifier(strategy='stratified', random_state=args.seed + fold_num)
        # Train on undersampled training labels (just needs label distribution)
        random_baseline.fit(
            np.zeros((len(train_issue_labels), 1)),  # Dummy features
            train_issue_labels
        )
        random_preds = random_baseline.predict(np.zeros((len(test_issue_labels), 1)))

        positive_id = _pick_positive_label(label_to_id)
        if positive_id is None:
            positive_id = sorted(set(label_to_id.values()))[0]

        _, _, random_f1, _ = precision_recall_fscore_support(
            test_issue_labels, random_preds, labels=[positive_id], zero_division=0
        )
        fold_metrics["random_issue_f1"].append(float(random_f1[0]) if hasattr(random_f1, '__len__') else float(random_f1))

        # Log fold results (include precision, recall, and random baseline)
        LOGGER.info("Fold %d Results:", fold_num)
        LOGGER.info("  Chunk  - Acc: %.3f, Prec: %.3f, Rec: %.3f, F1: %.3f",
                    chunk_metrics["chunk_accuracy"], chunk_metrics["chunk_precision"],
                    chunk_metrics["chunk_recall"], chunk_metrics["chunk_f1"])
        LOGGER.info("  Issue (by aggregation strategy):")
        for agg in aggregation_strategies:
            LOGGER.info("    %-5s - Acc: %.3f, Prec: %.3f, Rec: %.3f, F1: %.3f",
                        agg, fold_metrics[f"issue_accuracy_{agg}"][-1],
                        fold_metrics[f"issue_precision_{agg}"][-1],
                        fold_metrics[f"issue_recall_{agg}"][-1],
                        fold_metrics[f"issue_f1_{agg}"][-1])
        LOGGER.info("  Random - F1: %.3f", fold_metrics["random_issue_f1"][-1])

        # Clean up model to save memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate CV results (matching simple_ml.py output format)
    LOGGER.info("\n" + "="*60)
    LOGGER.info("CROSS-VALIDATION RESULTS SUMMARY")
    LOGGER.info("="*60)

    aggregated_metrics = {}
    for metric_name, scores in fold_metrics.items():
        scores_array = np.array(scores)
        aggregated_metrics[f"{metric_name}_mean"] = float(np.mean(scores_array))
        aggregated_metrics[f"{metric_name}_std"] = float(np.std(scores_array))
        aggregated_metrics[f"{metric_name}_all_folds"] = scores

        LOGGER.info("%s: %.3f ± %.3f", metric_name,
                    aggregated_metrics[f"{metric_name}_mean"],
                    aggregated_metrics[f"{metric_name}_std"])

    # Note: Significance testing is done per-aggregation-strategy in the summary section below

    LOGGER.info("\n--- Issue-Level Results by Aggregation Strategy (with random undersampling) ---")
    LOGGER.info("%-8s  %-18s  %-18s  %-18s  %-18s", "Strategy", "Accuracy", "Precision", "Recall", "F1")
    LOGGER.info("-" * 90)

    # Compute significance for each aggregation strategy
    best_agg = None
    best_f1 = -1
    agg_significance = {}

    for agg in aggregation_strategies:
        acc_mean = aggregated_metrics[f"issue_accuracy_{agg}_mean"]
        acc_std = aggregated_metrics[f"issue_accuracy_{agg}_std"]
        prec_mean = aggregated_metrics[f"issue_precision_{agg}_mean"]
        prec_std = aggregated_metrics[f"issue_precision_{agg}_std"]
        rec_mean = aggregated_metrics[f"issue_recall_{agg}_mean"]
        rec_std = aggregated_metrics[f"issue_recall_{agg}_std"]
        f1_mean = aggregated_metrics[f"issue_f1_{agg}_mean"]
        f1_std = aggregated_metrics[f"issue_f1_{agg}_std"]

        LOGGER.info("%-8s  %.3f ± %.3f       %.3f ± %.3f       %.3f ± %.3f       %.3f ± %.3f",
                    agg, acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std)

        # Track best strategy
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_agg = agg

        # Compute significance vs random for this aggregation
        model_f1_array = np.array(fold_metrics[f"issue_f1_{agg}"])
        random_f1_array = np.array(fold_metrics["random_issue_f1"])
        f1_imp = f1_mean - aggregated_metrics["random_issue_f1_mean"]

        if np.allclose(model_f1_array, random_f1_array):
            p_val = 1.0
        else:
            try:
                _, p_val = wilcoxon(model_f1_array, random_f1_array, alternative='greater')
            except ValueError:
                p_val = 1.0

        agg_significance[agg] = {
            "f1_improvement": float(f1_imp),
            "wilcoxon_p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        }

    LOGGER.info("\nBest aggregation strategy: %s (F1: %.3f)", best_agg, best_f1)

    LOGGER.info("\n--- Comparison to Random Baseline (F1: %.3f ± %.3f) ---",
                aggregated_metrics["random_issue_f1_mean"], aggregated_metrics["random_issue_f1_std"])
    LOGGER.info("%-8s  %-12s  %-15s  %s", "Strategy", "Improvement", "p-value", "Significant")
    LOGGER.info("-" * 50)
    for agg in aggregation_strategies:
        sig = agg_significance[agg]
        LOGGER.info("%-8s  +%.3f        %.4f          %s",
                    agg, sig["f1_improvement"], sig["wilcoxon_p_value"],
                    "[OK]" if sig["significant"] else "[X]")

    # Store significance info in aggregated_metrics
    aggregated_metrics["best_aggregation"] = best_agg
    aggregated_metrics["aggregation_significance"] = agg_significance

    # Save aggregated results
    run_paths = _derive_run_paths(
        data_path, args.model, args.text_source, args.chunk_words,
        args.issue_aggregation, Path(args.output_dir) if args.output_dir else None,
    )
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)

    aggregated_metrics.update({
        "label_space": label_space,
        "issue_aggregation": best_agg,  # Best performing aggregation
        "all_aggregation_strategies": aggregation_strategies,
        "chunk_words": args.chunk_words,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "total_folds": total_folds,
        "imbalance_strategy": "random_undersample",  # Hardcoded as requested
        "model_name": args.model,
        # Aliases to match simple_ml.py output format (using best aggregation)
        "accuracy_mean": aggregated_metrics[f"issue_accuracy_{best_agg}_mean"],
        "accuracy_std": aggregated_metrics[f"issue_accuracy_{best_agg}_std"],
        "precision_mean": aggregated_metrics[f"issue_precision_{best_agg}_mean"],
        "precision_std": aggregated_metrics[f"issue_precision_{best_agg}_std"],
        "recall_mean": aggregated_metrics[f"issue_recall_{best_agg}_mean"],
        "recall_std": aggregated_metrics[f"issue_recall_{best_agg}_std"],
        "f1_mean": aggregated_metrics[f"issue_f1_{best_agg}_mean"],
        "f1_std": aggregated_metrics[f"issue_f1_{best_agg}_std"],
        "random_f1_mean": aggregated_metrics["random_issue_f1_mean"],
        "random_f1_std": aggregated_metrics["random_issue_f1_std"],
    })

    # Save all predictions (one file per aggregation strategy)
    for agg in aggregation_strategies:
        if all_issue_rows[agg]:
            issue_df = pd.DataFrame(all_issue_rows[agg])
            agg_predictions_file = run_paths.run_dir / f"issue_predictions_{agg}.csv"
            issue_df.to_csv(agg_predictions_file, index=False)
            LOGGER.info("Predictions (%s) saved to %s", agg, agg_predictions_file)

    # Also save combined predictions with all strategies
    all_rows_combined = []
    for agg in aggregation_strategies:
        all_rows_combined.extend(all_issue_rows[agg])
    if all_rows_combined:
        combined_df = pd.DataFrame(all_rows_combined)
        combined_df.to_csv(run_paths.issue_predictions_file, index=False)
        LOGGER.info("All predictions (combined) saved to %s", run_paths.issue_predictions_file)

    # Save metrics
    with run_paths.metrics_file.open("w", encoding="utf-8") as f:
        json.dump(aggregated_metrics, f, indent=2)
    LOGGER.info("Aggregated metrics saved to %s", run_paths.metrics_file)

    # Save epoch-wise results for all folds
    if all_epoch_results:
        epoch_results_file = run_paths.run_dir / "epoch_results.json"
        with epoch_results_file.open("w", encoding="utf-8") as f:
            json.dump(all_epoch_results, f, indent=2)
        LOGGER.info("Epoch-wise results saved to %s", epoch_results_file)

        # Also save as CSV for easier analysis
        epoch_df = pd.DataFrame(all_epoch_results)
        epoch_csv_file = run_paths.run_dir / "epoch_results.csv"
        epoch_df.to_csv(epoch_csv_file, index=False)
        LOGGER.info("Epoch-wise results CSV saved to %s", epoch_csv_file)

        # Compute average metrics across folds for each epoch and aggregation method
        LOGGER.info("\n--- Best Epoch by Aggregation Method (averaged across all folds) ---")

        # Group by epoch and compute mean metrics across all folds
        epoch_aggregated = epoch_df.groupby('epoch').mean(numeric_only=True)

        # For each aggregation strategy, find the best epoch
        best_epochs_summary = {}
        for agg in aggregation_strategies:
            f1_col = f"issue_f1_{agg}"
            if f1_col in epoch_aggregated.columns:
                best_epoch = epoch_aggregated[f1_col].idxmax()
                best_f1_mean = epoch_aggregated.loc[best_epoch, f1_col]

                # Calculate std across folds for the best epoch
                best_epoch_data = epoch_df[epoch_df['epoch'] == best_epoch]
                best_f1_std = best_epoch_data[f1_col].std()

                best_epochs_summary[f"{agg}_mean"] = {
                    "best_epoch": int(best_epoch),
                    "f1_mean": float(best_f1_mean),
                    "f1_std": float(best_f1_std),
                }

                LOGGER.info("  %-5s: Epoch %2d (F1: %.3f ± %.3f)",
                           agg, best_epoch, best_f1_mean, best_f1_std)

        # Save best epochs summary
        best_epochs_file = run_paths.run_dir / "best_epochs_by_aggregation.json"
        with best_epochs_file.open("w", encoding="utf-8") as f:
            json.dump(best_epochs_summary, f, indent=2)
        LOGGER.info("Best epochs summary saved to %s", best_epochs_file)

        # Also save epoch-aggregated metrics (mean across folds) as CSV
        epoch_aggregated_csv = run_paths.run_dir / "epoch_metrics_averaged.csv"
        epoch_aggregated.to_csv(epoch_aggregated_csv)
        LOGGER.info("Epoch-averaged metrics CSV saved to %s", epoch_aggregated_csv)

    # Save run config
    with run_paths.config_file.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    LOGGER.info("Run config saved to %s", run_paths.config_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a RoBERTa classifier on issue texts with chunking support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(DATASET_BASE_DIR),
        help="Path to classification JSON or a directory containing classification datasets.",
    )
    parser.add_argument(
        "--text-source",
        "-t",
        choices=sorted(VALID_TEXT_SOURCES),
        default="description",
        help="Which text field(s) to use from the dataset.",
    )
    parser.add_argument(
        "--model",
        "-m",
        # default="roberta-large",
        default="deroberta-v3-large",
        help="Hugging Face model name or path to fine-tune.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=256,
        help="Maximum number of words per chunked sub-issue.",
    )
    parser.add_argument(
        "--max-chunks-per-issue",
        type=int,
        default=None,
        help="Optional cap on chunks per issue to keep training sets compact.",
    )
    parser.add_argument(
        "--issue-aggregation",
        choices=["vote", "mean", "max", "first"],
        default="vote",
        help="How to collapse chunk predictions back to an issue label.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length (tokens) before truncation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay for AdamW regularization.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on validation loss (disabled by default).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of eval steps with no improvement before stopping (default: 5).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Warmup ratio for the scheduler.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=2,
        help="Number of times to repeat cross-validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for run artifacts. If omitted, auto-configures under Output/hf_roberta/<project>/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splitting and model initialization.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    normalized_source = normalize_text_source(args.text_source)
    args.text_source = normalized_source

    _seed_everything(args.seed)
    datasets = _resolve_data_paths(args.data_path)

    for data_path in datasets:
        _run_single_dataset(data_path, args)


if __name__ == "__main__":
    main()
