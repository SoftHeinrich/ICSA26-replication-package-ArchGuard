#!/usr/bin/env python3
"""
Zero-Shot LLM-Based Architectural Smell Classification

Usage:
    # Binary classification
    python -m src.classification.zero_shot_llm_classifier \
        --data data.json \
        --text-source description

    # Auto-run all classification datasets under a directory
    python -m src.classification.zero_shot_llm_classifier \
        data/classification \
        --output-dir Output/llm_runs

    # With chain-of-thought reasoning
    python -m src.classification.zero_shot_llm_classifier \
        --data data.json \
        --text-source combined \
        --use-cot
"""

import json
import os
import sys
import time
import argparse
import threading
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Session-level timestamp for consistent output folder naming across a run
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from openai import OpenAI

# Reuse text extraction from simple_ml
sys.path.append(str(Path(__file__).parent))
from simple_ml import (
    extract_texts_from_data,
    VALID_TEXT_SOURCES,
    TEXT_SOURCE_DESCRIPTION,
    normalize_text_source,
)

# Model alias map for common short-hands
MODEL_ALIASES = {
    "gpt5-mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5-pro": "gpt-5-pro",
    "gpt-5-pro": "gpt-5-pro",
    "gpt5-1": "gpt-5.1",
    "gpt-5.1": "gpt-5.1",
    "o4": "o4-mini",
    "o4-mini": "o4-mini",
}


# Architectural smell definitions
SMELL_DEFINITIONS_ARCAN = """
An architectural smell is a design problem indicating violations of fundamental design principles:
- Cyclic Dependency (CD): Two or more architectural components depend on each other directly or indirectly, breaking the desirable acyclic nature of the dependency structure
- Unstable Dependency (UD): A component depends on other components that are less stable than itself, according to the Instability metric
- Hub-Like Dependency (HL): An abstraction has (outgoing and incoming) dependencies with a large number of other abstractions
"""

SMELL_DEFINITIONS = """
An architectural smell is a design problem indicating violations of fundamental design principles:
- Big Dependency Cycle (BDC): Circular dependencies between components violating architectural layering
- Link Overload (LO): One component has too many dependencies
"""

DEFAULT_PROMPT_STYLE = "arcade"
PROMPT_DEFINITIONS = {
    "arcade": SMELL_DEFINITIONS,
    "arcan": SMELL_DEFINITIONS_ARCAN,
}

# Import dataset configuration from shared config
from dataset_config import DATASET_BASE_DIR, CLASSIFICATION_DATA_FILENAME


def _select_smell_definitions(prompt_style: str) -> str:
    """Return smell definitions for the requested prompt style."""
    style = (prompt_style or DEFAULT_PROMPT_STYLE).lower()
    definitions = PROMPT_DEFINITIONS.get(style)
    if not definitions:
        supported = ", ".join(sorted(PROMPT_DEFINITIONS))
        raise ValueError(f"Unsupported prompt style '{prompt_style}'. Choose from: {supported}")
    return definitions


def get_prompt(text: str, use_cot: bool = False, prompt_style: str = DEFAULT_PROMPT_STYLE) -> str:
    """
    Generate classification prompt.

    Args:
        text: Issue text (title + description + discussion)
        use_cot: Use chain-of-thought reasoning
        prompt_style: Which smell definitions to include (arcade or arcan)

    Returns:
        Prompt string
    """
    smell_definitions = _select_smell_definitions(prompt_style)

    if use_cot:
        prompt = f"""You are an expert software architect analyzing GitHub issues.

{smell_definitions}

Analyze this issue:

{text}

Think step-by-step:
1. What is the main topic?
2. Does it discuss architectural/structural problems?
3. Does it mention dependencies, coupling, or design principles?

Respond with:
REASONING: [your analysis]
CLASSIFICATION: SMELL or NO_SMELL"""
    else:
        prompt = f"""You are an expert software architect analyzing Gitlab issues.

{smell_definitions}

Issue:
{text}

Classify as SMELL or NO_SMELL based on if resolving this issue would cause architectural smells defined above.
Answer (SMELL or NO_SMELL):"""

    return prompt


def parse_response(response: str) -> str:
    """
    Parse LLM response to extract label.

    Returns:
        'smell' or 'no_smell'
    """
    response = response.strip().upper()

    # Handle chain-of-thought format
    if 'CLASSIFICATION:' in response:
        parts = response.split('CLASSIFICATION:')
        response = parts[1].strip()

    # Binary classification
    if 'SMELL' in response and 'NO_SMELL' not in response and 'NO SMELL' not in response:
        return 'smell'
    else:
        return 'no_smell'


def _extract_project_name(data_path: Path) -> str:
    """
    Extract project/system name from the data path.

    Assumes structure like data/classification/<project>/... or similar.
    Falls back to parent directory name.
    """
    # Try to find a meaningful project name from path components
    parts = data_path.parts

    # Look for common classification data directory patterns
    for i, part in enumerate(parts):
        if part in ('classification', 'data') and i + 1 < len(parts):
            # Return the next component after 'classification' or 'data'
            candidate = parts[i + 1]
            if candidate not in ('classification', 'data'):
                return candidate

    # Fallback: use parent directory name
    return data_path.parent.name


def _extract_detector_from_path(data_path: Path) -> str:
    """
    Extract detector name (acdc/arcan) from the data path.

    Looks for folder names containing '_acdc' or '_arcan' suffix.
    Returns empty string if no detector found.
    """
    parts = data_path.parts
    for part in parts:
        if part.endswith('_acdc'):
            return 'acdc'
        elif part.endswith('_arcan'):
            return 'arcan'
    return ''


def _build_default_cache_path(
    data_path: str,
    output_file: str,
    text_source: str,
    use_cot: bool,
    model: str,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> Path:
    """
    Derive a deterministic cache path that is unique per configuration.

    Auto-configures under Output/LLM_output/<timestamp>/<project_name>/<detector>/ by default.
    """
    data_path_obj = Path(data_path)

    if output_file:
        # Use output file's directory for cache
        base_dir = Path(output_file).parent
    else:
        # Auto-config: Output/LLM_output/<timestamp>/<project_name>/<detector>/
        project_name = _extract_project_name(data_path_obj)
        detector = _extract_detector_from_path(data_path_obj)
        if detector:
            base_dir = Path("Output/LLM_output") / _SESSION_TIMESTAMP / project_name / detector
        else:
            base_dir = Path("Output/LLM_output") / _SESSION_TIMESTAMP / project_name

    stem = data_path_obj.stem
    cot_suffix = "_cot" if use_cot else ""
    prompt_suffix = f"_{prompt_style}"
    safe_model = model.replace(":", "_")
    cache_name = f"{stem}_{text_source}{prompt_suffix}{cot_suffix}_{safe_model}_responses.json"
    return (base_dir / cache_name).resolve()


def _derive_output_path(
    data_path: Path,
    text_source: str,
    use_cot: bool,
    model: str,
    output_dir: Path = None,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> Path:
    """
    Derive a deterministic output CSV name when none is provided.

    Auto-configures output under Output/LLM_output/<timestamp>/<project_name>/<detector>/ by default.
    """
    if output_dir:
        base_dir = output_dir
    else:
        # Auto-config: Output/LLM_output/<timestamp>/<project_name>/<detector>/
        project_name = _extract_project_name(data_path)
        detector = _extract_detector_from_path(data_path)
        if detector:
            base_dir = Path("Output/LLM_output") / _SESSION_TIMESTAMP / project_name / detector
        else:
            base_dir = Path("Output/LLM_output") / _SESSION_TIMESTAMP / project_name

    cot_suffix = "_cot" if use_cot else ""
    prompt_suffix = f"_{prompt_style}"
    safe_model = model.replace(":", "_")
    file_name = f"{data_path.stem}_{text_source}{prompt_suffix}{cot_suffix}_{safe_model}_llm.csv"
    return (base_dir / file_name).resolve()


def _derive_metrics_path(output_file: Path | None, data_path: Path, text_source: str,
                         use_cot: bool, model: str, prompt_style: str) -> Path:
    """
    Derive a metrics JSON path near the output CSV (or default location).
    """
    prompt_suffix = f"_{prompt_style}"
    cot_suffix = "_cot" if use_cot else ""
    safe_model = model.replace(":", "_")
    if output_file:
        base_dir = output_file.parent
        stem = output_file.stem
    else:
        project_name = _extract_project_name(data_path)
        detector = _extract_detector_from_path(data_path)
        if detector:
            base_dir = Path("Output/LLM_output") / _SESSION_TIMESTAMP / project_name / detector
        else:
            base_dir = Path("Output/LLM_output") / _SESSION_TIMESTAMP / project_name
        stem = f"{data_path.stem}_{text_source}{prompt_suffix}{cot_suffix}_{safe_model}_llm"
    metrics_name = f"{stem}_metrics.json"
    return (base_dir / metrics_name).resolve()


def _resolve_data_paths(target: str, detector: str = None) -> list[Path]:
    """
    Accept either a single JSON file or a directory of classification datasets.
    When a directory is provided, gather plausible classification JSON files.

    Args:
        target: Path to JSON file or directory containing datasets
        detector: Optional filter for detector type ('acdc' or 'arcan')
    """
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if path.is_file():
        # For single file, check detector filter if specified
        if detector:
            detected = _extract_detector_from_path(path)
            if detected and detected != detector:
                raise ValueError(f"File is for detector '{detected}', not '{detector}': {path}")
        return [path]

    # Only match the exact classification data filename
    unique: list[Path] = []
    for item in path.rglob(CLASSIFICATION_DATA_FILENAME):
        if item.is_file():
            # Filter by detector if specified
            if detector:
                detected = _extract_detector_from_path(item)
                if detected and detected != detector:
                    continue
            unique.append(item)

    if not unique:
        filter_msg = f" for detector '{detector}'" if detector else ""
        raise ValueError(f"No classification datasets found{filter_msg} under directory: {path}")

    print(f"Resolved {len(unique)} classification dataset(s) from {path}" +
          (f" (filtered to detector: {detector})" if detector else ""))
    return unique


def _normalize_model_name(model_name: str) -> str:
    """Normalize model aliases to fully-qualified names."""
    if not model_name:
        return model_name
    return MODEL_ALIASES.get(model_name.strip().lower(), model_name)


def _extract_response_text(response) -> str:
    """
    Extract text content from either the Responses API object or a chat completion.
    """
    # Responses API shape
    try:
        output = getattr(response, "output", None)
        if output:
            chunks = []
            for item in output:
                for block in getattr(item, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)
    except Exception:
        pass

    # Fallback to chat completions-style response
    try:
        return response.choices[0].message.content
    except Exception:
        return ""


def _load_text_source_config(config_path: Optional[str]) -> dict[str, str]:
    """Load optional text-source overrides from a JSON file."""
    if not config_path:
        return {}

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"text-source config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("text-source config must be a JSON object of key->text_source mappings.")

    normalized = {}
    for key, value in payload.items():
        normalized[key.lower()] = normalize_text_source(value)
    return normalized


def _load_model_config(config_path: Optional[str]) -> dict[str, str]:
    """Load optional model overrides from a JSON file."""
    if not config_path:
        return {}

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"model config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("model config must be a JSON object of key->model mappings.")

    normalized = {}
    for key, value in payload.items():
        normalized[key.lower()] = _normalize_model_name(value)
    return normalized


def _select_text_source(dataset_path: Path, default_text_source: str, config_map: dict[str, str]) -> str:
    """
    Choose text_source for a dataset using config overrides when available.

    Config keys match (case-insensitive) against:
    - full path string
    - dataset filename
    - dataset stem
    - parent directory name
    - optional special key '__default__'
    """
    candidates = [
        str(dataset_path),
        dataset_path.name,
        dataset_path.stem,
        dataset_path.parent.name,
    ]
    lower_map = {k.lower(): v for k, v in config_map.items()}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]

    if "__default__" in lower_map:
        return lower_map["__default__"]

    return normalize_text_source(default_text_source)


def _select_model(dataset_path: Path, default_model: str, config_map: dict[str, str]) -> str:
    """
    Choose model for a dataset using config overrides when available.
    """
    candidates = [
        str(dataset_path),
        dataset_path.name,
        dataset_path.stem,
        dataset_path.parent.name,
    ]
    lower_map = {k.lower(): v for k, v in config_map.items()}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return _normalize_model_name(lower_map[key])

    if "__default__" in lower_map:
        return _normalize_model_name(lower_map["__default__"])

    return _normalize_model_name(default_model)


def classify_with_llm(
    text: str,
    client: OpenAI,
    model: str = 'gpt-4o-mini',
    use_cot: bool = False,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> tuple:
    """
    Classify single text using OpenAI.

    Args:
        text: Issue text
        client: OpenAI client
        model: Model name
        use_cot: Use chain-of-thought
        prompt_style: Prompt variant to use (arcade or arcan)

    Returns:
        (predicted_label, raw_response, prompt) tuple
    """
    prompt = get_prompt(text, use_cot, prompt_style)

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are an expert software architect."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=300,
        )

        response_text = _extract_response_text(response)
        predicted_label = parse_response(response_text)
        return predicted_label, response_text, prompt

    except Exception as e:
        print(f"  Error: {e}")
        return 'no_smell', f"Error: {e}", prompt


def classify_batch_with_llm(
    batch_items,
    client: OpenAI,
    model: str = 'gpt-4o-mini',
    use_cot: bool = False,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
):
    """
    Classify a batch of texts in a single request.

    Args:
        batch_items: List of dicts with keys {'index', 'text', 'issue_id'}
        client: OpenAI client
        model: Model name
        use_cot: Use chain-of-thought
        prompt_style: Prompt variant to use (arcade or arcan)

    Returns:
        List of dicts with keys {'index', 'predicted_label', 'llm_response', 'prompt'}
    """
    issues_block = "\n\n".join(
        f"[{item['index']}] {item['text']}" for item in batch_items
    )

    instructions = (
        "Classify each issue as SMELL or NO_SMELL.\n"
        "Return ONLY a JSON array of objects with keys: index (int), classification (SMELL/NO_SMELL), "
        "and optional reasoning.\n"
        "Example: [{\"index\": 0, \"classification\": \"NO_SMELL\", \"reasoning\": \"...\"}]"
    )

    if use_cot:
        instructions += "\nProvide brief reasoning per item in the JSON 'reasoning' field."

    prompt = f"""You are an expert software architect analyzing multiple GitHub issues.

{_select_smell_definitions(prompt_style)}

{instructions}

Issues:
{issues_block}
"""

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are an expert software architect."},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = _extract_response_text(response)
    except Exception as e:
        print(f"  Batch error: {e}")
        return None

    # Parse JSON output
    try:
        parsed = json.loads(response_text)
    except Exception:
        # Sometimes wrapped in code fences
        cleaned = response_text.strip().strip("`")
        cleaned = cleaned.replace("json\n", "").replace("\njson", "")
        try:
            parsed = json.loads(cleaned)
        except Exception:
            print("  Failed to parse batch response; falling back to single calls.")
            return None

    results = []
    for entry in parsed:
        idx = entry.get('index')
        classification = entry.get('classification', '')
        predicted_label = parse_response(str(classification))
        results.append({
            'index': idx,
            'predicted_label': predicted_label,
            'llm_response': json.dumps(entry),
            'prompt': prompt
        })

    # Preserve original order for indices we asked for
    results_by_index = {r['index']: r for r in results}
    ordered_results = []
    for item in batch_items:
        ordered_results.append(results_by_index.get(item['index']))

    return ordered_results


def run_zero_shot_classification(
    data_path: str,
    text_source: str = 'combined',
    model: str = 'gpt-4o-mini',
    use_cot: bool = False,
    api_delay: float = 1.0,
    output_file: str = None,
    responses_cache: str = None,
    cache_only: bool = False,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    max_workers: int = 16,
) -> pd.DataFrame:
    """
    Run zero-shot classification on dataset.

    Args:
        data_path: Path to JSON data file
        text_source: 'description', 'discussion', or 'combined'
        model: OpenAI model name
        use_cot: Use chain-of-thought prompting
        api_delay: Delay between API calls (seconds)
        output_file: Path to save results CSV
        responses_cache: Path to cache/save detailed responses (JSON)
        cache_only: Do not call OpenAI; rely solely on cache
        prompt_style: Prompt variant to use (arcade or arcan)
        max_workers: Max concurrent API workers for parallel processing

    Returns:
        DataFrame with results and evaluation metrics
    """
    prompt_style = (prompt_style or DEFAULT_PROMPT_STYLE).lower()
    _select_smell_definitions(prompt_style)  # Validate early
    max_workers = max(1, int(max_workers))

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if "version" in df.columns:
        mask_not_yet = df["version"].astype(str).str.contains("not yet", case=False, na=False)
        if mask_not_yet.any():
            skipped = int(mask_not_yet.sum())
            print(f"Skipping {skipped} issue(s) with version 'not yet'")
            df = df.loc[~mask_not_yet].reset_index(drop=True)

    print(f"Loaded {len(df)} samples")
    print(f"True label distribution: {df['label'].value_counts().to_dict()}")

    # Extract texts
    print(f"\nExtracting texts (source: {text_source})...")
    texts = extract_texts_from_data(df, text_source=text_source)

    # Prepare cache paths
    if responses_cache:
        responses_file = Path(responses_cache).resolve()
    else:
        responses_file = _build_default_cache_path(
            data_path=data_path,
            output_file=output_file,
            text_source=text_source,
            use_cot=use_cot,
            model=model,
            prompt_style=prompt_style,
        )

    cached_responses = {}
    if responses_file.exists():
        print(f"\nLoading cached responses from {responses_file}...")
        with open(responses_file, 'r', encoding='utf-8') as f:
            cached_responses = json.load(f)
        print(f"Loaded {len(cached_responses)} cached items")

    # Initialize OpenAI client only if we need to call the API
    client = None
    if not cache_only:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
    else:
        if cached_responses:
            print("Cache-only mode enabled; reusing cached predictions")
        else:
            raise ValueError("Cache-only mode requires an existing responses cache.")

    # Classify
    print(f"\nClassifying with {model} (CoT: {use_cot}, prompt: {prompt_style}, workers: {max_workers})...")
    predictions = []
    responses = []
    prompts = []

    total = len(texts)
    predictions = [None] * total
    responses = [None] * total
    prompts = [None] * total

    # Fill from cache
    for i in range(total):
        issue_id = df.iloc[i]['id'] if 'id' in df.columns else i
        issue_key = str(issue_id)
        if issue_key in cached_responses:
            cached = cached_responses[issue_key]
            predictions[i] = cached.get('predicted_label')
            responses[i] = cached.get('llm_response', '')
            prompts[i] = cached.get('llm_prompt', '')
            print(f"  [{i+1}/{total}] Issue {issue_id}... {predictions[i]} (cached)")

    # Identify remaining items to classify
    to_classify = [i for i, p in enumerate(predictions) if p is None]

    if cache_only and to_classify:
        missing = [str(df.iloc[i]['id'] if 'id' in df.columns else i) for i in to_classify]
        raise ValueError(f"Cache-only mode enabled but missing cache for issues: {', '.join(missing[:10])}"
                         f"{' ...' if len(missing) > 10 else ''}")

    class _RateLimiter:
        def __init__(self, min_interval: float):
            self.min_interval = max(0.0, float(min_interval))
            self._lock = threading.Lock()
            self._next_time = 0.0

        def wait(self):
            if self.min_interval <= 0:
                return
            with self._lock:
                now = time.time()
                delay = max(0.0, self._next_time - now)
                self._next_time = max(now, self._next_time) + self.min_interval
            if delay > 0:
                time.sleep(delay)

    rate_limiter = _RateLimiter(api_delay)
    updates_lock = threading.Lock()

    def apply_update(update):
        """Apply a single classification result."""
        with updates_lock:
            idx = update['index']
            issue_id = update['issue_id']
            predictions[idx] = update['predicted_label']
            responses[idx] = update['llm_response']
            prompts[idx] = update['llm_prompt']
            cached_responses[str(issue_id)] = {
                'issue_id': issue_id,
                'true_label': update['true_label'],
                'predicted_label': update['predicted_label'],
                'correct': update['predicted_label'] == update['true_label'],
                'llm_prompt': update['llm_prompt'],
                'llm_response': update['llm_response'],
                'input_text': update['input_text'],
            }

    def classify_single_issue(idx):
        """Classify a single issue with rate limiting."""
        issue_id = df.iloc[idx]['id'] if 'id' in df.columns else idx
        true_label = df.iloc[idx]['label']
        text = texts[idx]

        rate_limiter.wait()

        pred_label, response_text, prompt = classify_with_llm(
            text, client, model, use_cot, prompt_style
        )

        update = {
            'index': idx,
            'issue_id': issue_id,
            'predicted_label': pred_label,
            'llm_response': response_text,
            'llm_prompt': prompt,
            'true_label': true_label,
            'input_text': text[:500] + '...' if len(text) > 500 else text,
        }

        apply_update(update)
        print(f"  [{idx+1}/{total}] Issue {issue_id}... {pred_label} (true: {true_label}) "
              f"{'[OK]' if pred_label == true_label else '[X]'}")

    # Process with parallel workers or sequentially
    if max_workers == 1:
        for idx in to_classify:
            classify_single_issue(idx)
    else:
        print(f"\nProcessing {len(to_classify)} issues with {max_workers} worker(s)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(classify_single_issue, idx): idx for idx in to_classify}
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions that occurred

    # Add predictions to dataframe
    df['predicted_label'] = predictions
    df['llm_response'] = responses
    df['llm_prompt'] = prompts
    df['correct'] = df['predicted_label'] == df['label']

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    y_true = df['label'].values
    y_pred = df['predicted_label'].values

    # Convert to numeric labels (same as simple_ml)
    # smell=1, no_smell=0
    y_true_num = np.array([1 if y == 'smell' else 0 for y in y_true])
    y_pred_num = np.array([1 if y == 'smell' else 0 for y in y_pred])

    # Overall metrics (same as simple_ml)
    accuracy = accuracy_score(y_true_num, y_pred_num)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_num, y_pred_num, labels=[1], zero_division=0
    )
    # Extract scalar values from arrays
    precision = float(precision[0])
    recall = float(recall[0])
    f1 = float(f1[0])

    print(f"\nOverall:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

    # Confusion matrix (manual computation)
    tp = int(np.sum((y_pred_num == 1) & (y_true_num == 1)))
    fp = int(np.sum((y_pred_num == 1) & (y_true_num == 0)))
    fn = int(np.sum((y_pred_num == 0) & (y_true_num == 1)))
    tn = int(np.sum((y_pred_num == 0) & (y_true_num == 0)))

    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}  FP: {fp}")
    print(f"  FN: {fn}  TN: {tn}")

    print(f"\nLabel Distribution:")
    print(f"  Predicted: smell={np.sum(y_pred_num == 1)}, no_smell={np.sum(y_pred_num == 0)}")
    print(f"  True:      smell={np.sum(y_true_num == 1)}, no_smell={np.sum(y_true_num == 0)}")

    # Save results
    if cached_responses:
        responses_file.parent.mkdir(parents=True, exist_ok=True)
        # Persist merged cache with latest correctness flags
        for idx, row in df.iterrows():
            issue_key = str(row['id']) if 'id' in df.columns else str(idx)
            cached_responses[issue_key] = {
                'issue_id': row['id'] if 'id' in df.columns else idx,
                'true_label': row['label'],
                'predicted_label': row['predicted_label'],
                'correct': bool(row['correct']),
                'llm_prompt': df.iloc[idx]['llm_prompt'],
                'llm_response': df.iloc[idx]['llm_response'],
                'input_text': texts[idx][:500] + '...' if len(texts[idx]) > 500 else texts[idx]
            }

        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(cached_responses, f, indent=2, ensure_ascii=False)
        print(f"\nCached responses saved to {responses_file}")

    if output_file:
        # Save CSV with basic results
        df_output = df[['id', 'label', 'predicted_label', 'correct']].copy()
        df_output.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        metrics_path = _derive_metrics_path(Path(output_file), data_path, text_source, use_cot, model, prompt_style)
    else:
        metrics_path = _derive_metrics_path(None, Path(data_path), text_source, use_cot, model, prompt_style)

    # Return summary
    results_summary = {
        'model': model,
        'text_source': text_source,
        'use_cot': use_cot,
        'prompt_style': prompt_style,
        'total_samples': len(df),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results_summary, handle, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return df, results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM classification for architectural smells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m src.classification.zero_shot_llm_classifier --data data.json

  # Use combined text with chain-of-thought
  python -m src.classification.zero_shot_llm_classifier \
      --data data.json --text-source combined --use-cot

  # With parallel workers and rate limiting
  python -m src.classification.zero_shot_llm_classifier \
      --data data.json --max-workers 8 --api-delay 0.5
        """
    )

    parser.add_argument(
        'data_path',
        nargs='?',
        default=str(DATASET_BASE_DIR),
        help='Path to JSON data file or directory containing classification datasets (default: %(default)s)'
    )

    parser.add_argument(
        '--text-source', '-t',
        choices=sorted(VALID_TEXT_SOURCES),
        default='combined',
        help='Text source for classification: description, combined (filtered discussion), combined_unfiltered (full discussion)'
    )

    parser.add_argument(
        '--model', '-m',
        default='gpt-5-mini',
        help='OpenAI model name (default: %(default)s)'
    )

    parser.add_argument(
        '--use-cot',
        action='store_true',
        help='Use chain-of-thought prompting'
    )
    parser.add_argument(
        '--prompt-style',
        default=DEFAULT_PROMPT_STYLE,
        choices=sorted(PROMPT_DEFINITIONS),
        help='Prompt variant to use (arcade or arcan)'
    )

    parser.add_argument(
        '--api-delay', '-d',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=32,
        help='Max concurrent API workers for parallel processing (default: 4)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to place auto-derived outputs when processing directories of datasets'
    )
    parser.add_argument(
        '--responses-cache',
        help='Path to cache/save raw LLM responses (JSON). Defaults to <output>_responses.json or <data>_responses.json'
    )
    parser.add_argument(
        '--model-config',
        help='JSON file mapping dataset identifiers to model overrides (keys: path/stem/parent, optional "__default__")'
    )
    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='Do not call OpenAI; reuse cached responses only'
    )
    parser.add_argument(
        '--text-source-config',
        help='JSON file mapping dataset identifiers to text_source overrides (keys: path/stem/parent, optional "__default__")'
    )
    parser.add_argument(
        '--detector',
        choices=['acdc', 'arcan'],
        help='Filter to only process datasets for a specific detector (acdc or arcan)'
    )

    args = parser.parse_args()

    print("Zero-Shot LLM Architectural Smell Classification")
    print("="*60)

    data_targets = _resolve_data_paths(args.data_path, detector=args.detector)
    text_source_overrides = _load_text_source_config(args.text_source_config)
    model_overrides = _load_model_config(args.model_config)

    if len(data_targets) > 1 and args.output:
        raise ValueError("Cannot use --output with multiple datasets; use --output-dir or rely on defaults.")
    if len(data_targets) > 1 and args.responses_cache:
        raise ValueError("Cannot use a single --responses-cache for multiple datasets; omit to auto-derive per file.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    for idx, data_file in enumerate(data_targets, start=1):
        resolved_text_source = _select_text_source(
            dataset_path=data_file,
            default_text_source=args.text_source,
            config_map=text_source_overrides,
        )
        resolved_model = _select_model(
            dataset_path=data_file,
            default_model=args.model,
            config_map=model_overrides,
        )

        # Auto-detect prompt style from dataset path if using default
        resolved_prompt_style = args.prompt_style
        if args.prompt_style == DEFAULT_PROMPT_STYLE:
            detector = _extract_detector_from_path(data_file)
            if detector in PROMPT_DEFINITIONS:
                resolved_prompt_style = detector

        derived_output = args.output or _derive_output_path(
            data_path=data_file,
            text_source=resolved_text_source,
            use_cot=args.use_cot,
            model=resolved_model,
            output_dir=output_dir,
            prompt_style=resolved_prompt_style,
        )

        print(f"\nProcessing dataset {idx}/{len(data_targets)}: {data_file}")
        print(f"  Text source: {resolved_text_source}")
        print(f"  Model:       {resolved_model}")
        print(f"  Prompt:      {resolved_prompt_style}")
        run_zero_shot_classification(
            data_path=str(data_file),
            text_source=resolved_text_source,
            model=resolved_model,
            use_cot=args.use_cot,
            api_delay=args.api_delay,
            output_file=str(derived_output) if derived_output else None,
            responses_cache=args.responses_cache,
            cache_only=args.cache_only,
            prompt_style=resolved_prompt_style,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
