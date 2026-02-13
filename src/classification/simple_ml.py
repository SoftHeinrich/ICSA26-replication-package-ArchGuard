"""
Simple Traditional ML with Imbalance Learning
A straightforward approach combining traditional ML models with different imbalance strategies.
"""
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, TYPE_CHECKING

# Traditional ML models - without built-in class weighting
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Imbalance learning strategies
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
IMBLEARN_AVAILABLE = True
from sklearn.base import clone
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

TEXT_SOURCE_DESCRIPTION = "description"  # title + description
TEXT_SOURCE_COMBINED_FILTERED = "combined"  # title + description + discussion (pre-MR filtered)
TEXT_SOURCE_COMBINED_UNFILTERED = "combined_unfiltered"  # title + description + full discussion

VALID_TEXT_SOURCES = {
    TEXT_SOURCE_DESCRIPTION,
    TEXT_SOURCE_COMBINED_FILTERED,
    TEXT_SOURCE_COMBINED_UNFILTERED,
}

from xgboost import XGBClassifier
import re

# Import NLTK for lemmatization (required dependency)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as e:
    raise ImportError(
        "NLTK is required for text preprocessing. Install with: pip install nltk"
    ) from e

# Download required NLTK data (only happens once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def clean_text_base(text: str) -> str:
    """
    Clean text without lemmatization - suitable for SBERT and other embeddings.

    Removes URLs, email addresses, code blocks, special characters, and extra whitespace.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string (without lemmatization)
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove code blocks (markdown style)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]*`', ' ', text)

    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_text_for_tfidf(text: str) -> str:
    """
    Clean text for TF-IDF encoding with lemmatization.

    Args:
        text: Raw text string

    Returns:
        Cleaned and lemmatized text string
    """
    # First apply base cleaning
    text = clean_text_base(text)

    if not text:
        return ""

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(lemmatized_tokens)

    return text


def _select_xgb_device() -> str:
    """Prefer GPU when available while remaining safe to run on CPU-only hosts."""
    forced_device = os.getenv("XGB_DEVICE", "").strip().lower()
    if forced_device in {"cpu", "cuda", "gpu"}:
        if forced_device in {"cuda", "gpu"} and not _xgb_supports_cuda():
            print("XGB_DEVICE requested CUDA but xgboost lacks GPU support; using CPU.")
            return "cpu"
        return "cuda" if forced_device in {"cuda", "gpu"} else "cpu"

    if _xgb_supports_cuda():
        try:
            import torch  # Local import to avoid hard dependency for non-SBERT paths
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            return "cpu"

    return 'cpu'


def _xgb_supports_cuda() -> bool:
    """Detect whether the installed xgboost build supports CUDA."""
    try:
        from xgboost.core import _has_cuda_support  # type: ignore
        return bool(_has_cuda_support())
    except Exception:
        return False


def _select_torch_device() -> str:
    """Prefer CUDA-capable device for SBERT, otherwise fall back to CPU."""
    forced_device = os.getenv("SBERT_DEVICE", "").strip().lower()
    if forced_device in {"cpu", "cuda"}:
        return forced_device

    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass

    return 'cpu'


def _load_sentence_transformer():
    """Lazily import SentenceTransformer to avoid heavy initialization during unrelated code paths."""
    from sentence_transformers import SentenceTransformer  # Local import to dodge OMP shmem issues when unused
    return SentenceTransformer


def _clear_cuda_cache():
    """Clear CUDA cache to free GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass



def get_base_models(encoding_method='tfidf'):
    """Get base traditional ML models optimized for positive smell detection."""
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, 
            max_iter=2000,
            C=0.1,  # Lower regularization for better recall
            solver='liblinear'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True
        ),
        # 'gradient_boosting': GradientBoostingClassifier(
        #     random_state=42,
        #     n_estimators=200,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     min_samples_split=5,
        #     min_samples_leaf=2
        # ),
        'svm': SVC(
            random_state=42,
            probability=True,
            C=1.0,  # Balanced regularization
            kernel='rbf',  # RBF kernel for text data
            gamma='scale'
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='cosine'
        )
    }
    
    # Choose appropriate Naive Bayes variant based on encoding
    if encoding_method == 'tfidf':
        models['naive_bayes'] = MultinomialNB(alpha=0.1)  # For sparse, non-negative features
    elif encoding_method.startswith('sbert_') or encoding_method == 'sbert' or encoding_method == 'openai':
        models['naive_bayes'] = GaussianNB()  # For dense, continuous features (SBERT, OpenAI)
    else:
        models['naive_bayes'] = GaussianNB()  # Default to GaussianNB for other embeddings

    # XGBoost (supports sparse and dense features)
    xgb_device = _select_xgb_device()
    xgb_predictor = 'gpu_predictor' if xgb_device == 'cuda' else 'cpu_predictor'
    xgb_tree_method = 'gpu_hist' if xgb_device == 'cuda' else 'hist'
    if xgb_device == 'cuda':
        print("Enabling XGBoost GPU acceleration (device=cuda)")
    if os.getenv("DISABLE_XGBOOST", "").strip().lower() not in {"1", "true", "yes"}:
        models['xgboost'] = XGBClassifier(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='mlogloss',
            tree_method=xgb_tree_method,
            device=xgb_device,
            predictor=xgb_predictor,
            n_jobs=8,
        )

    # Simple feed-forward network (only when features are dense)
    if encoding_method != 'tfidf':
        models['feedforward_nn'] = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            learning_rate='adaptive',
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
            verbose=False
        )

    return models


def get_imbalance_strategies():
    """Get available imbalance handling strategies."""
    strategies = {
        # 'none': None,  # No imbalance handling
        # 'class_weight': 'class_weight'  # Use sklearn's class_weight='balanced'
    }
    
    if IMBLEARN_AVAILABLE:
        strategies.update({
            # 'smote': SMOTE(random_state=42),
            # 'borderline_smote': BorderlineSMOTE(random_state=42),
            # 'adasyn': ADASYN(random_state=42),
            'random_undersample': RandomUnderSampler(random_state=42),
            # 'smote_enn': SMOTEENN(
            #     smote=SMOTE(random_state=42, k_neighbors=3),  # Reduce neighbors for less aggressive oversampling
            #     enn=EditedNearestNeighbours(n_neighbors=3, kind_sel='mode'),  # Less aggressive undersampling
            #     random_state=42
            # ),
            # 'smote_tomek': SMOTETomek(random_state=42)
        })
    
    return strategies


def apply_imbalance_strategy(X_train, y_train, strategy_name, strategy):
    """Apply imbalance strategy to training data."""
    if strategy_name == 'none':
        return X_train, y_train
    
    elif strategy_name == 'class_weight':
        # Return original data, class weighting will be applied to model
        return X_train, y_train
    
    elif strategy is not None and IMBLEARN_AVAILABLE:
        try:
            X_resampled, y_resampled = strategy.fit_resample(X_train, y_train)
            print(f"  {strategy_name}: {len(y_train)} -> {len(y_resampled)} samples")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"  {strategy_name} failed: {e}, using original data")
            return X_train, y_train
    
    else:
        return X_train, y_train


def configure_model_for_strategy(model, model_name, strategy_name):
    """Configure model based on the imbalance strategy."""
    if strategy_name == 'class_weight':
        # Apply class weighting to supported models
        if hasattr(model, 'class_weight'):
            model.set_params(class_weight='balanced')
        else:
            print(f"  Warning: {model_name} doesn't support class_weight, using original model")
    
    return model


def compute_balanced_sample_weights(y_train):
    """Compute per-sample weights using sklearn's balanced class weights."""
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weight_lookup = {cls: weight for cls, weight in zip(classes, class_weights)}
    return np.array([weight_lookup[label] for label in y_train])


def normalize_text_source(text_source: str) -> str:
    """Validate and normalize text_source to the supported set."""
    candidate = (text_source or TEXT_SOURCE_DESCRIPTION).strip().lower()
    if candidate not in VALID_TEXT_SOURCES:
        valid = ", ".join(sorted(VALID_TEXT_SOURCES))
        raise ValueError(f"Unsupported text_source '{text_source}'. Valid options: {valid}")
    return candidate


def _combine_title_and_description(title: str, description: str) -> str:
    """Combine title and description while avoiding repeated titles."""
    title = title or ''
    description = description or ''

    normalized_title = title.strip()
    normalized_description = description.strip()

    if normalized_title and normalized_description.lower().startswith(normalized_title.lower()):
        return normalized_description or normalized_title

    if normalized_title and normalized_description:
        return f"{normalized_title}\n\n{normalized_description}"
    return normalized_title or normalized_description


def _select_discussion_text(row: pd.Series | dict, prefer_unfiltered: bool) -> str:
    """Pick discussion text, preferring unfiltered content when requested."""
    raw = None
    if hasattr(row, 'get'):
        raw = row.get('discussion_unfiltered') or row.get('discussion_full') or row.get('discussion_all')
    filtered = row.get('discussion') if hasattr(row, 'get') else None

    if prefer_unfiltered and raw:
        return raw
    return filtered or raw or ''


def calculate_top_k_positive_count(y_true, y_pred_proba, k):
    """Calculate number of positive smell samples correctly predicted in Top-K."""
    if y_pred_proba is None:
        return 0, 0
    
    # Find positive samples (assuming label 0 is "no smell" and others are positive smells)
    positive_mask = y_true != 0
    positive_indices = np.where(positive_mask)[0]
    
    if len(positive_indices) == 0:
        return 0, 0
    
    # Get the top-k predicted classes for positive samples only
    top_k_predictions = np.argsort(y_pred_proba[positive_indices], axis=1)[:, -k:]
    y_true_positive = y_true[positive_indices]
    
    # Check if true label is in top-k predictions for positive samples
    correct = 0
    for i, true_label in enumerate(y_true_positive):
        if true_label in top_k_predictions[i]:
            correct += 1
    
    return correct, len(y_true_positive)


def evaluate_model_cv(model, X, y, model_name, strategy_name, encoding_name, n_folds=7):
    """Evaluate model using stratified cross-validation and return metrics."""
    # Use StratifiedKFold to maintain class distribution in each fold
    skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        # Handle both sparse and dense matrices
        if hasattr(X, 'toarray'):  # Sparse matrix
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        else:  # Dense matrix
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Clone model for this fold
        from sklearn.base import clone
        fold_model = clone(model)

        # Configure model for strategy
        fold_model = configure_model_for_strategy(fold_model, model_name, strategy_name)

        # Apply imbalance strategy to training data
        X_train_processed, y_train_processed = apply_imbalance_strategy(
            X_train_fold, y_train_fold, strategy_name,
            get_imbalance_strategies().get(strategy_name)
        )

        sample_weight = None
        if strategy_name == 'class_weight':
            sample_weight = compute_balanced_sample_weights(y_train_processed)

        fit_kwargs = {}
        if sample_weight is not None and 'sample_weight' in inspect.signature(fold_model.fit).parameters:
            fit_kwargs['sample_weight'] = sample_weight

        fold_model.fit(X_train_processed, y_train_processed, **fit_kwargs)
        y_pred = fold_model.predict(X_test_fold)
        y_pred_proba = fold_model.predict_proba(X_test_fold) if hasattr(fold_model, 'predict_proba') else None

        # Calculate metrics for this fold
        accuracy = accuracy_score(y_test_fold, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_fold, y_pred, labels=[1])

        # Positive smell recall
        positive_mask = y_test_fold != 0
        num_positive_samples = np.sum(positive_mask)
        if num_positive_samples > 0:
            positive_recall = np.sum((y_pred[positive_mask] != 0) & (y_pred[positive_mask] == y_test_fold[positive_mask])) / num_positive_samples
        else:
            positive_recall = 0.0

        # Top-K metrics
        top_k_metrics = {}
        if y_pred_proba is not None:
            for k in [1, 5, 10]:
                top_k_count, total_positive = calculate_top_k_positive_count(y_test_fold, y_pred_proba, k)
                top_k_metrics[f'top_{k}_positive_count'] = int(top_k_count)

        fold_result = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_smell_recall': positive_recall,
            'num_positive_samples': int(num_positive_samples),
            **top_k_metrics
        }
        fold_results.append(fold_result)

    # Calculate mean and std across folds
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'positive_smell_recall']
    cv_results = {
        'model': model_name,
        'strategy': strategy_name,
        'encoding': encoding_name,
        'cv_folds': n_folds
    }

    for metric in metrics:
        values = [result[metric] for result in fold_results]
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)

    return cv_results


def evaluate_model_repeated_cv(
    model, X, y, model_name, encoding_name, strategies,
    n_folds=5, n_repeats=2, random_state=42
):
    """
    Evaluate model using repeated stratified k-fold cross-validation with significance testing.

    This provides multiple independent test scores (n_folds × n_repeats) for robust evaluation
    and statistical significance testing against a random baseline.

    Args:
        model: sklearn-compatible model
        X: Feature matrix (sparse or dense) - ALREADY ENCODED
        y: Labels array
        model_name: Name of the model
        encoding_name: Name of the encoding method
        strategies: Dict of strategy_name -> strategy object
        n_folds: Number of folds per repetition (default: 5)
        n_repeats: Number of repetitions (default: 2)
        random_state: Random seed for reproducibility

    Returns:
        List of dicts, one per strategy, with repeated CV results including significance testing
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.dummy import DummyClassifier
    from scipy.stats import wilcoxon

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    results = []
    total_trials = n_folds * n_repeats

    for strategy_name, strategy in strategies.items():
        print(f"\n  Strategy: {strategy_name} - Running {n_repeats} × {n_folds}-fold CV ({total_trials} trials)")

        # Setup repeated stratified k-fold
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)

        # Storage for all trial results
        model_accuracies = []
        model_precisions = []
        model_recalls = []
        model_f1s = []
        random_f1s = []

        # Random baseline
        random_baseline = DummyClassifier(strategy='stratified', random_state=random_state)

        trial_num = 0
        for train_idx, test_idx in rskf.split(X, y):
            trial_num += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ================================================================
            # Evaluate actual model
            # ================================================================
            fold_model = clone(model)
            fold_model = configure_model_for_strategy(fold_model, model_name, strategy_name)

            # Apply imbalance strategy
            X_train_proc, y_train_proc = apply_imbalance_strategy(
                X_train, y_train, strategy_name, strategy
            )

            # Compute sample weights if needed
            sample_weight = None
            if strategy_name == 'class_weight':
                sample_weight = compute_balanced_sample_weights(y_train_proc)

            fit_kwargs = {}
            if sample_weight is not None and 'sample_weight' in inspect.signature(fold_model.fit).parameters:
                fit_kwargs['sample_weight'] = sample_weight

            fold_model.fit(X_train_proc, y_train_proc, **fit_kwargs)
            y_pred = fold_model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[1])
            precision = precision[0]
            recall = recall[0]
            f1 = f1[0]

            # Store all metrics
            model_accuracies.append(float(accuracy))
            model_precisions.append(float(precision))
            model_recalls.append(float(recall))
            model_f1s.append(float(f1))

            # ================================================================
            # Evaluate random baseline
            # ================================================================
            random_fold = clone(random_baseline)
            random_fold.fit(X_train, y_train)
            y_pred_random = random_fold.predict(X_test)

            _, _, random_f1, _ = precision_recall_fscore_support(y_test, y_pred_random, labels=[1])
            random_f1s.append(float(random_f1[0]))

            if trial_num % max(1, total_trials // 5) == 0:
                print(f"    Trial {trial_num}/{total_trials}: Acc={float(accuracy):.3f}, P={float(precision):.3f}, R={float(recall):.3f}, F1={float(f1):.3f}")

        # ================================================================
        # Aggregate results
        # ================================================================
        model_accuracy_array = np.array(model_accuracies)
        model_precision_array = np.array(model_precisions)
        model_recall_array = np.array(model_recalls)
        model_f1_array = np.array(model_f1s)
        random_f1_array = np.array(random_f1s)

        # Compute mean and std for all metrics
        accuracy_mean = float(np.mean(model_accuracy_array))
        accuracy_std = float(np.std(model_accuracy_array))
        precision_mean = float(np.mean(model_precision_array))
        precision_std = float(np.std(model_precision_array))
        recall_mean = float(np.mean(model_recall_array))
        recall_std = float(np.std(model_recall_array))
        f1_mean = float(np.mean(model_f1_array))
        f1_std = float(np.std(model_f1_array))
        random_f1_mean = float(np.mean(random_f1_array))
        random_f1_std = float(np.std(random_f1_array))
        f1_improvement = f1_mean - random_f1_mean

        # Statistical significance testing
        # Only consider significant if BOTH improvement > 0 AND p < 0.05
        if np.allclose(model_f1_array, random_f1_array):
            wilcoxon_p = 1.0
        else:
            try:
                _, wilcoxon_p = wilcoxon(model_f1_array, random_f1_array, alternative='greater')
            except ValueError:
                wilcoxon_p = 1.0

        significant = (f1_improvement > 0) and (wilcoxon_p < 0.05)

        # Compute effect size (Cohen's d)
        mean_diff = f1_mean - random_f1_mean
        pooled_std = np.sqrt((np.std(model_f1_array, ddof=1)**2 + np.std(random_f1_array, ddof=1)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        print(f"    Results: Acc={accuracy_mean:.3f}±{accuracy_std:.3f}, P={precision_mean:.3f}±{precision_std:.3f}, "
              f"R={recall_mean:.3f}±{recall_std:.3f}, F1={f1_mean:.3f}±{f1_std:.3f}")
        print(f"    Random F1: {random_f1_mean:.3f}±{random_f1_std:.3f}, Improvement: +{f1_improvement:.3f}, "
              f"p={wilcoxon_p:.4f}, Cohen's d={cohens_d:.3f} {'[OK]' if significant else '[X]'}")

        results.append({
            'model': model_name,
            'encoding': encoding_name,
            'strategy': strategy_name,
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
        })

    return results


def _compute_basic_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1])
    # metrics are 1-element arrays when labels=[1]
    return float(accuracy), float(precision[0]), float(recall[0]), float(f1[0])


def evaluate_model_single_split(model, X, y, model_name, encoding_name, strategies):
    """
    Single train/test split (85/15). Each imbalance strategy is evaluated separately.
    """
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    results = []

    for strategy_name, strategy in strategies.items():
        fold_model = clone(model)
        fold_model = configure_model_for_strategy(fold_model, model_name, strategy_name)
        X_train_proc, y_train_proc = apply_imbalance_strategy(
            X_train, y_train, strategy_name, strategy
        )
        sample_weight = None
        if strategy_name == 'class_weight':
            sample_weight = compute_balanced_sample_weights(y_train_proc)

        fit_kwargs = {}
        if sample_weight is not None and 'sample_weight' in inspect.signature(fold_model.fit).parameters:
            fit_kwargs['sample_weight'] = sample_weight

        fold_model.fit(X_train_proc, y_train_proc, **fit_kwargs)

        y_test_pred = fold_model.predict(X_test)
        test_accuracy, test_precision, test_recall, test_f1 = _compute_basic_metrics(y_test, y_test_pred)

        # Use same schema as CV run, with 0 padding for unavailable values
        results.append({
            'model': model_name,
            'encoding': encoding_name,
            'strategy': strategy_name,
            'cv_folds': 0,
            'cv_repeats': 0,
            'total_trials': 1,
            'accuracy_mean': test_accuracy,
            'accuracy_std': 0.0,
            'precision_mean': test_precision,
            'precision_std': 0.0,
            'recall_mean': test_recall,
            'recall_std': 0.0,
            'f1_mean': test_f1,
            'f1_std': 0.0,
            'random_f1_mean': 0.0,
            'random_f1_std': 0.0,
            'f1_improvement': 0.0,
            'wilcoxon_p_value': 0.0,
            'cohens_d': 0.0,
            'significantly_better_than_random': False,
        })

    return results


def evaluate_model(model, X_test, y_test, model_name, strategy_name):
    """Evaluate model and return metrics including positive smell detection metrics."""
    y_pred = model.predict(X_test)

    # Get prediction probabilities for Top-K metrics
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)

    # Get metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Class-specific metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # smell samples count and recall
    positive_mask = y_test != 0
    num_positive_samples = np.sum(positive_mask)

    # Positive smell recall (recall for positive samples only, excluding "no smell" class)
    if num_positive_samples > 0:
        positive_recall = np.sum((y_pred[positive_mask] != 0) & (y_pred[positive_mask] == y_test[positive_mask])) / num_positive_samples
    else:
        positive_recall = 0.0

    # Top-K positive smell detection metrics (K = 1, 5, 10) - return counts
    top_k_metrics = {}
    if y_pred_proba is not None:
        for k in [1, 5, 10]:
            top_k_count, total_positive = calculate_top_k_positive_count(y_test, y_pred_proba, k)
            top_k_metrics[f'top_{k}_positive_count'] = int(top_k_count)

    results = {
        'model': model_name,
        'strategy': strategy_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'positive_smell_recall': positive_recall,
        'num_positive_samples': int(num_positive_samples),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support': support,
        **top_k_metrics  # Add Top-K positive metrics
    }

    return results


def get_encoding_methods():
    """Get available text encoding methods with SBERT chunk strategies."""
    methods = {
        'tfidf': 'TF-IDF',
        'sbert_avg': 'SBERT (Avg Pooling)',
        'sbert_max': 'SBERT (Max Pooling)',
        'sbert_tfidf_weighted': 'SBERT (TF-IDF Weighted)',
    }

    if os.getenv("DISABLE_SBERT", "").strip().lower() in {"1", "true", "yes"}:
        return {'tfidf': 'TF-IDF'}

    return methods

def extract_texts_from_data(df, text_source='description'):
    """
    Extract texts from dataframe based on specified source.

    Args:
        df: DataFrame with 'description', 'title', and discussion columns
        text_source: One of 'description', 'combined', 'combined_unfiltered'
                     - 'description': title + description
                     - 'combined': title + description + discussion (pre-MR filtered)
                     - 'combined_unfiltered': title + description + full discussion

    Returns:
        List of text strings
    """
    normalized_source = normalize_text_source(text_source)

    if 'description' not in df.columns and 'title' not in df.columns:
        raise ValueError("Data must include 'title' and 'description' columns")

    texts: list[str] = []

    for _, row in df.iterrows():
        title = row['title'] if pd.notna(row.get('title')) else ''
        description = row['description'] if pd.notna(row.get('description')) else ''
        base_text = _combine_title_and_description(title, description)

        if normalized_source == TEXT_SOURCE_DESCRIPTION:
            texts.append(base_text)
            continue

        prefer_unfiltered = normalized_source == TEXT_SOURCE_COMBINED_UNFILTERED
        discussion = _select_discussion_text(row, prefer_unfiltered=prefer_unfiltered)

        if discussion.strip():
            texts.append(f"{base_text}\n\n[DISCUSSION]\n{discussion}")
        else:
            texts.append(base_text)

    return texts


def load_precomputed_embeddings(embeddings_path):
    """Load pre-computed embeddings from .npz file."""
    print(f"Loading pre-computed embeddings from {embeddings_path}...")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings, labels


def _infer_dataset_ids_for_embeddings(data_path: Optional[Path]) -> list[str]:
    """
    Derive dataset identifiers to search for OpenAI embeddings.

    Prefer the parent directory (system name) so we can pick up the standard
    embeddings/<system>/<system>_<source>_<model>.npz layout, but fall back to
    the filename stem for legacy flat outputs.
    """
    if data_path is None:
        return []

    dataset_ids: list[str] = []
    parent_name = data_path.parent.name
    stem = data_path.stem

    if parent_name:
        dataset_ids.append(parent_name)

    dataset_ids.append(stem)
    return dataset_ids


def _find_openai_embeddings_path(data_path: Optional[str], text_source: Optional[str]) -> Path:
    """
    Locate a compatible OpenAI embeddings file for the given dataset/text source.

    Supports legacy flat files (<dataset>_<text_source>_openai.npz) and the newer
    generate_openai_embeddings layout (<dataset>/<dataset>_<text_source>_<model>.npz).
    """
    resolved_text_source = normalize_text_source(text_source or TEXT_SOURCE_DESCRIPTION)
    candidates: list[Path] = []

    dataset_path = Path(data_path) if data_path else None
    dataset_ids = _infer_dataset_ids_for_embeddings(dataset_path)

    for dataset_id in dataset_ids:
        file_prefix = f"{dataset_id}_{resolved_text_source}"
        base_dir = Path("embeddings")

        candidates.extend([
            base_dir / f"{file_prefix}_openai.npz",  # Legacy flat path
            base_dir / dataset_id / f"{file_prefix}_openai.npz",
            base_dir / dataset_id / f"{file_prefix}_text-embedding-3-small.npz",  # Common new default
        ])

        # Pick up any other model-specific suffixes that follow the new layout
        candidates.extend(sorted((base_dir / dataset_id).glob(f"{file_prefix}_*.npz")))
        candidates.extend(sorted(base_dir.glob(f"{file_prefix}_*.npz")))

    # Legacy inkscape fallbacks (only when the dataset is inkscape)
    if any("inkscape" in dataset_id for dataset_id in dataset_ids):
        candidates.extend([
            Path(f"data/classification/inkscape/inkscape_{resolved_text_source}_openai.npz"),
            Path("data/classification/inkscape/inkscape_discussion_openai.npz"),
        ])

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)

    for path in unique_candidates:
        if path.exists():
            return path

    attempted_paths = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        "OpenAI embeddings not found. Tried:\n"
        f"{attempted_paths}\n\nGenerate embeddings first:\n"
        f"  python -m src.classification.generate_openai_embeddings --data {data_path} --text-source {resolved_text_source}"
    )


def chunk_text(text, max_words=256):
    """
    Split text into chunks of approximately max_words.

    Args:
        text: Input text string
        max_words: Maximum words per chunk

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks if chunks else [text]


def sbert_encode_with_chunking(texts, model, chunk_strategy='avg', max_words=512,
                                tfidf_vectorizer=None, batch_size=128, show_progress=False):
    """
    Encode texts using SBERT with different chunk pooling strategies.

    Args:
        texts: List of text strings
        model: SentenceTransformer model
        chunk_strategy: One of 'avg', 'max', 'tfidf_weighted'
                        - 'avg': Average pooling of chunk embeddings
                        - 'max': Max pooling of chunk embeddings (element-wise max)
                        - 'tfidf_weighted': TF-IDF weighted average of chunks
        max_words: Maximum words per chunk (default: 512)
        tfidf_vectorizer: Pre-fitted TfidfVectorizer (required for 'tfidf_weighted')
        batch_size: Batch size for encoding (default: 128)
        show_progress: Show progress bar during encoding (default: False)

    Returns:
        numpy array of embeddings with shape (n_texts, embedding_dim)
    """
    # Step 1: Chunk all texts and track chunk boundaries
    all_chunks = []
    chunk_boundaries = [0]
    text_chunks_list = []

    for text in texts:
        chunks = chunk_text(text, max_words=max_words)
        text_chunks_list.append(chunks)
        all_chunks.extend(chunks)
        chunk_boundaries.append(len(all_chunks))

    # Step 2: Batch encode all chunks
    chunk_embeddings_all = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    # Step 3: Apply pooling strategy for each text
    all_embeddings = []

    for i, chunks in enumerate(text_chunks_list):
        start_idx = chunk_boundaries[i]
        end_idx = chunk_boundaries[i + 1]
        chunk_embeddings = chunk_embeddings_all[start_idx:end_idx]

        # Apply pooling strategy
        if chunk_strategy == 'avg':
            final_embedding = np.mean(chunk_embeddings, axis=0)

        elif chunk_strategy == 'max':
            final_embedding = np.max(chunk_embeddings, axis=0)

        elif chunk_strategy == 'tfidf_weighted':
            if tfidf_vectorizer is None:
                raise ValueError("tfidf_vectorizer is required for 'tfidf_weighted' strategy")

            # Clean chunks before TF-IDF transformation
            cleaned_chunks = [clean_text_for_tfidf(chunk) for chunk in chunks]
            chunk_tfidf = tfidf_vectorizer.transform(cleaned_chunks)
            chunk_weights = np.array(chunk_tfidf.sum(axis=1)).flatten()

            if chunk_weights.sum() > 0:
                chunk_weights = chunk_weights / chunk_weights.sum()
            else:
                chunk_weights = np.ones(len(chunks)) / len(chunks)

            final_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_weights)

        else:
            raise ValueError(f"Unknown chunk_strategy: {chunk_strategy}")

        all_embeddings.append(final_embedding)

    return np.array(all_embeddings)


def _append_text_source_to_output_path(output_file: str | Path, text_source: str) -> Path:
    """
    Ensure the output filename includes the text source so downstream artifacts are disambiguated.
    Only appends when the text source token is absent.
    """
    path = Path(output_file)
    ext = path.suffix or ".csv"
    base = path.stem if path.suffix else path.name
    tokens = base.lower().split("_")
    normalized_source = text_source.lower()

    if normalized_source not in tokens:
        base = f"{base}_{text_source}"

    return path.with_name(f"{base}{ext}")


def test_sbert_chunk_strategies(texts, labels, model_name="paraphrase-mpnet-base-v2",
                                max_words=512, classifier=None, use_undersampling=True,
                                batch_size=128, clean_texts=True):
    """
    Test different SBERT chunk strategies and compare their performance.

    Args:
        texts: List of text strings
        labels: List of labels
        model_name: Name or path of SentenceTransformer model
        max_words: Maximum words per chunk (default: 512)
        classifier: sklearn classifier to use for evaluation (default: LogisticRegression)
        use_undersampling: Apply random undersampling to balance training data (default: True)
        batch_size: Batch size for SBERT encoding (default: 128)
        clean_texts: Apply text cleaning before encoding (default: True)

    Returns:
        Dictionary with results for each strategy including embeddings and metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    print(f"\n{'='*60}")
    print(f"Testing SBERT Chunk Strategies")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Chunk size: {max_words} words")
    print(f"Batch size: {batch_size}")
    print(f"Number of texts: {len(texts)}")
    print(f"Undersampling: {use_undersampling}")
    print(f"Text cleaning: {clean_texts}")

    # Apply text cleaning if requested
    if clean_texts:
        print("Cleaning texts (no lemmatization)...")
        texts = [clean_text_base(text) for text in texts]

    # Initialize model with GPU if available
    device = _select_torch_device()
    print(f"Device: {device}")
    SentenceTransformer = _load_sentence_transformer()
    model = SentenceTransformer(model_name, device=device)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Split data
    from sklearn.model_selection import train_test_split
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Train/test split: {len(texts_train)} train, {len(texts_test)} test")

    # Default classifier
    if classifier is None:
        classifier = LogisticRegression(random_state=42, max_iter=2000, C=0.1)

    # Test strategies
    strategies = ['avg', 'max', 'tfidf_weighted']
    results = {}

    for strategy in strategies:
        print(f"\n{'-'*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'-'*60}")

        # Prepare TF-IDF vectorizer if needed
        tfidf_vectorizer = None
        if strategy == 'tfidf_weighted':
            print("Fitting TF-IDF vectorizer for weighting...")
            print("Cleaning texts for TF-IDF weighting...")
            cleaned_texts_train = [clean_text_for_tfidf(text) for text in texts_train]
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_vectorizer.fit(cleaned_texts_train)

        # Encode texts with chunking strategy
        print(f"Encoding {len(texts_train)} train texts (batch_size={batch_size})...")
        X_train = sbert_encode_with_chunking(
            texts_train, model, chunk_strategy=strategy,
            max_words=max_words, tfidf_vectorizer=tfidf_vectorizer,
            batch_size=batch_size, show_progress=True
        )

        print(f"Encoding {len(texts_test)} test texts (batch_size={batch_size})...")
        X_test = sbert_encode_with_chunking(
            texts_test, model, chunk_strategy=strategy,
            max_words=max_words, tfidf_vectorizer=tfidf_vectorizer,
            batch_size=batch_size, show_progress=False
        )

        print(f"Embedding shape: {X_train.shape}")

        # Apply undersampling if requested
        X_train_resampled = X_train
        y_train_resampled = y_train

        if use_undersampling and IMBLEARN_AVAILABLE:
            print("Applying random undersampling...")
            from imblearn.under_sampling import RandomUnderSampler

            # Show class distribution before
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"  Before: {dict(zip(unique, counts))}")

            rus = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

            # Show class distribution after
            unique, counts = np.unique(y_train_resampled, return_counts=True)
            print(f"  After: {dict(zip(unique, counts))}")
            print(f"  Samples: {len(y_train)} -> {len(y_train_resampled)}")

        # Train classifier
        print("Training classifier...")
        from sklearn.base import clone
        clf = clone(classifier)
        clf.fit(X_train_resampled, y_train_resampled)

        # Evaluate
        print("Evaluating...")
        y_pred = clf.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import classification_report, accuracy_score
        accuracy = accuracy_score(y_test, y_pred)

        # Get precision, recall, f1 for positive class (label 1)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[1], average='binary', zero_division=0
        )

        # Positive smell recall (all non-zero classes)
        positive_mask = y_test != 0
        num_positive_samples = np.sum(positive_mask)
        if num_positive_samples > 0:
            positive_recall = np.sum(
                (y_pred[positive_mask] != 0) & (y_pred[positive_mask] == y_test[positive_mask])
            ) / num_positive_samples
        else:
            positive_recall = 0.0

        # Store results
        results[strategy] = {
            'embeddings_train': X_train,
            'embeddings_test': X_test,
            'classifier': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_smell_recall': positive_recall,
            'num_positive_samples': int(num_positive_samples)
        }

        print(f"\nResults for {strategy}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Positive Smell Recall: {positive_recall:.4f} ({num_positive_samples} samples)")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*60}")
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
              f"{r['recall']:<12.4f} {r['f1']:<12.4f}")

    # Best strategy by F1
    best_strategy = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest strategy by F1-score: {best_strategy[0].upper()} (F1={best_strategy[1]['f1']:.4f})")

    return results


def encode_texts(texts, encoding_method, data_path=None, text_source=None,
                 max_words=512, batch_size=128, show_progress=False, return_model=True):
    """
    Encode texts using specified method.

    Args:
        texts: List of text strings
        encoding_method: One of 'tfidf', 'sbert_avg', 'sbert_max', 'sbert_tfidf_weighted', 'openai'
        data_path: Path to data file (for OpenAI embeddings)
        text_source: Text source type (for OpenAI embeddings)
        max_words: Maximum words per chunk for SBERT (default: 512)
        batch_size: Batch size for SBERT encoding (default: 128)
        show_progress: Show progress bar for SBERT encoding (default: False)
        return_model: Return the model in the tuple (default: True). Set False to free GPU memory.

    Returns:
        Tuple of (embeddings/features, encoder/model or None)
    """
    if encoding_method == 'tfidf':
        # Clean texts before TF-IDF encoding (with lemmatization)
        print("Cleaning texts for TF-IDF encoding (with lemmatization)...")
        cleaned_texts = [clean_text_for_tfidf(text) for text in texts]
        vectorizer = TfidfVectorizer(max_features=50000, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(cleaned_texts)
        return X, vectorizer

    elif encoding_method in ['sbert_avg', 'sbert_max', 'sbert_tfidf_weighted']:
        # Determine chunk strategy from encoding method name
        strategy_map = {
            'sbert_avg': 'avg',
            'sbert_max': 'max',
            'sbert_tfidf_weighted': 'tfidf_weighted'
        }
        chunk_strategy = strategy_map[encoding_method]

        # Clean texts before SBERT encoding (without lemmatization)
        print("Cleaning texts for SBERT encoding (no lemmatization)...")
        cleaned_texts = [clean_text_base(text) for text in texts]

        # Initialize SBERT model with GPU support when available, otherwise CPU
        device = _select_torch_device()
        print(f"Using device: {device}")
        SentenceTransformer = _load_sentence_transformer()
        model = SentenceTransformer("paraphrase-mpnet-base-v2", device=device)

        # Prepare TF-IDF vectorizer if needed for tfidf_weighted strategy
        tfidf_vectorizer = None
        if chunk_strategy == 'tfidf_weighted':
            print("Fitting TF-IDF vectorizer for chunk weighting...")
            # Use lemmatized texts for TF-IDF weighting
            tfidf_texts = [clean_text_for_tfidf(text) for text in texts]
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_vectorizer.fit(tfidf_texts)

        # Encode with chunking
        embeddings = sbert_encode_with_chunking(
            cleaned_texts, model,
            chunk_strategy=chunk_strategy,
            max_words=max_words,
            tfidf_vectorizer=tfidf_vectorizer,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Clean up GPU memory after encoding
        if not return_model:
            del model
            _clear_cuda_cache()
            return embeddings, None

        # Clear cache but keep model
        _clear_cuda_cache()
        return embeddings, model

    elif encoding_method == 'openai':
        # Load pre-computed OpenAI embeddings
        embeddings_path = _find_openai_embeddings_path(data_path, text_source)

        print(f"Loading OpenAI embeddings from: {embeddings_path}")
        embeddings, _ = load_precomputed_embeddings(embeddings_path)
        num_texts = len(texts)
        if embeddings.shape[0] != num_texts:
            raise ValueError(
                (
                    f"OpenAI embeddings count ({embeddings.shape[0]}) does not match "
                    f"number of samples ({num_texts}). Regenerate embeddings for this "
                    f"dataset/text source or remove the incompatible file: {embeddings_path}"
                )
            )
        return embeddings, None

    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}. "
                        f"Choose from: 'tfidf', 'sbert_avg', 'sbert_max', 'sbert_tfidf_weighted', 'openai'")

def generate_simple_summary(results_csv_path, output_csv_path=None):
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


def run_traditional_ml_experiments(
    data_path: str,
    output_file: str = 'ml_results.csv',
    text_source: str = 'description',
    use_cv: bool = False,
    cv_folds: int = 5,
    cv_repeats: int = 2,
):
    """
    Run experiments combining traditional ML models with imbalance strategies and encoding methods.

    Supports two evaluation modes:
    1. Single 85/15 train/test split (default, fast)
    2. Repeated cross-validation with significance testing (use_cv=True)

    Args:
        data_path: Path to JSON data file
        output_file: Output CSV file for results
        text_source: Source of text for classification. One of:
                     - 'description': Use only issue description (default)
                     - 'combined': Combine description + filtered discussion
                     - 'combined_unfiltered': Combine description + full discussion
        use_cv: Use repeated cross-validation instead of single split (default: False)
        cv_folds: Number of folds per repetition (default: 5)
        cv_repeats: Number of repetitions (default: 2, total trials = folds × repeats)
    """
    normalized_text_source = normalize_text_source(text_source)
    output_path = _append_text_source_to_output_path(output_file, normalized_text_source)

    print("Loading and preprocessing data...")
    print(f"Text source: {normalized_text_source}")
    print(f"Output file (with text source): {output_path}")
    if use_cv:
        total_trials = cv_folds * cv_repeats
        print(f"Evaluation: Repeated CV ({cv_repeats} × {cv_folds}-fold = {total_trials} trials)")
        print("Includes significance testing against random baseline")
    else:
        print("Evaluation: single 85/15 train/test split; all imbalance strategies evaluated separately")

    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Extract texts based on text_source parameter
    texts = extract_texts_from_data(df, text_source=normalized_text_source)
    labels = df['label'].tolist()

    # Get encoding methods
    encoding_methods = get_encoding_methods()

    # Auto-enable OpenAI embeddings when a matching file is present
    if os.getenv("DISABLE_OPENAI_EMBEDDINGS", "").strip().lower() not in {"1", "true", "yes"}:
        try:
            openai_path = _find_openai_embeddings_path(data_path, normalized_text_source)
            if openai_path:
                encoding_methods = {'openai': 'OpenAI', **encoding_methods}
                print(f"Detected OpenAI embeddings at: {openai_path}")
        except FileNotFoundError as exc:
            print(f"No OpenAI embeddings detected: {exc}")

    print(f"Loaded {len(labels)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Get strategies
    strategies = get_imbalance_strategies()

    # Store results
    all_results = []

    # Main loop: for each encoding method and model (strategy selected via inner CV)
    for encoding_name, encoding_desc in encoding_methods.items():
        print(f"\n=== Encoding Method: {encoding_desc} ===")

        # Encode texts once
        print(f"Encoding texts with {encoding_desc}...")

        # Use batch encoding and GPU for SBERT methods
        if encoding_name.startswith('sbert_'):
            try:
                sbert_batch_size = int(os.getenv("SBERT_BATCH_SIZE", "128"))
            except ValueError:
                sbert_batch_size = 128
            X, encoder = encode_texts(
                texts, encoding_name,
                data_path=data_path,
                text_source=normalized_text_source,
                max_words=256,
                batch_size=sbert_batch_size,
                show_progress=True
            )
        else:
            X, encoder = encode_texts(texts, encoding_name, data_path=data_path, text_source=normalized_text_source)

        print(f"Encoded {X.shape[0]} samples")

        # Get models appropriate for this encoding method
        base_models = get_base_models(encoding_name)

        if use_cv:
            # Repeated CV mode
            print(f"\nRunning repeated CV experiments:")
            print(f"Encoding: {encoding_name}")
            print(f"Models: {list(base_models.keys())}")
            print(f"Strategies (evaluated individually): {list(strategies.keys())}")

            for model_name, base_model in base_models.items():
                print(f"\n--- {model_name}: Repeated CV ({cv_repeats} × {cv_folds}-fold) ---")
                cv_results = evaluate_model_repeated_cv(
                    model=base_model,
                    X=X,
                    y=y,
                    model_name=model_name,
                    encoding_name=encoding_name,
                    strategies=strategies,
                    n_folds=cv_folds,
                    n_repeats=cv_repeats,
                    random_state=42,
                )
                all_results.extend(cv_results)
        else:
            # Single split mode
            print(f"\nRunning single-split experiments:")
            print(f"Encoding: {encoding_name}")
            print(f"Models: {list(base_models.keys())}")
            print(f"Strategies (evaluated individually): {list(strategies.keys())}")

            for model_name, base_model in base_models.items():
                print(f"\n--- {model_name}: Single Split ---")
                single_results = evaluate_model_single_split(
                    model=base_model,
                    X=X,
                    y=y,
                    model_name=model_name,
                    encoding_name=encoding_name,
                    strategies=strategies,
                )
                all_results.extend(single_results)
                for res in single_results:
                    print(f"  Strategy: {res['strategy']}")
                    print(f"    Test Results: F1={res['f1_mean']:.3f}, "
                          f"Recall={res['recall_mean']:.3f}, "
                          f"Precision={res['precision_mean']:.3f}, "
                          f"Accuracy={res['accuracy_mean']:.3f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Generate simple summary
    generate_simple_summary(output_path)

    # Print summary
    if use_cv:
        print("\n=== REPEATED CV SUMMARY ===")
        print(f"Repeated CV ({cv_repeats} × {cv_folds}-fold = {cv_folds * cv_repeats} trials) with significance testing")

        # Count significant improvements
        if 'significantly_better_than_random' in results_df.columns:
            n_significant = results_df['significantly_better_than_random'].sum()
            n_total = len(results_df)
            print(f"\nSignificantly better than random (p<0.05): {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")

        # Display results
        summary_cols = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean',
                       'f1_std', 'f1_improvement', 'wilcoxon_p_value',
                       'significantly_better_than_random', 'strategy']
        available_cols = [col for col in summary_cols if col in results_df.columns]

        print("\nResults by encoding and model:")
        if 'f1_mean' in results_df.columns:
            summary = results_df[['encoding', 'model'] + available_cols].sort_values('f1_mean', ascending=False)
            print(summary.to_string(index=False))

        # Top 10 by F1 mean
        print("\nTop 10 model/encoding combinations by Mean F1 Score:")
        display_cols = ['encoding', 'model', 'strategy',
                       'accuracy_mean', 'accuracy_std',
                       'precision_mean', 'precision_std',
                       'recall_mean', 'recall_std',
                       'f1_mean', 'f1_std',
                       'f1_improvement', 'wilcoxon_p_value', 'cohens_d',
                       'significantly_better_than_random']
        available_display = [col for col in display_cols if col in results_df.columns]

        if 'f1_mean' in results_df.columns:
            top_f1 = results_df.nlargest(10, 'f1_mean')[available_display]
            print(top_f1.to_string(index=False))
    else:
        print("\n=== SINGLE SPLIT SUMMARY ===")
        print("Single 85/15 train/test split; all strategies evaluated separately")
        summary_cols = ['f1_mean', 'recall_mean', 'precision_mean', 'accuracy_mean', 'strategy']
        available_cols = [col for col in summary_cols if col in results_df.columns]

        print("\nResults by encoding and model:")
        summary = results_df[['encoding', 'model'] + available_cols].sort_values('f1_mean', ascending=False)
        print(summary.to_string(index=False))

        # Best performing combinations by test F1 score
        print("\nTop 10 model/encoding combinations by Test F1 Score:")
        display_cols = ['encoding', 'model', 'strategy',
                        'f1_mean', 'recall_mean', 'precision_mean']
        available_display = [col for col in display_cols if col in results_df.columns]

        top_f1 = results_df.nlargest(10, 'f1_mean')[available_display]
        print(top_f1.to_string(index=False))

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run traditional ML experiments with imbalance strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default: inkscape/ACDC, single 85/15 train/test split)
  python -m src.classification.simple_ml

  # Specific dataset
  python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/shepard_acdc/data/classification/dlr-shepard-shepard/classification_data_clean.json

  # Title + description + filtered discussion
  python -m src.classification.simple_ml --text-source combined
        """
    )

    parser.add_argument(
        'data_path',
        nargs='?',
        default='data-output/Output/supervised_ml_all/run/inkscape_acdc/data/classification/inkscape/classification_data_clean.json',
        help='Path to JSON data file (default: %(default)s)'
    )

    parser.add_argument(
        '--output', '-o',
        default='ml_results.csv',
        help='Output CSV file for results (default: %(default)s)'
    )

    parser.add_argument(
        '--text-source', '-t',
        choices=sorted(VALID_TEXT_SOURCES),
        default=TEXT_SOURCE_DESCRIPTION,
        help='Source of text for classification: description, combined (filtered discussions), combined_unfiltered'
    )

    parser.add_argument(
        '--use-cv',
        action='store_true',
        help='Use repeated cross-validation instead of single train/test split'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds per repetition (default: 5)'
    )

    parser.add_argument(
        '--cv-repeats',
        type=int,
        default=2,
        help='Number of CV repetitions (default: 2, total trials = folds × repeats)'
    )

    args = parser.parse_args()

    print("Traditional ML Classification")
    print("=" * 50)
    if args.use_cv:
        total_trials = args.cv_folds * args.cv_repeats
        print(f"Config: Repeated CV ({args.cv_repeats} × {args.cv_folds}-fold = {total_trials} trials)")
        print("Includes significance testing against random baseline")
    else:
        print("Config: single 85/15 train/test split, all imbalance strategies evaluated")

    results = run_traditional_ml_experiments(
        data_path=args.data_path,
        output_file=args.output,
        text_source=args.text_source,
        use_cv=args.use_cv,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
    )

    print("\nExperiment completed!")
