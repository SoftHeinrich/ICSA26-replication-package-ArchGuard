# Replication Package: Architectural Smell Classification

This is a standalone replication package for architectural smell detection research, including:
1. **Data Generation** - Generate classification datasets from preprocessed project artifacts
2. **Classification Experiments** - Run supervised learning, zero-shot, and fine-tuning experiments

## Package Contents

```
src-rep/
├── README.md                          # This file - Main documentation
├── DATA_GENERATION.md                 # Data generation guide
├── QUICKSTART.md                      # 5-minute quick start
├── MANIFEST.md                        # Complete file inventory
├── requirements.txt                   # Python dependencies
│
├── casestudy/                        # Main experiment scripts
│   ├── finetune_all.py              # RoBERTa fine-tuning wrapper
│   └── zero_shot_all.py             # Zero-shot LLM wrapper
│
├── pipeline/                         # End-to-end pipelines
│   ├── preprocessing_pipeline.py    # Fact extraction & smell detection
│   └── classification_pipeline.py   # Full classification data pipeline
│
├── classification/                   # Core classification modules
│   ├── dataset_config.py            # Dataset configuration
│   ├── simple_ml.py                 # Traditional ML
│   ├── roberta_finetune.py          # RoBERTa implementation
│   └── zero_shot_llm_classifier.py  # Zero-shot classifier
│
├── preprocessing/                    # Data generation & preprocessing
│   ├── gen_classification_data.py   # Generate classification datasets
│   ├── extract_facts.py             # Extract architecture facts
│   ├── run_clustering.py            # ARCADE clustering
│   ├── detect_smells.py             # Smell detection
│   └── dependency_filter.py         # Filter test dependencies
│
├── mapper/                           # Issue-smell mapping
│   ├── component_mapper.py          # Issue-to-component mapping
│   └── smell_to_issue_mapper.py     # Smell-to-issue mapping
│
├── commit/                           # Commit analysis
│   ├── batch_dependency_analyzer.py # Dependency change analysis
│   └── dependency_change_detector.py # Detect dependency changes
│
├── issues/                           # Issue management
│   ├── config.py                    # Configuration & enums
│   ├── base.py                      # Base classes
│   └── utils.py                     # Utility functions
│
├── utils/                            # Utilities
│   ├── rsf_utils.py                 # Parse architecture files
│   ├── release_utils.py             # Version utilities
│   └── system_paths.py              # Path configuration
│
├── version/                          # Version management
│   └── mr_version_merger.py         # Enrich issues with versions
│
└── smell/                            # Smell analysis
    └── analyze_smell_evolution.py   # Track smell evolution
```

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### 2. Install Dependencies

```bash
cd src-rep
pip install -r requirements.txt
```

### 3. Download NLTK Data (first-time setup)
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

### 4. Environment Variables

For **zero-shot classification**, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Optional environment variables for optimization:
```bash
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export XGB_DEVICE="cuda:0"           # Use GPU for XGBoost
export SBERT_DEVICE="cuda:0"         # Use GPU for sentence embeddings
```

## Data Requirements

All scripts expect classification data in JSON format at paths specified in `classification/dataset_config.py`:

```json
[
  {
    "description": "text description of architectural issue",
    "label": "smell_category"
  }
]
```

**Default data location:** `Output/supervised_ml_all/20251204_120354/`

You may need to update `DATASET_BASE_DIR` in `classification/dataset_config.py` to point to your data location.

### Generating Classification Data

If you need to generate classification data from preprocessed project artifacts, see **[DATA_GENERATION.md](DATA_GENERATION.md)** for the complete guide.

**Quick example:**
```bash
# Generate classification dataset from preprocessed data
python -m preprocessing.gen_classification_data \
  --s2i data/mapping/project/evolution_to_issue_project_acdc.json \
  --mapping data/mapping/project/mapping_project_acdc.json \
  --issues data/issues/project \
  --output-clean data/classification/project/classification_data_clean.json \
  --output-dirty data/classification/project/classification_data_dirty.json
```

**Prerequisites:** Preprocessed artifacts including:
- Issue data (scraped from GitHub/GitLab)
- Architecture clusters (from ARCADE)
- Smell detection results
- Component and smell-to-issue mappings

**Note:** The full preprocessing pipeline requires external tools (Understand, ARCADE, MALLET) not included in this package. See DATA_GENERATION.md for details.

## Usage

### 1. Fine-Tune RoBERTa Models (finetune_all.py)

Runs RoBERTa fine-tuning across all project-detector-text_source combinations.

**Basic Usage:**
```bash
python -m casestudy.finetune_all
```

**Filter to specific configurations:**
```bash
# Run only for inkscape project
python -m casestudy.finetune_all --projects inkscape

# Run only ACDC detector
python -m casestudy.finetune_all --detectors acdc

# Run only description text source
python -m casestudy.finetune_all --text-sources description

# Customize model and training parameters
python -m casestudy.finetune_all \
  --model roberta-base \
  --epochs 6 \
  --batch-size 12 \
  --learning-rate 2e-5 \
  --chunk-words 256 \
  --n-splits 5 \
  --n-repeats 2
```

**Enable early stopping:**
```bash
python -m casestudy.finetune_all --early-stopping --early-stopping-patience 5
```

**Dry run (preview configurations):**
```bash
python -m casestudy.finetune_all --dry-run
```

**Output:**
- Results saved to: `Output/hf_roberta_batch/{timestamp}/`
- Summary files: `finetune_results_summary.csv` and `finetune_results_summary.json`

---

### 2. Zero-Shot LLM Classification (zero_shot_all.py)

Runs zero-shot LLM classification using OpenAI models.

**Basic Usage:**
```bash
python -m casestudy.zero_shot_all
```

**Filter to specific configurations:**
```bash
# Run only for specific projects
python -m casestudy.zero_shot_all --projects inkscape shepard

# Run only ACDC detector
python -m casestudy.zero_shot_all --detectors acdc

# Specify model and text source
python -m casestudy.zero_shot_all \
  --model gpt-4o-mini \
  --text-source combined
```

**Enable chain-of-thought prompting:**
```bash
python -m casestudy.zero_shot_all --use-cot
```

**Control API rate limiting:**
```bash
python -m casestudy.zero_shot_all \
  --max-workers 8 \
  --api-delay 0.5
```

**Use cached responses only (no API calls):**
```bash
python -m casestudy.zero_shot_all --cache-only
```

**Dry run (preview configurations):**
```bash
python -m casestudy.zero_shot_all --dry-run
```

**Output:**
- Results saved to: `Output/zero_shot_runs/{timestamp}/`
- Summary file: `summary.json`
- Per-run results: `{project}/{detector}/results_{model}_{text_source}.csv`

---

### 3. End-to-End Pipelines

#### 3a. Preprocessing Pipeline (preprocessing_pipeline.py)

Runs the full preprocessing workflow: fact extraction, clustering, smell detection, and evolution analysis.

**Basic Usage:**
```bash
python -m pipeline.preprocessing_pipeline \
  --system myproject \
  --language java \
  --memory 8
```

**Options:**
```bash
# Skip specific stages
python -m pipeline.preprocessing_pipeline \
  --system myproject \
  --language python \
  --skip-facts \
  --skip-clustering

# Filter test dependencies
python -m pipeline.preprocessing_pipeline \
  --system myproject \
  --language java \
  --filter-keywords test mock
```

**Prerequisites:**
- SciTools Understand (for fact extraction)
- ARCADE Core (for clustering and smell detection)
- MALLET (for topic modeling)

---

#### 3b. Classification Pipeline (classification_pipeline.py)

Orchestrates the complete classification data generation workflow from preprocessed artifacts.

**Basic Usage:**
```bash
python -m pipeline.classification_pipeline \
  --system myproject \
  --algorithm acdc \
  --issue-dir data/issues/myproject \
  --evolution-file data/smells/myproject/evolution/myproject_acdc_evolution.json \
  --output-root Output/pipeline_runs
```

**What it does:**
1. Maps issues to architectural components
2. Links smell evolution to issues
3. Generates classification datasets (clean & dirty)
4. Optionally analyzes dependency changes

**Options:**
```bash
# Specify text source
python -m pipeline.classification_pipeline \
  --system myproject \
  --algorithm acdc \
  --issue-dir data/issues/myproject \
  --evolution-file data/smells/myproject/evolution.json \
  --output-root Output/pipeline_runs \
  --text-source combined

# Skip smell detection
python -m pipeline.classification_pipeline \
  --system myproject \
  --no-smell
```

**Output:**
- Results saved to: `{output_root}/{system}_{algorithm}_{timestamp}/`
- Includes:
  - `data/mapping/` - Component and smell-to-issue mappings
  - `data/classification/` - classification_data_clean.json and _dirty.json
  - `data/issues/` - Dependency change checklist

---

## Expected Outputs

### Metrics Reported

All classification methods report:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1 Score**: Weighted average F1 score

**RoBERTa also reports:**
- Chunk-level metrics (predictions on text chunks)
- Issue-level metrics (aggregated predictions per issue)

### Result Files

1. **CSV files**: Tabular results with predictions and metrics
2. **JSON files**: Structured metrics and configuration
3. **Summary files**: Aggregated results across runs

---

## Configuration

### Dataset Configuration (`classification/dataset_config.py`)

Key constants:
- `DATASET_BASE_DIR`: Base directory for datasets
- `DATASET_FOLDERS`: Mapping of (project, detector) to folder names
- `PROJECT_DETECTORS`: Available detectors per project
- `PROJECT_DIR_NAMES`: Project directory name mappings
- `CLASSIFICATION_DATA_FILENAME`: Default data filename

**To use your own data:**
1. Update `DATASET_BASE_DIR` to point to your data location
2. Ensure data follows the expected JSON format
3. Update `DATASET_FOLDERS` and `PROJECT_DETECTORS` if needed

---

## Technical Details

### Classification Methods

1. **Traditional ML** (simple_ml.py)
   - Encoding: TF-IDF, SBERT, OpenAI embeddings
   - Models: Logistic Regression, Random Forest, SVM, Naive Bayes, KNN, XGBoost
   - Imbalance handling: class weights, SMOTE, undersampling

2. **RoBERTa Fine-tuning** (roberta_finetune.py)
   - Text chunking with word-based strategy
   - Repeated stratified K-fold cross-validation
   - Issue-level aggregation (voting, averaging)
   - Early stopping support

3. **Zero-shot LLM** (zero_shot_llm_classifier.py)
   - OpenAI GPT models
   - Optional chain-of-thought prompting
   - Caching support for repeated runs
   - Parallel API workers

### Text Sources
- `description`: Issue description only
- `combined`: Description + discussion + code snippets
- `combined_unfiltered`: Combined without preprocessing filters
