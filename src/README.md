# Detailed Usage

All commands run from the **project root** using `python -m src.` module syntax.

## Installation

```bash
conda env create -f environment.yml
conda activate di
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

Optional environment variables:
```bash
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export XGB_DEVICE="cuda:0"           # Use GPU for XGBoost
export SBERT_DEVICE="cuda:0"         # Use GPU for sentence embeddings
```

## Data

Classification data is in JSON format at paths configured in `src/classification/dataset_config.py` (auto-resolved relative to project root):

```json
[
  {
    "description": "text description of architectural issue",
    "label": "smell_category"
  }
]
```

**Default data location:** `data-output/Output/supervised_ml_all/run/`

**Text sources:**
- `description`: Issue title + description
- `combined`: Title + description + filtered discussion

---

## 1. Traditional ML (`simple_ml.py`)

TF-IDF/SBERT encodings with 6 classifiers (LogReg, RF, SVM, NB, KNN, XGBoost) and random undersampling.

```bash
# Default (inkscape/ACDC)
python -m src.classification.simple_ml

# All datasets:
# inkscape / ACDC
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/inkscape_acdc/data/classification/inkscape/classification_data_clean.json
# shepard / ACDC
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/shepard_acdc/data/classification/dlr-shepard-shepard/classification_data_clean.json
# shepard / ARCAN
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/shepard_arcan/data/classification/dlr-shepard-shepard/classification_data_clean.json
# stackgres / ACDC
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/stackgres_acdc/data/classification/ongresinc-stackgres/classification_data_clean.json
# stackgres / ARCAN
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/stackgres_arcan/data/classification/ongresinc-stackgres/classification_data_clean.json

# With repeated cross-validation (any dataset)
python -m src.classification.simple_ml \
  data-output/Output/supervised_ml_all/run/inkscape_acdc/data/classification/inkscape/classification_data_clean.json \
  --text-source combined \
  --use-cv --cv-folds 5 --cv-repeats 2
```


---

## 2. RoBERTa Fine-tuning (`finetune_all.py`)

RoBERTa fine-tuning with text chunking (256 words) and repeated stratified k-fold CV (2x5).

```bash
# All project-detector-text_source combinations
python -m src.casestudy.finetune_all

# Preview without running
python -m src.casestudy.finetune_all --dry-run
```

**Output:** `Output/hf_roberta_batch/{timestamp}/` with `finetune_results_summary.csv` and `.json`.

---

## 3. Cross-Project Leave-One-Out (`cross_project.py`)

Trains on N-1 projects, evaluates on the held-out project.
- ACDC: 3 combinations (inkscape, shepard, stackgres)
- ARCAN: 2 combinations (shepard, stackgres)

```bash
# All detectors, all encodings
python -m src.classification.cross_project --text-source description

# Preview without running
python -m src.classification.cross_project --dry-run
```

**Output:** `data-output/cross_project_loo/{timestamp}/` with `cross_project_loo_*_combined.csv`.

---

## Configuration (`src/classification/dataset_config.py`)

Dataset paths resolve automatically relative to the project root:
- `DATASET_BASE_DIR`: `data-output/Output/supervised_ml_all/run/`
- `PROJECT_DETECTORS`: inkscape (acdc), shepard (acdc, arcan), stackgres (acdc, arcan)
- `DATASET_FOLDERS`: Maps (project, detector) pairs to folder names
