# Installation and Setup

Configuration and installation take approximately **5 minutes**.

## Prerequisites

- **Python 3.14.2** (with pip or conda)
- **GPU with CUDA support** is optional but recommended for SBERT encoding and RoBERTa fine-tuning

## Step 1: Obtain the Artifact

Download from Figshare: https://doi.org/10.6084/m9.figshare.30833303

Or clone from GitHub:

```bash
git clone https://github.com/SoftHeinrich/ICSA26-replication-package-ArchGuard
cd ICSA26-replication-package-ArchGuard
```

## Step 2: Install Dependencies

**Option A: Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate di
```

**Option B: Pip**

```bash
pip install -r requirements.txt
```

Then download required NLTK data:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## Step 3: Verify Installation

Run the following command from the project root to verify the artifact is working:

```bash
python -m src.classification.simple_ml --text-source description --encoding tfidf
```

**Expected output** (runtime ~1 minute on CPU):

```
Traditional ML Classification
==================================================
Config: single 85/15 train/test split, all imbalance strategies evaluated
Loading and preprocessing data...
...
=== Encoding Method: TF-IDF ===
...
--- logistic_regression: Single Split ---
  ...
    Test Results: F1=0.210, Recall=0.436, Precision=0.138, Accuracy=0.664
...
```

The script produces `ml_results_description.csv` in the current directory with per-model metrics (Accuracy, Precision, Recall, F1). If you see the results table printed and the CSV file created, the installation is successful.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project root**, not from `src/` |
| NLTK `LookupError` | Run `python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"` |
| GPU out of memory | Set `export SBERT_DEVICE="cpu"` to use CPU for sentence embeddings |
| Tokenizer warnings | Set `export TOKENIZERS_PARALLELISM=false` |

For detailed usage of all experiment scripts, see [README.md](README.md) and [src/README.md](src/README.md).
