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

## Step 2: Prepare Data

Download `data-output.zip` from the Figshare link above and extract it into the project root:

```bash
unzip data-output.zip -d data-output
```

This creates the `data-output/` folder containing the classification datasets and smell evolution data. The expected structure is:

```
data-output/
├── Output/supervised_ml_all/run/
│   ├── inkscape_acdc/data/classification/inkscape/classification_data_clean.json
│   ├── shepard_acdc/data/classification/dlr-shepard-shepard/classification_data_clean.json
│   ├── shepard_arcan/data/classification/dlr-shepard-shepard/classification_data_clean.json
│   ├── stackgres_acdc/data/classification/ongresinc-stackgres/classification_data_clean.json
│   └── stackgres_arcan/data/classification/ongresinc-stackgres/classification_data_clean.json
└── data/smells/...
```

## Step 3: Install Dependencies

**Option A: Conda**

```bash
conda env create -f environment.yml
conda activate di
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Option B: Pip**

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Docker

Alternatively, use Docker for a self-contained environment with all dependencies and NLTK data pre-installed:

```bash
docker build -t icsa26 .
docker run -it --rm --gpus=all icsa26
```

The container drops into a bash shell at the project root, ready to run experiments. Remove `--gpus=all` if no GPU is available.

## Step 4: Verify Installation

Run the following command from the project root to verify the artifact is working:

```bash
python -m src.classification.simple_ml
```

**Expected output** (runtime ~10 minutes, faster with GPU for SBERT encoding):

```
Traditional ML Classification
==================================================
Config: single 85/15 train/test split, all imbalance strategies evaluated
Loading and preprocessing data...
Text source: description
Output file (with text source): ml_results_description.csv
Loaded 2536 samples
Label distribution: {'no_smell': 2278, 'smell': 258}

=== Encoding Method: TF-IDF ===
Encoding texts with TF-IDF...
Cleaning texts for TF-IDF encoding (with lemmatization)...
Encoded 2536 samples

Running single-split experiments:
...
--- logistic_regression: Single Split ---
  random_undersample: 2155 -> 438 samples
  Strategy: random_undersample
    Test Results: F1=0.210, Recall=0.436, Precision=0.138, Accuracy=0.664
...
```

The script produces `ml_results_description.csv` in the current directory with per-model metrics (Accuracy, Precision, Recall, F1). If you see the results table printed and the CSV file created, the installation is successful.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project root**, not from `src/` |
| NLTK `LookupError` | Run `python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"` |
| GPU out of memory | Set `export SBERT_DEVICE="cpu"` to use CPU for sentence embeddings |
| Tokenizer warnings | Set `export TOKENIZERS_PARALLELISM=false` |

For detailed usage of all experiment scripts, see [README.md](README.md) and [src/README.md](src/README.md).
