# ArchGuard

Welcome to the ArchGuard replication package!
This framework analyzes software issue reports to find those that are inducing architectural smells.
Based on the analysis, it trains models that can classify unseen issues based on title and description, using three complementary machine learning approaches: traditional ML, RoBERTa fine-tuning, and zero-shot LLM classification.

The approach and evaluation are detailed in our paper:
**Architecture in the Cradle: Early Prevention of Architectural Decay with ArchGuard**

## Features
- **Three Subject Systems**: Evaluated on Inkscape, Shepard, and StackGres with ACDC and ARCAN smell detectors
- **Three Classification Approaches**: Compare traditional ML (TF-IDF/SBERT + 6 classifiers), RoBERTa fine-tuning, and zero-shot LLM (GPT) on the same datasets
- **Cross-Project Evaluation**: Leave-one-out cross-project classification to assess generalizability across software systems


## Getting Started

### 1. Clone the Repository

You can download from figshare link https://doi.org/10.6084/m9.figshare.30833303   

Or clone from Github

```bash
git clone https://github.com/SoftHeinrich/ICSA26-replication-package-ArchGuard
cd ICSA26-replication-package-ArchGuard
```

### 2. Install Dependencies

Requires Python 3.14.2. GPU with CUDA support is optional but recommended.

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate di

# Option B: Pip
pip install -r requirements.txt

# Then download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

### 3. Run an Experiment

All commands run from the **project root** using `python -m`:

#### Traditional ML

```bash
# inkscape / ACDC (default)
python -m src.classification.simple_ml
# shepard / ACDC
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/shepard_acdc/data/classification/dlr-shepard-shepard/classification_data_clean.json
# shepard / ARCAN
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/shepard_arcan/data/classification/dlr-shepard-shepard/classification_data_clean.json
# stackgres / ACDC
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/stackgres_acdc/data/classification/ongresinc-stackgres/classification_data_clean.json
# stackgres / ARCAN
python -m src.classification.simple_ml data-output/Output/supervised_ml_all/run/stackgres_arcan/data/classification/ongresinc-stackgres/classification_data_clean.json
```

#### RoBERTa Fine-tuning

```bash
# All project-detector-text_source combinations
python -m src.casestudy.finetune_all
# Preview without running
python -m src.casestudy.finetune_all --dry-run
```

#### Cross-Project Leave-One-Out

```bash
# All detectors, all encodings
python -m src.classification.cross_project --text-source description
# Preview without running
python -m src.classification.cross_project --dry-run
```

#### Zero-Shot LLM (requires OpenAI API key)

```bash
export OPENAI_API_KEY="your-key-here"
python -m src.casestudy.zero_shot_all --projects inkscape --detectors acdc --text-source description
```

> [!NOTE]
> Use `--dry-run` with any experiment script to preview configurations without running them.

### Configuration

- **Subject systems**: inkscape (ACDC only), shepard (ACDC + ARCAN), stackgres (ACDC + ARCAN)
- **Text sources**: `description` (title + description), `combined` (+ filtered discussion)
- **Data format**: JSON arrays of `{"description": "...", "label": "smell_category"}`

### Results

Results are saved with per-model metrics:

| Method | Output Location |
|--------|----------------|
| Traditional ML | `ml_results_*.csv` (current directory) |
| RoBERTa | `Output/hf_roberta_batch/{timestamp}/` |
| Zero-shot LLM | `Output/zero_shot_runs/{timestamp}/` |
| Cross-project | `data-output/cross_project_loo/{timestamp}/` |

All methods report Accuracy, Precision, Recall, and F1 Score. RoBERTa additionally reports chunk-level and issue-level metrics with multiple aggregation strategies (vote, mean, max).



## Documentation

- [Detailed Usage](src/README.md): Full documentation for all experiment scripts and pipelines

## Architecture


| Module | Purpose |
|--------|---------|
| `src/classification/` | Core classifiers: traditional ML, RoBERTa, zero-shot LLM, cross-project |
| `src/pipeline/` | End-to-end pipelines for preprocessing and classification data generation |
| `src/mapper/` | Maps MR file changes to architectural components and links smells to issues |
| `src/smell/` | Tracks architectural smell evolution across versions via Jaccard similarity |


## Evaluation

ArchGuard has been evaluated on three open-source systems using two architectural smell detectors:

| System | Detector |
|--------|----------|
| Inkscape | ACDC |
| Shepard | ACDC | 
| Shepard | ARCAN | 
| StackGres | ACDC |
| StackGres | ARCAN |

Classification performance is assessed using single split train-test or repeated stratified k-fold cross-validation (2 repeats x 5 folds) with Wilcoxon signed-rank tests against random baselines.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

*For comprehensive technical details on each module, see the [full documentation](src/README.md).*
