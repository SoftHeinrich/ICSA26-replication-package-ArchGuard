# Quick Start Guide

This guide will get you running experiments in 5 minutes.

## 1. Install Dependencies

```bash
conda env create -f environment.yml
conda activate di
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## 2. Run Your First Experiment

All commands run from the **project root**.

### Traditional ML (fastest)

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

### RoBERTa Fine-tuning (requires GPU)

```bash
# All project-detector-text_source combinations
python -m src.casestudy.finetune_all
# Preview without running
python -m src.casestudy.finetune_all --dry-run
```

### Cross-Project Leave-One-Out

```bash
# All detectors, all encodings
python -m src.classification.cross_project --text-source description
# Preview without running
python -m src.classification.cross_project --dry-run
```

### Zero-Shot LLM (requires OpenAI API key)

```bash
export OPENAI_API_KEY="your-key-here"
python -m src.casestudy.zero_shot_all --projects inkscape --detectors acdc --text-source description
```

## 3. View Results

| Method | Output Location |
|--------|----------------|
| Traditional ML | `ml_results_*.csv` (current directory) |
| RoBERTa | `Output/hf_roberta_batch/{timestamp}/` |
| Zero-shot LLM | `Output/zero_shot_runs/{timestamp}/` |
| Cross-project | `data-output/cross_project_loo/{timestamp}/` |

For more details, see the full [README.md](../README.md).
