# Quick Start Guide

This guide will get you running experiments in 5 minutes.

## 1. Install Dependencies (1 minute)

```bash
cd src-rep
pip install -r requirements.txt
```

## 2. Download NLTK Data (first-time only)

```python
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## 3. Configure Data Path

Edit `classification/dataset_config.py` and update `DATASET_BASE_DIR` to point to your data:

```python
DATASET_BASE_DIR = Path("/path/to/your/data")
```

## 4. Run Your First Experiment

### Option A: Quick Traditional ML Test (sentence-bert encoding is speeded up by GPU)
```bash
python -m casestudy.classification_pipeline_all \
  --methods traditional_ml \
  --projects inkscape \
  --text-sources description \
  --encoding-methods tfidf \
  --ml-models logistic_regression \
  --imbalance-strategies none
```

### Option B: RoBERTa Fine-tuning (requires GPU)
```bash
python -m casestudy.finetune_all \
  --projects inkscape \
  --detectors acdc \
  --text-sources description \
  --epochs 3 \
  --dry-run  # Remove --dry-run to actually run
```

### Option C: Zero-shot LLM (requires OpenAI API key)
```bash
export OPENAI_API_KEY="your-key-here"

python -m casestudy.zero_shot_all \
  --projects inkscape \
  --detectors acdc \
  --text-source description \
  --model gpt-5-mini
```

## 5. View Results

Results are saved to `Output/` directory with timestamped subdirectories.

## Common Commands

### Dry Run (Preview without Running)
```bash
# See what would be executed
python -m casestudy.finetune_all --dry-run
python -m casestudy.zero_shot_all --dry-run
python -m casestudy.classification_pipeline_all --dry-run
```

### Filter to Specific Projects
```bash
# Run only inkscape
python -m casestudy.finetune_all --projects inkscape

# Run only shepard and stackgres
python -m casestudy.zero_shot_all --projects shepard stackgres
```

### Run All Combinations
```bash
# Fine-tune all configurations
python -m casestudy.finetune_all

# Zero-shot all configurations
python -m casestudy.zero_shot_all

# ALL methods (traditional ML + RoBERTa + zero-shot)
python -m casestudy.classification_pipeline_all
```


For more details, see the full [README.md](README.md)
