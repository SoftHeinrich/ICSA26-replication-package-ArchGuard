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

## 3. Configure Data Path (30 seconds)

Edit `classification/dataset_config.py` and update `DATASET_BASE_DIR` to point to your data:

```python
DATASET_BASE_DIR = Path("/path/to/your/data")
```

## 4. Run Your First Experiment (2-30 minutes depending on method)

### Option A: Quick Traditional ML Test
```bash
python -m casestudy.classification_pipeline_all \
  --methods traditional_ml \
  --projects inkscape \
  --text-sources description \
  --encoding-methods tfidf \
  --ml-models logistic_regression \
  --imbalance-strategies none
```
**Time:** ~2 minutes

### Option B: RoBERTa Fine-tuning (requires GPU)
```bash
python -m casestudy.finetune_all \
  --projects inkscape \
  --detectors acdc \
  --text-sources description \
  --epochs 3 \
  --dry-run  # Remove --dry-run to actually run
```
**Time:** ~30 minutes with GPU

### Option C: Zero-shot LLM (requires OpenAI API key)
```bash
export OPENAI_API_KEY="your-key-here"

python -m casestudy.zero_shot_all \
  --projects inkscape \
  --detectors acdc \
  --text-source description \
  --model gpt-5-mini
```
**Time:** ~5-10 minutes depending on dataset size

## 5. View Results

Results are saved to `Output/` directory with timestamped subdirectories:
- `Output/comprehensive_classification/{timestamp}/` - Pipeline results
- `Output/hf_roberta_batch/{timestamp}/` - Fine-tuning results
- `Output/zero_shot_runs/{timestamp}/` - Zero-shot results

Look for:
- `*_summary.csv` - Quick overview of metrics
- `*_summary.json` or `summary.json` - Detailed results

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

### Run All Combinations (WARNING: Very slow!)
```bash
# Fine-tune all configurations
python -m casestudy.finetune_all

# Zero-shot all configurations
python -m casestudy.zero_shot_all

# ALL methods (traditional ML + RoBERTa + zero-shot)
python -m casestudy.classification_pipeline_all
```

## Troubleshooting

**Import Error:** Make sure you're in the parent directory of `src-rep`:
```bash
cd /path/to/parent
python -m src-rep.casestudy.finetune_all  # If src-rep is a package
# OR
cd src-rep
python -m casestudy.finetune_all  # Run from inside src-rep
```

**Data Not Found:** Update `DATASET_BASE_DIR` in `classification/dataset_config.py`

**GPU Memory Error:** Reduce batch size: `--batch-size 4`

**OpenAI API Error:** Check your API key: `echo $OPENAI_API_KEY`

---

For more details, see the full [README.md](README.md)
