from datetime import datetime

FACTS_ROOT = "data/facts"
CLUSTERS_ROOT = "data/clusters"
SUBJECT_SYSTEMS_ROOT = "data/subject_systems"
STOPWORDS_DIR_PATH = "data/stopwords"
METRICS_ROOT = "data/metrics"
SMELLS_ROOT = "data/smells"

def time_print(text: str):
  now = datetime.now()
  strnow = now.strftime("%H:%M:%S")
  print(f"{strnow}: {text}")
