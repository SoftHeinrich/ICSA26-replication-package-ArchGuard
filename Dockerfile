FROM python:3.14.2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

COPY src/requirements.txt /app/src/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/src/requirements.txt \
    && python - <<'PY'
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
PY

COPY . /app

ENTRYPOINT bash -c "cat README.md && bash"
