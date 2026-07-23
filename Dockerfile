# syntax=docker/dockerfile:1

# ---- Stage 1: builder ----------------------------------------------------
# Compiles wheels (incl. C extensions like mmh3) into an isolated venv.
# Build tools live only here, so they never bloat the final image.
FROM python:3.13-slim AS builder

# gcc + Python headers are needed to build mmh3 and other C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv we can copy wholesale into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: runtime ----------------------------------------------------
# Slim final image: just Python + the prebuilt venv + app code. No compilers.
FROM python:3.13-slim AS runtime

# Don't buffer stdout/stderr (so logs stream); don't write .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as a non-root user for safety.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

# Pre-download the NLTK data the BM25 tokenizer needs (punkt_tab + stopwords) into
# a fixed, world-readable path, so the first query doesn't fetch it at runtime and
# it works even if the NLTK servers are unreachable at boot. NLTK_DATA points the
# runtime lookup here (otherwise it writes corpora/ + tokenizers/ under $HOME).
ENV NLTK_DATA=/opt/nltk_data
RUN python -m nltk.downloader -d /opt/nltk_data punkt_tab stopwords

WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Single worker on purpose: the APScheduler cron loop and in-memory job state
# assume one process. Scaling out requires an external queue first.
CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
