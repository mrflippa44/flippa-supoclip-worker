# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Linux x86_64 Docker container tuned for Render Free Web Service (512 MB RAM, dynamic $PORT)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    TEMP_DIR=/tmp/supoclip-worker \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FFMPEG_THREADS=1 \
    FFMPEG_PRESET=superfast \
    FACE_DETECTOR=opencv \
    TRANSCRIBE_PROVIDER=auto

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libass9 \
    fonts-dejavu-core \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN mkdir -p src/pipeline /tmp/supoclip-worker/uploads /tmp/supoclip-worker/clips && \
    [ -f src/__init__.py ] || printf '"""SupoClip Linux worker package."""\n' > src/__init__.py && \
    [ -f src/pipeline/__init__.py ] || printf '"""SupoClip Linux processing pipeline."""\n' > src/pipeline/__init__.py && \
    for f in config.py exceptions.py; do [ -f "$f" ] && [ ! -f "src/$f" ] && mv "$f" "src/$f" || true; done && \
    for f in analyze.py clip.py download.py face_detect.py subtitles.py transcribe.py; do [ -f "$f" ] && [ ! -f "src/pipeline/$f" ] && mv "$f" "src/pipeline/$f" || true; done && \
    chown -R user:user /tmp/supoclip-worker /app

USER user

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
