# Hugging Face Docker Space image: one container that serves predictions AND retrains.
# Spaces run containers as uid 1000 with an ephemeral disk; retrained models persist
# through the Hub model registry (src/registry.py, HF_TOKEN + HF_MODEL_REPO).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY --chown=user src/ src/
COPY --chown=user api/ api/
COPY --chown=user models/ models/
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh \
    && mkdir -p data && chown -R user:user /app

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    DOWNLOAD_DATA=true \
    ENABLE_TRAINING=true \
    INFERENCE_THREADS=2

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
