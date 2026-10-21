FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY requirements.txt .
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/app/runtime /home/user/.cache/huggingface \
    && chown -R user:user /home/user
COPY --chown=user:user app ./app
COPY --chown=user:user evaluations/documents ./evaluations/documents

ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    DATA_DIR=/home/user/app/runtime/data \
    UPLOAD_DIR=/home/user/app/runtime/uploads \
    DEMO_MODE=true \
    EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    EMBEDDING_REVISION=e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
    DEMO_ALLOW_GENERATION=false \
    DEMO_REQUESTS_PER_MINUTE=20 \
    DEMO_GENERATIONS_PER_DAY=20

USER user
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
