# AI Document Search

Semantic PDF search API built with FastAPI, Sentence Transformers, FAISS and Google Gemini. Upload PDFs, ask questions, and receive answers with document names and page references.

## Requirements

- Python 3.12 (the tested runtime).
- A private `API_KEY` to use upload, list, delete and search routes.
- Internet access to download `sentence-transformers/all-MiniLM-L6-v2` on first use, or an existing local cache.
- A Gemini API key and access to the configured Gemini model to generate answers.

The API can start without either key or the embedding model. The health route and interactive documentation remain available. Missing configuration is reported only when a dependent operation is requested.

## Local installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cp .env.example .env
```

The CPU PyTorch installation avoids downloading unused CUDA libraries. For GPU development, install an appropriate PyTorch build using its official instructions instead.

Edit `.env` and set `API_KEY` to your own private key. You can generate one locally:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Set `GEMINI_API_KEY` securely to enable answer generation, and choose a `GEMINI_MODEL` available to your account. Environment variables take precedence over `.env`; in a cloud environment, use secure environment settings rather than committing credentials. The default model name is retained from the original project and must be available to your account.

Start a single worker:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Documentation is at `/docs`, and OpenAPI at `/openapi.json`. Do not expose the service publicly before configuring a private API key. The former default key `changeme` is rejected.

## Configuration

| Variable | Purpose / default |
| --- | --- |
| `API_KEY` | Required for protected routes; no default key |
| `GEMINI_API_KEY` | Required only for generated answers |
| `GEMINI_MODEL` | Default: `gemini-3.5-flash-lite`; account availability must be verified |
| `EMBEDDING_MODEL` | Default: `sentence-transformers/all-MiniLM-L6-v2`; may be a local path to a compatible 384-dimensional model |
| `HF_HOME` | Optional persistent Hugging Face cache directory |
| `SIMILARITY_THRESHOLD` | Maximum squared L2 distance; default `1.2` |
| `TOP_K` | Maximum retrieved chunks; positive integer, default `4` |
| `CHUNK_SIZE` | Positive chunk size in characters; default `800` |
| `CHUNK_OVERLAP` | Overlap in characters, from zero to less than chunk size; default `150` |
| `MAX_UPLOAD_SIZE_MB` | Positive upload size limit; default `15` |
| `UPLOAD_DIR` | Uploaded files; default `uploads`, relative to the working directory |
| `DATA_DIR` | Persistent FAISS index and metadata; default `data`, relative to the working directory |

Model loading happens only on the first embedding request. Use a writable persistent `HF_HOME` to reuse downloads. Required network destinations include Hugging Face and its model storage/CDN hosts, plus `generativelanguage.googleapis.com` for Gemini. A cached compatible model can be used without network access to Hugging Face.

For installation/model troubleshooting, retain TLS and download verification; do not disable certificate validation.

## API

| Method | Route | Authentication |
| --- | --- | --- |
| GET | `/` | Public health/configuration summary |
| POST | `/documents/upload` | `X-API-Key` |
| GET | `/documents` | `X-API-Key` |
| DELETE | `/documents/{filename}` | `X-API-Key` |
| GET | `/search?question=...` | `X-API-Key` |

Supply your configured key using the `X-API-Key` header. The health response reports whether keys are configured without returning their values; it does not contact Gemini or download a model, so it is not an end-to-end readiness test.

Upload the included demonstration PDF (with `API_KEY` exported in your terminal):

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/documents/upload \
  -H "X-API-Key: $API_KEY" \
  -F 'file=@tests/fixtures/sample.pdf;type=application/pdf'
curl --fail-with-body --get http://127.0.0.1:8000/search \
  -H "X-API-Key: $API_KEY" \
  --data-urlencode 'question=What does Project Aurora research?'
```

Answers include source filenames, page numbers, squared L2 distance and a relevance value calculated as `1 / (1 + distance)`. This relevance value is not a probability or a calibrated confidence score. The PDF contains fictional project information and has no personal data.

Expected errors:

- `401`: missing or incorrect request API key.
- `503`: server API key is absent/uses `changeme`, the embedding model cannot load, Gemini is missing/unavailable, or the provider returns an empty answer.
- `400`: invalid PDF, PDF without extractable text, blank question, or no indexed documents.
- `409`: duplicate document filename.
- `413`: upload exceeds the size limit.

Without a Gemini key, upload/list/delete can still work if the API key and embedding model are available. Search only calls Gemini when relevant chunks are found. Scanned PDFs need OCR, which is not included.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The default suite requires no Gemini credentials, model download or external PDF. It uses the included two-page fixture, real PDF extraction and real FAISS persistence. Only embeddings and Gemini responses are simulated. Tests use temporary upload/index directories and block outbound socket connections. They do not modify your indexed documents.

The upload → index → search → answer → delete test checks source pages and reopens the saved FAISS index. Other tests cover authentication, missing configuration, corrupt/blank PDFs, upload size, chunk settings, provider errors and cleanup after embedding failure. These tests verify integration wiring, not the semantic quality of the real model or answers.

### Optional live checks

Live tests are excluded by default. Run them explicitly after configuring network access, a compatible model and a valid Gemini key:

```bash
python -m pytest -q -m live tests/test_live.py
```

The embedding test downloads/loads the real model and checks a basic semantic ranking. The Gemini test makes an actual provider request and can incur usage charges; it is skipped if no key is configured. A skipped test does not establish that Gemini works.

### GitHub Actions

The **Offline tests** workflow runs the default suite for pushes and Pull Requests, with Hugging Face offline mode enabled and no API keys. It does not download the embedding model or call Gemini. Installation still downloads Python dependencies.

## Docker

The Dockerfile uses Python 3.12 and CPU PyTorch, matching local setup:

```bash
cp .env.example .env
# Configure your own keys before using protected routes.
docker compose up --build
```

Compose retains uploads, index data and a model cache. It binds port 8000 to loopback for local development. Do not add multiple workers or replicas sharing the same files: the current FAISS persistence is intended for a single process and does not provide atomic snapshots across its three files. Multi-worker consistency and disk-failure recovery require a separate storage change.

If your build network uses a corporate TLS proxy, provide its trusted CA bundle using BuildKit's optional `build_ca` secret rather than disabling verification:

```bash
docker build --secret id=build_ca,src=/path/to/trusted-ca-bundle.pem -f dockerfile -t ai-document-search .
```

Proxy settings can be forwarded with the standard `--build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY` options. The CA bundle is mounted only during installation and is not copied into the image. Runtime model/provider access still needs its own valid network and trust configuration.

## Structure

- `app/main.py`: API routes and authentication.
- `app/document_processor.py`: PDF extraction and page-aware chunking.
- `app/embedding_service.py`: lazy model loading and embeddings.
- `app/vector_store.py`: FAISS storage and persistence.
- `app/llm.py`: lazy Gemini client and answer generation.
- `tests/fixtures/sample.pdf`: reproducible, fictional two-page document.
- `tests/test_live.py`: optional checks against real external services.
