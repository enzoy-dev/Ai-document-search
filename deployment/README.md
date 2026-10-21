# Publish the demonstration on Hugging Face Spaces

The public demo uses a separate fixed collection of fictional PDFs. It allows no
uploads, deletions, or access to the private collection. Semantic retrieval uses the
multilingual MiniLM model at commit `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
Gemini generation is off by default.

## Test locally

```bash
docker compose -f compose.demo.yml up --build
```

Visit http://127.0.0.1:7860/ui. The model is loaded/downloaded on the first query;
the health endpoint `/health` does not test model readiness. Docker volumes retain
the demo index, usage counters and model cache across ordinary container restarts.

## Prepare the upload

From the repository root, with the project environment active:

```bash
python -m deployment.prepare_space --output /tmp/ai-document-search-space
```

Choose an empty directory outside the checkout (on Windows, for example
`C:\Temp\ai-document-search-space`). The preparer copies only application code,
runtime dependencies, the four public example PDFs, a Dockerfile and the Space
README metadata. It excludes `.env`, `.git`, private PDFs, databases, uploads,
tests and benchmark reports. It refuses to overwrite a nonempty destination.

## Publish through your account

1. Sign in to https://huggingface.co and create a new **Space** named
   `ai-document-search`. Select **Docker → Blank**, choose its visibility and a
   CPU environment with enough memory for the model. Check the platform's current
   resource availability and pricing before selecting hardware.
2. Upload the prepared directory's contents to the Space, preserving the `app/`
   and `evaluations/documents/` directories. You can use the CLI below after
   authenticating in your own terminal:

   ```bash
   hf auth login
   hf upload YOUR_USERNAME/ai-document-search /tmp/ai-document-search-space . --repo-type space
   ```

   The CLI is installed with the project's Hugging Face dependency. Use its
   interactive login in your own terminal to authenticate.
3. Wait for the Space build to finish. Open the app, try an example question,
   inspect the excerpt and open its PDF. Confirm that upload is absent and that
   `/documents` returns `403`. `/health` should report `demo_mode: true`.
4. Put the real Space URL in the repository/profile README after that check.
   Preparing a bundle does not create or publish a Space.

## Optional generated answers

To enable generation, add `GEMINI_API_KEY` as a **Space secret**, set
`DEMO_ALLOW_GENERATION=true` as a variable, and choose a `GEMINI_MODEL` available
to your account. The key is used only by the server. Configure the provider's
own quota/billing limits before enabling paid requests.

The app defaults to 20 shared requests/minute and 20 generation attempts/day per
instance. Attempts are reserved atomically before processing and include failures
or searches that later abstain. These are fixed UTC windows, not per-user limits;
one visitor can exhaust the shared allowance. Counters survive process restarts
only while their SQLite file remains. Ephemeral Space storage can be reset by
rebuilds/restarts, so this application limit is not a durable billing cap.
Multiple replicas each have their own budget; this deployment is designed for one
worker/instance. A sleeping or rebuilt Space may download the model again.

For a hosted private document collection, use a separate deployment with proper
user isolation and authentication. This fixed-corpus demo is not a multi-user
document management service.

## Updates

Generate a new bundle into a new empty directory and upload it to the same Space.
If changing the model, rebuild the disposable demo index in fresh runtime storage;
never reuse an incompatible private collection. The full project's existing model
identity checks also apply to the demo index.
