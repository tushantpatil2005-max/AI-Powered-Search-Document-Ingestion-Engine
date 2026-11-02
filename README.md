# Document Ingestion & Search Engine

A service that ingests PDF, CSV and XLSX files, splits them into chunks, and indexes each chunk in a Qdrant
vector database. Search finds the chunks whose meaning is closest to a natural-language query, with optional
filters by file type, source file and category.

The whole application is a single file, `main.py`.

**Stack:** FastAPI, sentence-transformers, Qdrant, PyMuPDF, pandas and openpyxl (XLSX).

## Features

- Ingestion of PDF (per page), CSV and XLSX (per row, across all sheets).
- Semantic search with metadata filters and an optional minimum score.
- Safe re-uploads: identical content is a no-op, and a changed file replaces its earlier version.
- Upload limits, content checks, and clear error codes for bad files.
- Optional API-key protection.
- Batched embedding, so large files do not need to fit in memory as vectors.
- Health endpoint that checks the vector store.

## Requirements

- Python 3.10 or newer
- A Qdrant server for real use (without one, an in-memory store is used and data is lost on restart)

```bash
pip install fastapi "uvicorn[standard]" python-multipart sentence-transformers \
            "qdrant-client>=1.10" pymupdf pandas openpyxl
```

The first start downloads the embedding model (`all-MiniLM-L6-v2` by default).

## Run

```bash
# Optional: start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

export QDRANT_URL=http://localhost:6333
uvicorn main:app            # or: python main.py
```

Interactive API docs are at `http://localhost:8000/docs`.

## Usage

```bash
# Ingest a file (optionally tag it with a category)
curl -X POST localhost:8000/ingest \
  -F "file=@handbook.pdf" -F "custom_category=hr"

# Search
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "how many vacation days do I get?", "limit": 3, "doc_type": "pdf"}'

# Remove a file from the index
curl -X DELETE "localhost:8000/documents?source=handbook.pdf"

# Health
curl localhost:8000/health
```

If `API_KEY` is set, add `-H "X-API-Key: <your key>"` to `/ingest`, `/search` and `/documents` requests.

## API

### `POST /ingest`

Multipart form: `file` (required, `.pdf`, `.csv` or `.xlsx`) and `custom_category` (optional, default `general`).

| Status | Meaning |
|---|---|
| `201` | Indexed. Response: `filename`, `document_id`, `chunks_indexed`, `status: "indexed"` |
| `200` | Identical content was already indexed: `status: "unchanged"` |
| `401` | Missing or invalid API key |
| `413` | File larger than `MAX_UPLOAD_MB`, or more than `MAX_CHUNKS_PER_DOCUMENT` chunks |
| `415` | Unsupported file type |
| `422` | Empty, corrupt, password-protected or text-free document, or content that does not match the extension |

### `POST /search`

| Field | Type | Description |
|---|---|---|
| `query` | string, 1 to 2000 characters | Natural-language query |
| `limit` | integer, 1 to 50 (default 5) | Number of results |
| `doc_type` | string, optional | `pdf`, `csv` or `xlsx` |
| `source` | string, optional | Original filename |
| `category` | string, optional | Category given at ingestion |
| `score_threshold` | number from -1 to 1, optional | Minimum similarity score |

Response: `query`, `total_results`, and `results`, each with `id`, `score`, `text` and `metadata`.

Metadata always includes `source`, `doc_type`, `document_id`, `chunk_index` and `category`, plus `page` for PDFs
and `row` (and `sheet` for XLSX) for tables.

### `DELETE /documents?source=<filename>`

Removes every chunk of one file. Returns `source` and `deleted_chunks`.

### `GET /health`

Returns `200` with `status`, `vector_db` and `model` when Qdrant is reachable, and `503` otherwise.
This endpoint does not need an API key.

## How it works

1. **Parse.** PDFs are read page by page. CSV and XLSX files produce one chunk per row in the form
   `column: value, column: value` (XLSX rows are prefixed with `[Sheet: name]`). Values are kept exactly as
   written, so identifiers such as `007` stay intact. CSV delimiters (`,` `;` tab `|`) are detected from the
   header row, and non-UTF-8 files are read as Latin-1.
2. **Chunk.** Text is split into overlapping windows of words (`CHUNK_WORDS`, `CHUNK_OVERLAP`). Table rows
   longer than one window are split the same way.
3. **Embed.** Chunks are embedded in batches with the sentence-embedding model and normalised.
4. **Store.** Vectors and metadata are upserted into Qdrant (cosine distance). Payload indexes are created on
   `source`, `doc_type`, `category` and `document_id` so filtered search stays fast.
5. **Search.** The query is embedded the same way and the closest chunks are returned.

**Chunk size matters.** `all-MiniLM-L6-v2` reads at most 256 word pieces per input and ignores the rest.
The default of 160 words fits. If you raise `CHUNK_WORDS`, the service warns at startup when the value is too large
for the model. If you change `EMBEDDING_MODEL` to one with a different vector size, use a new `COLLECTION_NAME`;
the service refuses to start against a collection with a mismatched size.

**Idempotent uploads.** Each file gets a `document_id` (SHA-256 of its bytes) and each chunk a deterministic
point id. Uploading the same content again changes nothing; uploading a changed file with the same name replaces
the old chunks only after the new ones are stored. A failed upload removes its partial chunks.

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | unset | Qdrant server URL. Unset: in-memory store, data lost on restart |
| `QDRANT_API_KEY` | unset | Qdrant API key, if the server requires one |
| `COLLECTION_NAME` | `enterprise_documents` | Qdrant collection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `API_KEY` | unset | When set, `/ingest`, `/search` and `/documents` require the `X-API-Key` header |
| `MAX_UPLOAD_MB` | `50` | Largest accepted file |
| `MAX_CHUNKS_PER_DOCUMENT` | `50000` | Largest number of chunks per file |
| `CHUNK_WORDS` | `160` | Words per chunk |
| `CHUNK_OVERLAP` | `30` | Words shared between neighbouring chunks |
| `EMBED_BATCH_SIZE` | `64` | Chunks embedded and uploaded per batch |
| `LOG_LEVEL` | `INFO` | Logging level |

## Limitations

- Scanned PDFs without a text layer return `422`; there is no OCR.
- Legacy `.xls` files are not supported. Convert them to `.xlsx`.
- Search is semantic only; there is no keyword or hybrid ranking.
- Files are processed in the API process. For heavy or high-volume ingestion, move parsing and embedding to a
  background worker queue.
- Filenames identify documents. Two different files with the same name replace each other; use distinct names
  (or add a prefix per tenant or project) if that is a concern.
- Not included: per-user access control, rate limiting, and OCR.

## Tests

```bash
pip install httpx
python -m unittest test_main -v
```

The tests use in-memory stand-ins for the embedding model and Qdrant, so no model download or server is needed.
They cover chunking, filename and content checks, CSV and XLSX parsing, batching, idempotent re-uploads,
replacement of old versions, cleanup after failures, size limits, search filters, deletion, settings, API-key
protection and error codes.

They do not cover PDF parsing or the real Qdrant and sentence-transformers adapters; try those against a real
Qdrant instance with a sample PDF before relying on them in production.
