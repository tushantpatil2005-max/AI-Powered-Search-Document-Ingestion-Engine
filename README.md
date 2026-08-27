# AI-Powered Search & Document Ingestion Engine

An automated ETL and semantic search engine that parses unstructured enterprise documents (PDF, CSV, XLSX), generates dense vector embeddings, and delivers sub-second hybrid retrieval using **FastAPI**, **Sentence-Transformers**, and **Qdrant Vector DB**.

---

## 1. Complete Application (`app.py`)

Save the following code as `app.py`:

```python
"""
AI-Powered Search & Document Ingestion Engine
Tech Stack: FastAPI, Sentence-Transformers, Qdrant Vector DB, PyMuPDF, Pandas
"""

import io
import os
import uuid
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration & Global State
# ==========================================
COLLECTION_NAME = "enterprise_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
QDRANT_HOST = os.getenv("QDRANT_HOST", ":memory:")

client: Optional[QdrantClient] = None
model: Optional[SentenceTransformer] = None


# ==========================================
# Lifespan Events
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, model
    # Initialize Embedding Model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Initialize Qdrant Client (In-memory default or remote service)
    client = QdrantClient(location=QDRANT_HOST)
    
    # Ensure Collection Exists
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSION, 
                distance=models.Distance.COSINE
            ),
        )
    yield
    if client:
        client.close()


app = FastAPI(
    title="AI Document Ingestion & Search Engine",
    description="Automated ETL pipeline and hybrid semantic retrieval engine.",
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================
# Schemas
# ==========================================
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(5, ge=1, le=50, description="Max results to return")
    doc_type: Optional[str] = Field(None, description="Filter by extension (e.g., 'pdf', 'csv')")
    source: Optional[str] = Field(None, description="Filter by exact source filename")


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchHit]


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str


# ==========================================
# ETL & Parsing Utilities
# ==========================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits raw text into sliding window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def parse_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF documents using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_content = [page.get_text() for page in doc]
    return "\n".join(text_content)


def parse_csv(file_bytes: bytes) -> List[str]:
    """Converts CSV rows into semantic key-value strings."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    return [
        ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        for _, row in df.iterrows()
    ]


def parse_xlsx(file_bytes: bytes) -> List[str]:
    """Converts Excel rows across all sheets into semantic text rows."""
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    chunks = []
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_chunks = [
            f"[Sheet: {sheet_name}] " + ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            for _, row in df.iterrows()
        ]
        chunks.extend(sheet_chunks)
    return chunks


# ==========================================
# Endpoints
# ==========================================
@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    custom_category: Optional[str] = Form(None)
):
    """
    Ingests PDF, CSV, or XLSX files, creates chunks, generates embeddings,
    and indexes vectors into Qdrant.
    """
    filename = file.filename or "unknown"
    ext = filename.split(".")[-1].lower()
    content = await file.read()
    
    if ext == "pdf":
        raw_text = parse_pdf(content)
        chunks = chunk_text(raw_text)
    elif ext == "csv":
        chunks = parse_csv(content)
    elif ext in ["xlsx", "xls"]:
        chunks = parse_xlsx(content)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format: .{ext}. Supported formats: PDF, CSV, XLSX."
        )

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in file.")

    # Generate Embeddings (Batched)
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False).tolist()

    # Index Points into Qdrant
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source": filename,
                    "doc_type": ext,
                    "chunk_index": i,
                    "text": chunk,
                    "category": custom_category or "general"
                }
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    return IngestResponse(
        filename=filename,
        chunks_indexed=len(points),
        status="Indexed successfully"
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Hybrid search combining vector similarity with payload metadata filtering.
    """
    query_vector = model.encode(request.query).tolist()

    filter_conditions = []
    if request.doc_type:
        filter_conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=request.doc_type.lower())
            )
        )
    if request.source:
        filter_conditions.append(
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=request.source)
            )
        )

    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=request.limit
    )

    results = [
        SearchHit(
            id=str(hit.id),
            score=hit.score,
            text=hit.payload.get("text", ""),
            metadata={k: v for k, v in hit.payload.items() if k != "text"}
        )
        for hit in hits
    ]

    return SearchResponse(query=request.query, total_results=len(results), results=results)


@app.get("/health")
def health_check():
    return {"status": "healthy", "vector_db": "connected", "model": EMBEDDING_MODEL_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
