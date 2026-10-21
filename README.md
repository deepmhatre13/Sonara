# Sonara — Audio Intelligence & Knowledge Retrieval

Production-oriented backend for audio ingestion, asynchronous speech transcription, speaker diarization, hybrid vector/keyword retrieval, and evidence-grounded question answering (RAG).

---

## 1. Project Overview

Sonara enables users to securely upload audio recordings, transcribe conversations with speaker diarization, index content across dense embeddings and PostgreSQL full-text search, and query audio knowledge with verifiable, timestamp-linked citations.

### Tech Stack
- **Language**: Python 3.12+
- **Framework**: FastAPI, Pydantic v2, Pydantic Settings
- **Database & ORM**: PostgreSQL 16, pgvector, TSVECTOR full-text search, SQLAlchemy 2.x, Alembic
- **Background Workers**: Celery, Redis
- **Object Storage**: S3-compatible (MinIO / AWS S3) via `boto3`
- **Audio Infrastructure**: ffmpeg and ffprobe (safe subprocess inspection & 16kHz mono normalization)
- **Speech Intelligence**: `faster-whisper` (ASR), `pyannote.audio` (Diarization)
- **Retrieval Engine**: `sentence-transformers` (`BAAI/bge-small-en-v1.5`), `pgvector`, PostgreSQL Full-Text Search (GIN), Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking (`BAAI/bge-reranker-base`)
- **Tooling**: Docker, Docker Compose, Ruff, pytest

---

## 2. Audio Ingestion & Swagger Testing Workflow

### Direct Audio Upload via Swagger UI (`multipart/form-data`)
Sonara exposes a direct file upload endpoint optimized for interactive testing via Swagger UI:

```http
POST /api/v1/recordings/upload-direct
Authorization: Bearer <jwt_access_token>
Content-Type: multipart/form-data

Form fields:
- file: <binary audio file (wav, mp3, m4a, flac, ogg)>
- title: (Optional) "Executive Meeting Audio"
```

**Response (201 Created):**
```json
{
  "recording_id": "893c5240-5e3a-4efb-86ea-ca72a813876e",
  "job_id": "4b68db6b-67a1-4ee7-9759-9fce79860c04",
  "status": "uploaded",
  "file_size_bytes": 1048576,
  "audio_s3_key": "originals/893c5240-5e3a-4efb-86ea-ca72a813876e/sample.wav",
  "message": "Audio file successfully uploaded and queued for processing."
}
```

### Presigned Upload Workflow (Production Client PUT)
1. **Initiate**: `POST /api/v1/recordings/uploads` with JSON file metadata to receive a presigned PUT URL and staging key.
2. **Upload**: Client streams binary audio directly to the presigned S3/MinIO URL (`PUT`).
3. **Finalize**: `POST /api/v1/recordings/uploads/{upload_session_id}/finalize` to promote staging audio to canonical storage and enqueue ingestion.

---

## 3. Evidence-Grounded RAG & Search API Contracts

### Ask Question (Evidence-Grounded RAG)
```http
POST /api/v1/ask
Authorization: Bearer <jwt_access_token>
Content-Type: application/json

{
  "question": "What was the agreed project deadline?",
  "recording_id": "782ff774-6725-4523-a1bf-4fa9b42c8230",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "enable_reranking": true
}
```

**Response (200 OK):**
```json
{
  "question": "What was the agreed project deadline?",
  "status": "answered",
  "answer": "The project deadline was officially confirmed for December 15th [E1].",
  "citations": [
    {
      "evidence_id": "[E1]",
      "chunk_id": "c1f729e2-3490-4820-9114-61c925e01083",
      "recording_id": "782ff774-6725-4523-a1bf-4fa9b42c8230",
      "start_ms": 12000,
      "end_ms": 16000,
      "speaker_labels": ["SPEAKER_00"],
      "exact_text": "The project deadline was officially confirmed for December 15th."
    }
  ],
  "abstention_reason": null,
  "provider_name": "gemini",
  "model_name": "gemini-2.5-flash",
  "prompt_tokens": 340,
  "completion_tokens": 42,
  "request_id": "req-987abc"
}
```

### Search Recordings (Hybrid Retrieval)
```http
POST /api/v1/search
Authorization: Bearer <jwt_access_token>
Content-Type: application/json

{
  "query": "What were the quarterly revenue numbers?",
  "mode": "hybrid",
  "recording_id": "782ff774-6725-4523-a1bf-4fa9b42c8230",
  "speaker_filter": "SPEAKER_00",
  "top_k": 5,
  "enable_reranking": true
}
```
**Response (200 OK):**
```json
{
  "query": "What were the quarterly revenue numbers?",
  "mode": "hybrid",
  "total_results": 1,
  "results": [
    {
      "chunk_id": "c1f729e2-3490-4820-9114-61c925e01083",
      "recording_id": "782ff774-6725-4523-a1bf-4fa9b42c8230",
      "text": "In the third quarter, our total revenue reached twenty-four million dollars, representing a thirty percent increase year over year.",
      "start_ms": 14500,
      "end_ms": 23400,
      "token_count": 24,
      "source_segment_ids": [
        "e2a1464c-7c0b-4d4d-872f-518291a92e10"
      ],
      "speaker_labels": [
        "SPEAKER_00"
      ],
      "transcript_version": 1,
      "chunking_version": "v1.0",
      "index_version": 1,
      "semantic_rank": 1,
      "semantic_score": 0.9124,
      "keyword_rank": 1,
      "keyword_score": 1.4502,
      "rrf_score": 0.03278,
      "reranker_score": 0.941
    }
  ]
}
```

---

## 4. Maintenance, Lifecycle & Security Endpoints

### Secure Recording Deletion
```http
DELETE /api/v1/maintenance/recordings/{recording_id}
Authorization: Bearer <jwt_access_token>
```
Deletes database records (cascading to chunks, turns, segments, jobs) and cleans up S3/MinIO objects.

### Job Reconciliation
```http
POST /api/v1/maintenance/reconcile
Authorization: Bearer <jwt_admin_token>
```
Detects stranded background jobs where task dispatch failed, marks them as failed to maintain state consistency, and alerts operators.

---

## 5. Database Migrations


Apply database migrations:

```bash
cd backend
alembic upgrade head
```

---

## 6. Testing & Verification

Run automated tests:

```bash
python -m pytest backend/tests -v
```

Run linter:

```bash
python -m ruff check backend
```
