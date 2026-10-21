# Architecture Documentation: Sonara Backend

## 1. System Overview

Sonara is a high-performance, production-oriented system for audio ingestion, transcription, speaker diarization, hybrid retrieval, and evidence-grounded question answering (RAG).

The system is organized as a **Modular Monolith** paired with dedicated asynchronous background workers.

```
                          [ Client Applications ]
                                    │
                                    ▼ (HTTPS / REST)
                       ┌─────────────────────────┐
                       │   FastAPI Web Service   │
                       │   (/api/v1/search &     │
                       │    /api/v1/recordings)  │
                       └────────────┬────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  PostgreSQL   │           │     Redis     │           │  MinIO / S3   │
│ (Jobs, Audio, │           │  (Broker &    │           │ (Raw Staging  │
│  Transcripts, │           │   Results)    │           │ & Normalized) │
│  pgvector,    │           │               │           │               │
│  TSVECTOR)    │           │               │           │               │
└───────▲───────┘           └───────┬───────┘           └───────▲───────┘
        │                           │                           │
        │                           ▼                           │
        │                   ┌───────────────┐                   │
        └───────────────────┤ Celery Worker ├───────────────────┘
                            │ (ingest, asr  │
                            │  index: bge/  │
                            │  sentence-tx) │
                            └───────────────┘
```

---

## 2. Semantic Indexing & Hybrid Retrieval Subsystem

### A. Deterministic Transcript Chunking
- **Engine**: `app/chunking/chunker.py`.
- **Strategy**: Groups contiguous transcript segments by sequence index and start time into token-bounded retrieval units (`CHUNK_MAX_TOKENS = 250`, `CHUNK_OVERLAP_TOKENS = 40`).
- **Provenance**: Retains exact ordered source transcript segment IDs (`source_segment_ids`) and overlapping speaker labels (`speaker_labels`).
- **Timeline Invariant**: Start and end timestamps are preserved as integer milliseconds (`start_ms`, `end_ms`).

### B. Dense Vector Embedding Generation
- **Model**: `BAAI/bge-small-en-v1.5` via `SentenceTransformer`.
- **Dimension**: 384 dimensions.
- **Normalization**: Unit $L_2$ normalized embeddings allowing cosine distance computation via inner dot product.
- **pgvector Storage**: Persisted in PostgreSQL using the `vector(384)` type with indexing support.

### C. Lexical Retrieval (PostgreSQL Full-Text Search)
- **Engine**: PostgreSQL `TSVECTOR` with GIN indexing (`ix_transcript_chunks_tsv`).
- **Trigger**: Automatic `tsvectorupdate` database trigger generating English lexical stems on insert/update.
- **Query Parser**: Parameterized `plainto_tsquery('english', :query)` with `ts_rank` scoring.

### D. Reciprocal Rank Fusion (RRF)
- **Formula**: $RRF\_score(d) = \sum_{r \in \{semantic, keyword\}} \frac{1}{k + rank_r(d)}$ with configurable constant $k = 60$.
- **Score Fusion**: Combines independent 1-based retriever rankings rather than uncalibrated raw distance or BM25 scores.
- **Determinism**: Deduplicates candidates and sorts by RRF score descending, tie-breaking by chunk ID.

### E. Optional Cross-Encoder Reranking
- **Model**: `BAAI/bge-reranker-base` via `SentenceTransformers CrossEncoder`.
- **Target**: Reranks top $N$ hybrid candidates to maximize contextual precision.

### F. Evaluation Metrics
- **Module**: `app/evaluation/retrieval.py`.
- **Metrics**: Standard Information Retrieval metrics (Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG@K).

---

## 3. Evidence-Grounded RAG Subsystem

### A. Context Architecture & Prompt Injection Defense
- **Untrusted Audio Transcripts**: All retrieved chunk texts are treated strictly as untrusted user-supplied data.
- **XML Delimitation**: Chunks are wrapped in explicit XML tags `<evidence id="[E#]" recording_id="..." start_ms="..." end_ms="..." speakers="...">...</evidence>`.
- **System Directives**: The system prompt instructs the model to ignore any instructions or attempts to bypass system constraints embedded within evidence snippets, enforcing strict reliance only on verifiable factual statements.

### B. LLM Providers & Fallback Hierarchy
- **Primary Provider**: Google Gemini (`gemini-2.5-flash`) via the modern `google.genai` SDK using structured JSON schema output and bounded execution timeouts.
- **Resilient Fallback**: Groq (`llama-3.3-70b-versatile`) via the official `groq` SDK providing fast fallback if Gemini encounters rate limits or service unavailability.
- **Stateless Execution**: The RAG subsystem is strictly stateless; every question evaluation retrieves relevant chunks and injects evidence fresh without conversational contamination.

### C. Server-Side Citation Validation & Abstention
- **Deterministic Numbering**: Chunks are assigned identifiers `[E1]`, `[E2]`, etc., before prompt assembly.
- **Authoritative Invariants**: The model outputs cited labels (e.g., `["[E1]"]`); the server resolves exact chunk IDs, recording IDs, and start/end millisecond timestamps from persisted database entities. Models cannot fabricate timestamps or non-existent chunk IDs.
- **Strict Abstention**: If retrieval yields 0 candidates, the LLM signals `insufficient_evidence=true`, or zero valid citations are verified for an unanswerable question, the system sets `status="insufficient_evidence"` and declines to invent information.

### D. RAG Quality Evaluation
- **Module**: `app/evaluation/rag.py`.
- **Metrics**: Evaluates citation precision, citation recall, citation F1, groundedness score, abstention accuracy, and hallucinated citation detection.

---

## 4. Storage Key Conventions

- **Staging Object**: `staging/{recording_id}/{upload_session_id}`
- **Immutable Source**: `originals/{recording_id}/source`
- **Normalized Speech Audio**: `processed/{recording_id}/{pipeline_version}/audio.wav`

---

## 4. Processing Lifecycle & State Machine

```
[AWAITING_UPLOAD]
       │
       ▼ (PUT + Finalize)
  [UPLOADED]
       │
       ▼ (Ingest worker)
 [VALIDATING] ──► [NORMALIZING] ──► [INGESTION_COMPLETE]
                                            │
                                            ▼ (ASR worker dispatched)
                                      [TRANSCRIBING]
                                            │
                                            ▼
                                       [DIARIZING]
                                            │
                                            ▼
                                    [SPEECH_COMPLETE]
                                            │
                                            ▼ (Index worker dispatched)
                                       [INDEXING]
                                            │
                                            ▼
                                      [SEARCHABLE]
```
