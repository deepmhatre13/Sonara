import json
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import EntityNotFoundError, ProviderUnavailableError, SonaraException
from app.llm import get_gemini_provider, get_groq_provider
from app.models.core import Recording, RecordingStatus
from app.rag.citation_validator import validate_citations
from app.rag.context_builder import build_rag_messages
from app.rag.evidence_selector import select_evidence
from app.retrieval.hybrid import retrieve_hybrid
from app.retrieval.keyword import retrieve_keyword
from app.retrieval.semantic import retrieve_semantic
from app.schemas.rag import (
    AnswerStatus,
    AskRequest,
    AskResponse,
    StructuredAnswerPayload,
)
from app.schemas.search import RetrievalMode

logger = logging.getLogger("sonara.rag.service")


class RAGService:
    """
    Coordinates evidence retrieval, context construction, LLM generation (Gemini with Groq fallback),
    server-side citation validation, and strict abstention enforcement.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def answer_question(
        self,
        request: AskRequest,
        user_id: uuid.UUID,
        request_id: str | None = None,
    ) -> AskResponse:
        # 1. Validate recording scope and ownership if recording_id is provided
        if request.recording_id is not None:
            rec = await self.db.get(Recording, request.recording_id)
            if not rec:
                raise EntityNotFoundError("Recording", str(request.recording_id))
            if rec.user_id != user_id:
                raise SonaraException(
                    status_code=403,
                    detail="You do not have permission to access this recording.",
                    code="RECORDING_ACCESS_DENIED",
                )
            if rec.status == RecordingStatus.DELETING:
                raise SonaraException(
                    status_code=409,
                    detail="Recording is currently being deleted and cannot be queried.",
                    code="RECORDING_DELETING",
                )


        # 2. Retrieve evidence chunks using existing retrieval subsystem
        retrieval_k = max(request.top_k, 5)
        if request.retrieval_mode == RetrievalMode.KEYWORD:
            raw_candidates = await retrieve_keyword(
                db=self.db,
                query=request.question,
                user_id=user_id,
                recording_id=request.recording_id,
                top_k=retrieval_k,
            )
            # Map KeywordCandidate to RetrievedEvidence format
            from app.retrieval.hybrid import RetrievedEvidence
            candidates = [
                RetrievedEvidence(
                    chunk_id=c.chunk.id,
                    recording_id=c.chunk.recording_id,
                    text=c.chunk.text,
                    start_ms=c.chunk.start_ms,
                    end_ms=c.chunk.end_ms,
                    token_count=c.chunk.token_count,
                    source_segment_ids=c.chunk.source_segment_ids,
                    speaker_labels=c.chunk.speaker_labels,
                    transcript_version=c.chunk.transcript_version,
                    chunking_version=c.chunking_version,
                    index_version=c.chunk.index_version,
                    keyword_rank=c.rank,
                    keyword_score=c.score,
                )
                for c in raw_candidates
            ]
        elif request.retrieval_mode == RetrievalMode.SEMANTIC:
            raw_candidates = await retrieve_semantic(
                db=self.db,
                query=request.question,
                user_id=user_id,
                recording_id=request.recording_id,
                top_k=retrieval_k,
            )
            from app.retrieval.hybrid import RetrievedEvidence
            candidates = [
                RetrievedEvidence(
                    chunk_id=c.chunk.id,
                    recording_id=c.chunk.recording_id,
                    text=c.chunk.text,
                    start_ms=c.chunk.start_ms,
                    end_ms=c.chunk.end_ms,
                    token_count=c.chunk.token_count,
                    source_segment_ids=c.chunk.source_segment_ids,
                    speaker_labels=c.chunk.speaker_labels,
                    transcript_version=c.chunk.transcript_version,
                    chunking_version=c.chunking_version,
                    index_version=c.chunk.index_version,
                    semantic_rank=c.rank,
                    semantic_score=c.score,
                )
                for c in raw_candidates
            ]
        else:  # Hybrid
            candidates = await retrieve_hybrid(
                db=self.db,
                query=request.question,
                user_id=user_id,
                recording_id=request.recording_id,
                top_k=retrieval_k,
                enable_reranking=request.enable_reranking,
            )

        # 3. Evidence selection and bounding
        selected_evidence = select_evidence(
            candidates=candidates,
            max_items=settings.MAX_EVIDENCE_ITEMS,
            max_context_tokens=settings.MAX_CONTEXT_TOKENS,
        )

        # Immediate abstention if zero evidence found
        if not selected_evidence:
            return AskResponse(
                question=request.question,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer="No relevant recording transcript evidence was found to answer this question.",
                citations=[],
                abstention_reason="No retrieval candidates matched the query within authorized recordings.",
                provider_name="none",
                model_name="none",
                request_id=request_id,
            )

        # 4. Construct prompt-injection-resistant message payload
        messages = build_rag_messages(
            question=request.question,
            evidence_items=selected_evidence,
        )

        # 5. Execute generation via Gemini with optional Groq fallback
        gemini = get_gemini_provider()
        groq = get_groq_provider()

        llm_response = None
        provider_used = None

        if gemini.is_configured:
            try:
                llm_response = await gemini.generate_response(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                    response_schema=StructuredAnswerPayload,
                )
                provider_used = gemini
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}. Checking Groq fallback...")

        if llm_response is None and groq.is_configured:
            try:
                llm_response = await groq.generate_response(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                    response_schema=StructuredAnswerPayload,
                )
                provider_used = groq
            except Exception as e:
                logger.error(f"Groq fallback generation failed: {e}")

        if llm_response is None:
            raise ProviderUnavailableError(
                provider_name="gemini/groq",
                detail="All configured LLM providers are currently unavailable or unconfigured.",
            )

        # 6. Parse structured response
        raw_text = llm_response.content.strip()
        parsed_payload = None

        try:
            # Try parsing JSON structure
            clean_json = raw_text
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:-3].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:-3].strip()

            parsed_dict = json.loads(clean_json)
            parsed_payload = StructuredAnswerPayload.model_validate(parsed_dict)
        except Exception:
            # Fallback to plain answer text if JSON parse fails
            parsed_payload = StructuredAnswerPayload(
                answer=raw_text,
                citations=[],
                insufficient_evidence=False,
            )

        # 7. Server-side citation validation
        validated_citations = validate_citations(
            cited_labels=parsed_payload.citations,
            evidence_items=selected_evidence,
            answer_text=parsed_payload.answer,
        )

        # 8. Enforce abstention policy
        is_abstention = parsed_payload.insufficient_evidence or (
            len(validated_citations) == 0 and "insufficient" in parsed_payload.answer.lower()
        )

        final_status = AnswerStatus.INSUFFICIENT_EVIDENCE if is_abstention else AnswerStatus.ANSWERED

        return AskResponse(
            question=request.question,
            status=final_status,
            answer=parsed_payload.answer,
            citations=validated_citations if not is_abstention else [],
            abstention_reason=parsed_payload.abstention_reason if is_abstention else None,
            provider_name=provider_used.provider_name if provider_used else "unknown",
            model_name=provider_used.model_name if provider_used else "unknown",
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            request_id=request_id,
        )

