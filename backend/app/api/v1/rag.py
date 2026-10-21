import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.database.session import get_db_session
from app.models.core import User
from app.rag.service import RAGService
from app.schemas.rag import AskRequest, AskResponse

logger = logging.getLogger("sonara.api.rag")

router = APIRouter(prefix="/ask", tags=["Evidence-Grounded RAG & Question Answering"])


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against audio transcripts",
    description=(
        "Retrieves transcript evidence, formats a prompt-injection-resistant context, "
        "queries Gemini (with Groq fallback), validates citations against authoritative "
        "audio timestamps, and strictly abstains if evidence is insufficient."
    ),
)
async def ask_question(
    payload: AskRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    http_request: Request,
) -> AskResponse:
    request_id = getattr(http_request.state, "request_id", None)
    rag_service = RAGService(db=db)
    return await rag_service.answer_question(
        request=payload,
        user_id=current_user.id,
        request_id=request_id,
    )

