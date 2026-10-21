import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationRequiredError
from app.core.security import decode_access_token
from app.database.session import get_db_session
from app.models.core import User

security_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_bearer)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """
    Validates JWT Bearer token and returns authenticated User entity.
    """
    if not credentials:
        raise AuthenticationRequiredError("Bearer token missing.")

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
        user_id_str: str = payload.get("sub")
        if not user_id_str:
            raise AuthenticationRequiredError("Token payload missing subject.")
        user_id = uuid.UUID(user_id_str)
    except Exception as e:
        raise AuthenticationRequiredError(f"Invalid or expired token: {e}") from e

    stmt = select(User).where(User.id == user_id, User.is_active.is_(True))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationRequiredError("User account not found or inactive.")

    return user

