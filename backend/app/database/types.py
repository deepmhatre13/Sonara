from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.types import TypeDecorator


class FullTextSearchType(TypeDecorator):
    """
    SQLAlchemy TypeDecorator for PostgreSQL TSVECTOR.
    Falls back gracefully if used with non-PostgreSQL dialects in unit tests.
    """

    impl = TSVECTOR
    cache_ok = True


# Export pgvector Vector type wrapper
PgVector = Vector

