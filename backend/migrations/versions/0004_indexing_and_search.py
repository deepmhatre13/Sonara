"""add_transcript_chunks_and_search_indexes

Revision ID: 0004_indexing_and_search
Revises: 0003_speech_pipeline
Create Date: 2026-10-20 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '0004_indexing_and_search'
down_revision: Union[str, None] = '0003_speech_pipeline'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create transcript_chunks table
    op.create_table(
        'transcript_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recording_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('recordings.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('start_ms', sa.BigInteger(), nullable=False),
        sa.Column('end_ms', sa.BigInteger(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('source_segment_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('speaker_labels', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('embedding', Vector(384), nullable=True),
        sa.Column('embedding_model', sa.String(length=128), nullable=False),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False, server_default='384'),
        sa.Column('tsv', postgresql.TSVECTOR(), nullable=True),
        sa.Column('transcript_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('chunking_version', sa.String(length=32), nullable=False, server_default='v1.0'),
        sa.Column('index_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 2. Indexes for efficient lookup, filtering and ordering
    op.create_index(op.f('ix_transcript_chunks_recording_id'), 'transcript_chunks', ['recording_id'], unique=False)
    op.create_index(op.f('ix_transcript_chunks_start_ms'), 'transcript_chunks', ['start_ms'], unique=False)
    op.create_index(op.f('ix_transcript_chunks_index_version'), 'transcript_chunks', ['index_version'], unique=False)
    op.create_index(op.f('ix_transcript_chunks_is_active'), 'transcript_chunks', ['is_active'], unique=False)

    # 3. GIN index for PostgreSQL full-text search
    op.execute(
        "CREATE INDEX ix_transcript_chunks_tsv ON transcript_chunks USING GIN(tsv)"
    )

    # 4. Trigger to maintain tsvector automatically from chunk text
    op.execute("""
        CREATE OR REPLACE FUNCTION transcript_chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            new.tsv := to_tsvector('english', coalesce(new.text, ''));
            return new;
        END
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
        ON transcript_chunks FOR EACH ROW EXECUTE FUNCTION transcript_chunks_tsv_trigger();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS tsvectorupdate ON transcript_chunks")
    op.execute("DROP FUNCTION IF EXISTS transcript_chunks_tsv_trigger")
    op.execute("DROP INDEX IF EXISTS ix_transcript_chunks_tsv")
    op.drop_table('transcript_chunks')

