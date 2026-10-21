"""add_transcript_segments_and_speaker_turns

Revision ID: 0003_speech_pipeline
Revises: 0002_upload_pipeline
Create Date: 2026-10-19 23:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0003_speech_pipeline'
down_revision: Union[str, None] = '0002_upload_pipeline'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create transcript_segments table
    op.create_table(
        'transcript_segments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recording_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('recordings.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sequence_index', sa.Integer(), nullable=False),
        sa.Column('start_ms', sa.BigInteger(), nullable=False),
        sa.Column('end_ms', sa.BigInteger(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('language', sa.String(length=16), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('model_identifier', sa.String(length=128), nullable=False),
        sa.Column('transcript_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(op.f('ix_transcript_segments_recording_id'), 'transcript_segments', ['recording_id'], unique=False)
    op.create_index(op.f('ix_transcript_segments_start_ms'), 'transcript_segments', ['start_ms'], unique=False)

    # 2. Create speaker_turns table
    op.create_table(
        'speaker_turns',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recording_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('recordings.id', ondelete='CASCADE'), nullable=False),
        sa.Column('speaker_label', sa.String(length=64), nullable=False),
        sa.Column('start_ms', sa.BigInteger(), nullable=False),
        sa.Column('end_ms', sa.BigInteger(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('diarization_model', sa.String(length=128), nullable=False),
        sa.Column('turn_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(op.f('ix_speaker_turns_recording_id'), 'speaker_turns', ['recording_id'], unique=False)
    op.create_index(op.f('ix_speaker_turns_start_ms'), 'speaker_turns', ['start_ms'], unique=False)
    op.create_index(op.f('ix_speaker_turns_speaker_label'), 'speaker_turns', ['speaker_label'], unique=False)


def downgrade() -> None:
    op.drop_table('speaker_turns')
    op.drop_table('transcript_segments')

