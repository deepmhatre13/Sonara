"""add_upload_sessions_and_recording_status

Revision ID: 0002_upload_pipeline
Revises: 0001_initial_schema
Create Date: 2026-10-19 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0002_upload_pipeline'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add new columns to recordings
    op.add_column('recordings', sa.Column('normalized_s3_key', sa.String(length=512), nullable=True))
    op.add_column('recordings', sa.Column('status', sa.String(length=32), nullable=False, server_default='awaiting_upload'))
    op.add_column('recordings', sa.Column('idempotency_key', sa.String(length=128), nullable=True))
    op.add_column('recordings', sa.Column('codec', sa.String(length=64), nullable=True))
    op.add_column('recordings', sa.Column('checksum_sha256', sa.String(length=64), nullable=True))
    op.add_column('recordings', sa.Column('normalization_profile', sa.String(length=64), nullable=True))
    op.add_column('recordings', sa.Column('pipeline_version', sa.String(length=32), nullable=True))
    op.add_column('recordings', sa.Column('error_message', sa.Text(), nullable=True))

    op.create_index(op.f('ix_recordings_status'), 'recordings', ['status'], unique=False)
    op.create_index(op.f('ix_recordings_idempotency_key'), 'recordings', ['idempotency_key'], unique=False)

    # 2. Create upload_sessions table
    op.create_table(
        'upload_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recording_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('recordings.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('staging_s3_key', sa.String(length=512), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('declared_mime_type', sa.String(length=64), nullable=False),
        sa.Column('expected_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('expected_checksum', sa.String(length=64), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_finalized', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(op.f('ix_upload_sessions_recording_id'), 'upload_sessions', ['recording_id'], unique=True)
    op.create_index(op.f('ix_upload_sessions_user_id'), 'upload_sessions', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('upload_sessions')
    op.drop_index(op.f('ix_recordings_idempotency_key'), table_name='recordings')
    op.drop_index(op.f('ix_recordings_status'), table_name='recordings')
    op.drop_column('recordings', 'error_message')
    op.drop_column('recordings', 'pipeline_version')
    op.drop_column('recordings', 'normalization_profile')
    op.drop_column('recordings', 'checksum_sha256')
    op.drop_column('recordings', 'codec')
    op.drop_column('recordings', 'idempotency_key')
    op.drop_column('recordings', 'status')
    op.drop_column('recordings', 'normalized_s3_key')

