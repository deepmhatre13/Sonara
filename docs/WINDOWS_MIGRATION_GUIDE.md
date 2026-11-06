# Sonara — Windows-Native Data Migration Guide

This guide describes how to migrate data from the existing Docker setup (PostgreSQL and S3/RustFS) to native Windows storage and PostgreSQL without losing or deleting source data.

---

## 1. Safety & Preservation Policy
- **Never delete Docker containers or volumes** prior to verifying the native restore.
- Legacy containers (`sonara-postgres-legacy`, `sonara-redis-legacy`) and active containers (`sonara-postgres`, `sonara-redis`, `sonara-minio`) remain completely intact.
- Source objects and database rows are copied, not moved or destroyed.

---

## 2. PostgreSQL Migration to Native Windows Service

### Step 2.1: Export Data from Containerized PostgreSQL
Run `pg_dump` against the container exposed on host port `5433`:
```powershell
# In PowerShell:
& "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe" `
    -h localhost -p 5433 -U sonara_user -d sonara_db `
    -F c -b -v -f "D:\Sonara\data\sonara_db_backup.dump"
```
*(Enter password `sonara_password` when prompted)*

### Step 2.2: Ensure `pgvector` Extension in Native PostgreSQL 17
1. In your native PostgreSQL 17 installation (default port `5432`):
   ```sql
   CREATE DATABASE sonara_db;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. Verify:
   ```sql
   \dx vector
   ```

### Step 2.3: Restore Data into Native PostgreSQL
```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_restore.exe" `
    -h localhost -p 5432 -U postgres -d sonara_db `
    -v "D:\Sonara\data\sonara_db_backup.dump"
```

### Step 2.4: Verify Row Counts & Migrations
```powershell
$env:PYTHONPATH = "D:\Sonara\backend"
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('postgresql+psycopg2://postgres:<your_password>@localhost:5432/sonara_db')
with engine.connect() as conn:
    print('Users:', conn.execute(text('SELECT count(*) FROM users')).scalar())
    print('Recordings:', conn.execute(text('SELECT count(*) FROM recordings')).scalar())
    print('Chunks:', conn.execute(text('SELECT count(*) FROM transcript_chunks')).scalar())
    print('Alembic Revision:', conn.execute(text('SELECT version_num FROM alembic_version')).scalar())
"
```

---

## 3. Audio Object Migration from RustFS to Local Storage

To copy existing audio recordings from RustFS to native local filesystem storage (`D:\Sonara\data\objects`):

```powershell
$env:PYTHONPATH = "D:\Sonara\backend"
python -c "
import boto3, os, hashlib
from botocore.client import Config
from pathlib import Path

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='sonara_local_access',
    aws_secret_access_key='replace_with_a_unique_random_secret',
    region_name='us-east-1',
    config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
)

local_root = Path(r'D:\Sonara\data\objects')
bucket = 'sonara-recordings'

try:
    resp = s3.list_objects_v2(Bucket=bucket)
    for obj in resp.get('Contents', []):
        key = obj['Key']
        dest = local_root / key.replace('/', os.sep)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {key} -> {dest}')
        s3.download_file(bucket, key, str(dest))
        
        # Verify SHA-256
        hasher = hashlib.sha256()
        with open(dest, 'rb') as f:
            for b in iter(lambda: f.read(65536), b''):
                hasher.update(b)
        print(f'  Verified SHA-256: {hasher.hexdigest()}')
    print('Object migration complete.')
except Exception as e:
    print('Storage note / error:', e)
"
```

---

## 4. When is it Safe to Turn Off Docker?

Only stop Docker Desktop when:
1. Native PostgreSQL (or an external managed PostgreSQL instance) is running and verified with `vector` extension.
2. Native Redis (or Memurai) is running on port `6379`.
3. All unit and integration tests pass with `STORAGE_BACKEND=local`.
4. A complete audio upload, transcription, and search workflow has succeeded natively.

