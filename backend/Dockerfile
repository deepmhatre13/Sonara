FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (ffmpeg for media processing, libpq for PostgreSQL, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libpq5 \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first for optimal Docker layer caching
COPY pyproject.toml requirements.txt ./

# Configure pip timeout and retry limits for reliable downloads on slow or fluctuating network connections
RUN pip install --upgrade pip \
    && pip install --default-timeout=120 --retries=10 -e ".[dev]" \
    && apt-get purge -y --auto-remove build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY . /app/

# Create a non-privileged system user
RUN useradd -m -u 1000 sonarauser && chown -R sonarauser:sonarauser /app
USER sonarauser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
