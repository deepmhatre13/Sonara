from celery import Celery
from kombu import Queue

from app.core.config import settings

celery_app = Celery(
    "sonara_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Celery Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 min soft limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,  # Ensure task is acknowledged only after successful completion
    task_reject_on_worker_lost=True,
)

# Explicit Queue Routing
celery_app.conf.task_queues = (
    Queue("ingest", routing_key="ingest.#"),
    Queue("asr", routing_key="asr.#"),
    Queue("index", routing_key="index.#"),
    Queue("delete", routing_key="delete.#"),
    Queue("maintenance", routing_key="maintenance.#"),
    Queue("eval", routing_key="eval.#"),
)

celery_app.conf.task_default_queue = "ingest"

# Auto-discover tasks in workers module
celery_app.autodiscover_tasks(["app.workers"])

