import logging
from typing import Any

from celery import Task

from app.core.celery_app import celery_app

logger = logging.getLogger("sonara.tasks")


class BaseTask(Task):
    """
    Base Celery task providing standardized logging and retry handling.
    """

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="sonara.health_check",
    queue="maintenance",
)
def health_check_task(self, message: str = "ping", correlation_id: str | None = None) -> dict[str, Any]:
    """
    A lightweight, idempotent verification task to validate worker execution.
    """
    logger.info(
        f"[Task {self.request.id}] Executing health check. Msg: {message}, Correlation: {correlation_id}"
    )
    return {
        "status": "success",
        "task_id": self.request.id,
        "message": message,
        "correlation_id": correlation_id,
        "retries": self.request.retries,
    }

