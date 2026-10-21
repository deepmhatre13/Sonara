from app.core.celery_app import celery_app
from app.workers.tasks import health_check_task


def test_celery_task_registration():
    """Verify that tasks are properly registered with Celery."""
    assert "sonara.health_check" in celery_app.tasks
    task = celery_app.tasks["sonara.health_check"]
    assert task.name == "sonara.health_check"
    assert task.queue == "maintenance"


def test_celery_queues_configured():
    """Verify that all required queues are properly defined."""
    queue_names = {q.name for q in celery_app.conf.task_queues}
    expected_queues = {"ingest", "asr", "index", "delete", "maintenance", "eval"}
    assert expected_queues.issubset(queue_names)


def test_celery_health_check_task_execution():
    """
    Test direct local execution of health_check_task.
    Demonstrates task idempotency and structured output.
    """
    result = health_check_task.apply(
        kwargs={"message": "unit-test-ping", "correlation_id": "corr-456"}
    )
    assert result.successful()
    data = result.result
    assert data["status"] == "success"
    assert data["message"] == "unit-test-ping"
    assert data["correlation_id"] == "corr-456"

