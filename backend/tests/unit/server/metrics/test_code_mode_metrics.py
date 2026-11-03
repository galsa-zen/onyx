"""Code mode capacity metrics, read as deltas from the default registry."""

from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

from onyx.server.metrics import code_mode as metrics
from onyx.tools.tool_implementations.code_mode import sandbox


def _value(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_program_end_counts_the_status_and_its_duration() -> None:
    programs = _value("onyx_code_mode_programs_total", {"status": "terminated"})
    durations = _value("onyx_code_mode_program_duration_seconds_count")
    total = _value("onyx_code_mode_program_duration_seconds_sum")

    metrics.observe_program_end("terminated", 2.5)

    assert (
        _value("onyx_code_mode_programs_total", {"status": "terminated"})
        == programs + 1
    )
    assert _value("onyx_code_mode_program_duration_seconds_count") == durations + 1
    assert _value("onyx_code_mode_program_duration_seconds_sum") == total + 2.5


def test_worker_losses_and_suspensions_count_by_label() -> None:
    killed = _value("onyx_code_mode_sandbox_workers_lost_total", {"reason": "killed"})
    over = _value("onyx_code_mode_suspensions_total", {"outcome": "over_budget"})

    metrics.count_worker_lost("killed")
    metrics.count_suspension("over_budget")

    assert (
        _value("onyx_code_mode_sandbox_workers_lost_total", {"reason": "killed"})
        == killed + 1
    )
    assert (
        _value("onyx_code_mode_suspensions_total", {"outcome": "over_budget"})
        == over + 1
    )


def test_the_pool_sets_every_limit_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_pool", None)
    with patch.object(sandbox, "Monty"):
        sandbox.sandbox_pool()
    assert _value("onyx_code_mode_sandbox_workers_limit") == (
        sandbox.MAX_SANDBOX_PROCESSES
    )
    assert _value("onyx_code_mode_suspended_bytes_limit") == (
        sandbox.MAX_SUSPENDED_BYTES
    )
    assert _value("onyx_code_mode_tool_call_threads_limit") == (
        sandbox.MAX_TOOL_CALL_THREADS
    )
