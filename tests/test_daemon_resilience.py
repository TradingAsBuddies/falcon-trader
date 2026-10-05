"""A failing cycle must not end the trading daemon.

PostgreSQL recycles every backend when one dies — a broken COPY pipe during a
backup was enough on 2026-10-05 — and the next query raised OperationalError,
which reached main() and exited 1. systemd restarted the container thirty
seconds later, so the session lost a cycle over a two-second database blip.
"""

import pytest

from falcon_trader import run_orchestrator as orch


class _Boom(Exception):
    """Stands in for psycopg2.OperationalError."""


def _run_cycles(outcomes):
    """Drive the guard the daemon uses, returning what happened per cycle.

    Mirrors the loop body rather than the whole daemon, which sleeps 300s.
    """
    consecutive_failures = 0
    log = []
    for cycle, ok in enumerate(outcomes, start=1):
        try:
            if not ok:
                raise _Boom("server closed the connection unexpectedly")
            consecutive_failures = 0
            log.append(("ok", cycle, consecutive_failures))
        except KeyboardInterrupt:                     # pragma: no cover
            raise
        except Exception:
            consecutive_failures += 1
            log.append(("failed", cycle, consecutive_failures))
            if consecutive_failures >= orch.MAX_CONSECUTIVE_CYCLE_FAILURES:
                log.append(("exited", cycle, consecutive_failures))
                break
    return log


def test_a_single_failure_does_not_stop_the_loop():
    log = _run_cycles([True, False, True, True])
    assert [entry[0] for entry in log] == ["ok", "failed", "ok", "ok"]
    assert "exited" not in [entry[0] for entry in log]


def test_the_failure_counter_resets_after_a_good_cycle():
    log = _run_cycles([False, False, True, False])
    assert log[-1] == ("failed", 4, 1)


def test_enough_failures_in_a_row_exits_for_a_restart():
    """A real breakage must not be hidden by retrying forever."""
    log = _run_cycles([False] * 10)
    assert log[-1][0] == "exited"
    assert log[-1][2] == orch.MAX_CONSECUTIVE_CYCLE_FAILURES


def test_the_threshold_rides_out_a_database_restart():
    """A restart costs one or two cycles; the threshold must exceed that."""
    assert orch.MAX_CONSECUTIVE_CYCLE_FAILURES >= 3


def test_the_guard_is_in_the_daemon_loop():
    """The loop body must be guarded, not just main()."""
    import inspect
    source = inspect.getsource(orch.main)
    assert "consecutive_failures" in source
    assert "MAX_CONSECUTIVE_CYCLE_FAILURES" in source
    # KeyboardInterrupt must stay re-raised so Ctrl-C still stops it.
    assert "except KeyboardInterrupt:" in source


def test_monitoring_and_processing_are_both_inside_the_guard():
    import inspect
    source = inspect.getsource(orch.main)
    guarded = source.split("try:")[-1]
    assert "process_screener_results" in guarded
    assert "monitor_positions" in guarded
