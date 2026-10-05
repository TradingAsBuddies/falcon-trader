"""The performance tracker's table creation, on both backends.

AUTOINCREMENT is SQLite-only. Against PostgreSQL the first CREATE raised
`syntax error at or near "AUTOINCREMENT"`, the except swallowed it, and the
other two statements never ran. It only worked on this deployment because the
three tables already existed; a fresh PostgreSQL database would have had none.
"""

import pytest

from falcon_trader.orchestrator.monitors.performance_tracker import PerformanceTracker

TABLES = ("routing_decisions", "trade_tracking", "strategy_metrics")


class _FakeDB:
    def __init__(self, db_type):
        self.db_type = db_type
        self.statements = []

    def execute(self, query, params=None, fetch=None):
        if query.strip().upper().startswith("CREATE TABLE"):
            self.statements.append(query)
        return None


def _create(db_type):
    tracker = PerformanceTracker.__new__(PerformanceTracker)
    tracker.db = _FakeDB(db_type)
    tracker._create_tables()
    return tracker.db.statements


@pytest.mark.parametrize("db_type", ["postgresql", "sqlite"])
def test_all_three_tables_are_created(db_type):
    """The regression: a failure on the first statement skipped the rest."""
    statements = _create(db_type)
    assert len(statements) == 3
    for table in TABLES:
        assert any(table in s for s in statements), table


def test_postgres_uses_serial_and_never_autoincrement():
    statements = _create("postgresql")
    assert all("SERIAL PRIMARY KEY" in s for s in statements)
    assert not any("AUTOINCREMENT" in s for s in statements)


def test_sqlite_keeps_autoincrement():
    statements = _create("sqlite")
    assert all("AUTOINCREMENT" in s for s in statements)
    assert not any("SERIAL" in s for s in statements)


def test_an_unknown_backend_is_treated_as_postgres():
    """DatabaseManager without db_type must not emit SQLite-only syntax."""
    tracker = PerformanceTracker.__new__(PerformanceTracker)

    class _NoType:
        def execute(self, query, params=None, fetch=None):
            self.last = query
            return None

    tracker.db = _NoType()
    tracker._create_tables()
    assert "AUTOINCREMENT" not in tracker.db.last


def test_creation_is_idempotent_sql():
    for statement in _create("postgresql"):
        assert "IF NOT EXISTS" in statement
