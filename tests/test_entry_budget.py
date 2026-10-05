"""The per-session entry budget.

Without it, the first cycle of a session spends the entire cash balance: the
screener surfaces a dozen-plus candidates and on 2026-10-05 the engines
signalled BUY on 12 of them at ~$1,490 notional each against $7,835 of cash.
With 12-20 day holds there would then be nothing to trade with for days, which
is the opposite of trading daily.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from falcon_trader import trading_guards
from falcon_trader.orchestrator.execution.trade_executor import TradeExecutor

ET = ZoneInfo("America/New_York")


class _CountDB:
    def __init__(self, count, row_style="mapping"):
        self.count = count
        self.row_style = row_style
        self.calls = []

    def execute(self, query, params=None, fetch=None):
        self.calls.append((query, params, fetch))
        if self.row_style == "tuple":
            return (self.count,)
        if self.row_style == "none":
            return None
        return {"entries": self.count}


# ── session boundary ────────────────────────────────────────────────────

def test_session_start_is_eastern_midnight_expressed_in_utc():
    """EDT: 00:00 ET is 04:00 UTC the same day."""
    moment = dt.datetime(2026, 10, 5, 10, 30, tzinfo=ET)
    assert trading_guards.session_start_utc(moment) == dt.datetime(2026, 10, 5, 4, 0)


def test_session_start_in_winter_uses_est():
    """EST: 00:00 ET is 05:00 UTC."""
    moment = dt.datetime(2026, 12, 1, 10, 30, tzinfo=ET)
    assert trading_guards.session_start_utc(moment) == dt.datetime(2026, 12, 1, 5, 0)


def test_late_evening_still_counts_todays_session():
    """At 21:00 ET a naive local date would already be tomorrow in UTC."""
    moment = dt.datetime(2026, 10, 5, 21, 0, tzinfo=ET)
    assert trading_guards.session_start_utc(moment) == dt.datetime(2026, 10, 5, 4, 0)


def test_the_count_is_bound_as_a_parameter():
    db = _CountDB(0)
    moment = dt.datetime(2026, 10, 5, 10, 30, tzinfo=ET)
    trading_guards.entries_today(db, moment)
    query, params, _ = db.calls[0]
    assert "interval" not in query.lower() and "now()" not in query.lower()
    assert params == ("BUY", dt.datetime(2026, 10, 5, 4, 0))


# ── counting ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("style", ["mapping", "tuple"])
def test_entries_today_reads_either_row_shape(style):
    assert trading_guards.entries_today(_CountDB(4, style)) == 4


def test_no_row_counts_as_no_entries():
    assert trading_guards.entries_today(_CountDB(0, "none")) == 0


def test_the_count_comes_from_the_database_not_memory():
    """A restart mid-session must not hand out a fresh budget."""
    db = _CountDB(3)
    assert trading_guards.check_entry_budget(db, 3).allowed is False
    assert "orders" in db.calls[0][0].lower()


# ── the guard ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("used,budget,allowed", [
    (0, 3, True), (1, 3, True), (2, 3, True),
    (3, 3, False), (4, 3, False),
])
def test_budget_allows_up_to_its_limit(used, budget, allowed):
    assert trading_guards.check_entry_budget(_CountDB(used), budget).allowed is allowed


def test_blocked_result_says_how_many_and_what_the_budget_is():
    result = trading_guards.check_entry_budget(_CountDB(3), 3)
    assert result.reason == "entry_budget_spent"
    assert "3 entries already placed" in result.message
    assert "budget is 3" in result.message


def test_one_entry_reads_as_singular():
    assert "1 entry already placed" in trading_guards.check_entry_budget(_CountDB(1), 1).message


@pytest.mark.parametrize("budget", [0, -1, None])
def test_a_non_positive_budget_is_unlimited(budget):
    db = _CountDB(99)
    assert trading_guards.check_entry_budget(db, budget).allowed is True
    assert db.calls == []  # not even queried


def test_default_budget():
    assert trading_guards.DEFAULT_MAX_NEW_ENTRIES_PER_DAY == 3


# ── the executor's knob ─────────────────────────────────────────────────

def _executor(config):
    executor = TradeExecutor.__new__(TradeExecutor)
    executor.config = config
    return executor


def test_executor_reads_the_config(monkeypatch):
    monkeypatch.delenv("FALCON_MAX_NEW_ENTRIES", raising=False)
    assert _executor({"session": {"max_new_entries_per_day": 5}}).max_new_entries_per_day() == 5


@pytest.mark.parametrize("config", [{}, {"session": {}}])
def test_executor_falls_back_to_the_default(config, monkeypatch):
    monkeypatch.delenv("FALCON_MAX_NEW_ENTRIES", raising=False)
    assert _executor(config).max_new_entries_per_day() == 3


def test_env_overrides_the_config(monkeypatch):
    monkeypatch.setenv("FALCON_MAX_NEW_ENTRIES", "7")
    assert _executor({"session": {"max_new_entries_per_day": 2}}).max_new_entries_per_day() == 7


def test_unlimited_via_env(monkeypatch):
    monkeypatch.setenv("FALCON_MAX_NEW_ENTRIES", "0")
    assert _executor({}).max_new_entries_per_day() == 0


def test_garbage_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("FALCON_MAX_NEW_ENTRIES", "lots")
    assert _executor({}).max_new_entries_per_day() == 3


def test_shipped_config_budgets_three():
    import yaml
    from pathlib import Path
    import falcon_trader
    path = Path(falcon_trader.__file__).parent / "orchestrator" / "orchestrator_config.yaml"
    assert yaml.safe_load(path.read_text())["session"]["max_new_entries_per_day"] == 3


# ── stopping the loop ───────────────────────────────────────────────────

class _BudgetExecutor(TradeExecutor):
    """process_recommendations with the per-stock work stubbed out."""

    def __init__(self, budget, already_used):
        self.config = {"session": {"max_new_entries_per_day": budget}}
        self.db = _CountDB(already_used)
        self.processed = []

    def process_stock(self, symbol, rec=None):
        self.processed.append(symbol)
        self.db.count += 1  # as a filled BUY would
        return {"symbol": symbol, "success": True, "action": "BUY"}


def test_processing_stops_when_the_budget_is_spent(monkeypatch):
    monkeypatch.delenv("FALCON_MAX_NEW_ENTRIES", raising=False)
    executor = _BudgetExecutor(budget=3, already_used=0)
    recs = [{"ticker": t} for t in ("A", "B", "C", "D", "E")]

    summary = executor.process_recommendations(recs)

    assert executor.processed == ["A", "B", "C"]
    assert summary["trades_executed"] == 3
    assert summary["skipped"] == 2
    assert summary["total_stocks"] == 5
    assert "budget is 3" in summary["budget_reached"]


def test_a_session_that_already_traded_gets_what_is_left(monkeypatch):
    monkeypatch.delenv("FALCON_MAX_NEW_ENTRIES", raising=False)
    executor = _BudgetExecutor(budget=3, already_used=2)
    summary = executor.process_recommendations([{"ticker": t} for t in ("A", "B", "C")])
    assert executor.processed == ["A"]
    assert summary["skipped"] == 2


def test_unlimited_budget_processes_everything(monkeypatch):
    monkeypatch.delenv("FALCON_MAX_NEW_ENTRIES", raising=False)
    executor = _BudgetExecutor(budget=0, already_used=0)
    executor.process_recommendations([{"ticker": t} for t in ("A", "B", "C", "D")])
    assert executor.processed == ["A", "B", "C", "D"]
