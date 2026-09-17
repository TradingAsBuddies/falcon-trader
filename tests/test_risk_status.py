"""Tests for the risk panel's status report and the engine's kill switch.

No Flask, database or container. mountinfo fixtures are the real lines read from
the falcon-dashboard and falcon-trader containers on 2026-09-17.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

from falcon_trader import risk_status
from falcon_trader.risk_limits import KillSwitch
from falcon_trader.trading_guards import CooldownPolicy

# falcon-trader: /var/lib/falcon is the falcon-data volume.
TRADER_MOUNTINFO = """\
1234 1200 0:52 / / rw,relatime - overlay overlay rw
1250 1234 253:1 /volumes/falcon-data/_data /var/lib/falcon rw,relatime - btrfs /dev/dm-1 rw
1251 1234 0:5 / /proc rw,nosuid - proc proc rw
"""

# falcon-dashboard: nothing mounted at /var/lib/falcon.
DASHBOARD_MOUNTINFO = """\
1334 1300 0:60 / / rw,relatime - overlay overlay rw
1351 1334 0:5 / /proc rw,nosuid - proc proc rw
"""

HALT_FILE = "/var/lib/falcon/TRADING_HALTED"


# ── is_mount_point_or_below ─────────────────────────────────────────────

def test_halt_file_under_a_volume_is_shared():
    assert risk_status.is_mount_point_or_below(HALT_FILE, TRADER_MOUNTINFO)


def test_halt_file_on_container_root_is_not_shared():
    """The real dashboard container, which is why the warning exists."""
    assert not risk_status.is_mount_point_or_below(HALT_FILE, DASHBOARD_MOUNTINFO)


def test_container_root_mount_does_not_count_as_shared():
    """Everything is 'below /'; that must not read as a shared mount."""
    assert not risk_status.is_mount_point_or_below("/anything/at/all", "1 0 0:1 / / rw - overlay o rw")


def test_unrelated_mount_does_not_count():
    assert not risk_status.is_mount_point_or_below(HALT_FILE, TRADER_MOUNTINFO.replace("/var/lib/falcon", "/srv/other"))


def test_similar_prefix_is_not_a_parent():
    """/var/lib/falcon-other must not be treated as containing /var/lib/falcon/…"""
    mi = "1 0 0:1 / /var/lib/falcon-other rw - btrfs x rw"
    assert not risk_status.is_mount_point_or_below(HALT_FILE, mi)


def test_unreadable_mountinfo_is_treated_as_not_shared():
    """Fail towards the warning, not towards false reassurance."""
    assert not risk_status.is_mount_point_or_below(HALT_FILE, "")


# ── kill_switch_view ────────────────────────────────────────────────────

def _ks_status(**over):
    base = {"trading_enabled": True, "reason": None, "env_enabled": True,
            "halt_file": HALT_FILE, "halt_file_present": False, "in_process_halt": None}
    base.update(over)
    return base


def test_warning_present_when_halt_file_is_local():
    view = risk_status.kill_switch_view(_ks_status(), DASHBOARD_MOUNTINFO)
    assert view["halt_file_on_shared_mount"] is False
    assert "not visible to the trader" in view["reach_warning"]


def test_no_warning_when_halt_file_is_shared():
    view = risk_status.kill_switch_view(_ks_status(), TRADER_MOUNTINFO)
    assert view["halt_file_on_shared_mount"] is True
    assert "reach_warning" not in view


# ── KillSwitch.status ───────────────────────────────────────────────────

def test_status_reports_each_source_separately(tmp_path):
    ks = KillSwitch(halt_file=tmp_path / "HALT", env={"FALCON_TRADING_ENABLED": "1"})
    assert ks.status() == {
        "trading_enabled": True, "reason": None, "env_enabled": True,
        "halt_file": str(tmp_path / "HALT"), "halt_file_present": False,
        "in_process_halt": None,
    }


def test_status_names_the_env_source(tmp_path):
    ks = KillSwitch(halt_file=tmp_path / "HALT", env={"FALCON_TRADING_ENABLED": "0"})
    st = ks.status()
    assert st["trading_enabled"] is False
    assert st["env_enabled"] is False
    assert st["halt_file_present"] is False


def test_status_names_the_file_source(tmp_path):
    halt = tmp_path / "HALT"
    halt.write_text("x")
    st = KillSwitch(halt_file=halt, env={}).status()
    assert st["trading_enabled"] is False and st["halt_file_present"] is True


def test_status_does_not_change_state(tmp_path):
    ks = KillSwitch(halt_file=tmp_path / "HALT", env={})
    ks.status(); ks.status()
    assert ks.is_trading_enabled()
    assert not (tmp_path / "HALT").exists()


# ── cooldown / symbols ──────────────────────────────────────────────────

NOW = dt.datetime(2026, 9, 17, 11, 0)


def _state(symbol, *, exit_hours_ago=None, pnl=None, streak=0, rts=0):
    last = NOW - dt.timedelta(hours=exit_hours_ago) if exit_hours_ago is not None else None
    return SimpleNamespace(symbol=symbol, last_exit_at=last, last_exit_pnl=pnl,
                           loss_streak=streak, round_trips_today=rts)


def test_symbol_never_traded_is_allowed():
    rows = risk_status.symbol_rows([_state("AAPL")], CooldownPolicy(), held=["AAPL"], now=NOW)
    assert rows[0]["entry_allowed"] is True and rows[0]["held"] is True


def test_recent_exit_is_blocked_by_cooldown():
    rows = risk_status.symbol_rows([_state("CDXS", exit_hours_ago=2, pnl=-277.0)],
                                   CooldownPolicy(), held=[], now=NOW)
    assert rows[0]["entry_allowed"] is False
    assert rows[0]["block_reason"]


def test_blocked_symbols_sort_first():
    rows = risk_status.symbol_rows(
        [_state("ZZZ"), _state("CDXS", exit_hours_ago=2)], CooldownPolicy(), held=[], now=NOW,
    )
    assert [r["symbol"] for r in rows] == ["CDXS", "ZZZ"]


def test_policy_view_is_marked_enforced():
    view = risk_status.cooldown_policy_view(CooldownPolicy())
    assert view["enforced"] is True
    assert view["cooldown_seconds"] == 86400
    assert view["max_round_trips_per_day"] == 1


# ── the honesty guarantees ──────────────────────────────────────────────

def test_pre_trade_limits_are_reported_as_not_enforced():
    """The whole point: a daily loss limit that blocks nothing must say so."""
    block = risk_status.defined_not_enforced()
    assert block["enforced"] is False
    assert "max_daily_loss_pct" in block["limits"]
    assert "check_pre_trade is not called" in block["note"]


def test_exits_are_listed_as_never_blocked():
    paths = [p["path"] for p in risk_status.NEVER_BLOCKED]
    assert "exits and sells" in paths


def test_das_is_listed_as_unreachable_not_enforced():
    enforced = {p["path"] for p in risk_status.ENFORCED_ON}
    unreachable = {p["path"] for p in risk_status.NOT_REACHABLE}
    assert "DAS execution" in unreachable
    assert "DAS execution" not in enforced


def test_build_status_shape():
    st = risk_status.build_status(_ks_status(), DASHBOARD_MOUNTINFO, CooldownPolicy(),
                                  [_state("AAPL")], held=["AAPL"], now=NOW)
    assert set(st) == {"kill_switch", "enforced_on", "never_blocked", "not_reachable",
                       "cooldown_policy", "symbols", "pre_trade_limits"}
    assert st["kill_switch"]["halt_file_on_shared_mount"] is False


# ── the engine now honors the kill switch ───────────────────────────────

def test_engine_buy_is_refused_when_halted(tmp_path, monkeypatch):
    """Before this change BaseEngine.execute_buy never consulted the kill switch."""
    from falcon_trader.orchestrator.engines import base_engine

    halt = tmp_path / "HALT"
    halt.write_text("halted for test")

    engine_cls = next(
        obj for obj in vars(base_engine).values()
        if isinstance(obj, type) and hasattr(obj, "execute_buy") and obj.__module__ == base_engine.__name__
    )
    engine = engine_cls.__new__(engine_cls)
    engine.kill_switch = KillSwitch(halt_file=halt, env={})

    def _must_not_reach(*a, **k):
        raise AssertionError("cooldown checked after a halt; the halt must short-circuit")

    monkeypatch.setattr(base_engine, "load_symbol_state", _must_not_reach)
    result = engine.execute_buy("AAPL", 10, 100.0, 95.0, 110.0)
    assert result.success is False
    assert "trading_halted" in result.error


def test_engine_constructs_a_kill_switch():
    src = open(__import__("falcon_trader.orchestrator.engines.base_engine",
                          fromlist=["x"]).__file__).read()
    assert "self.kill_switch = KillSwitch()" in src


# ── held_and_recent_symbols (the logic that 500'd in the route) ──────────

class _FakeDB:
    """Records queries; answers the two the helper makes."""

    def __init__(self, held, recent):
        self._held, self._recent = held, recent
        self.calls = []

    def execute(self, query, params=None, fetch=None):
        self.calls.append((query, params, fetch))
        if "FROM positions" in query:
            return [{"symbol": s} for s in self._held]
        if "FROM orders" in query:
            return [{"symbol": s} for s in self._recent]
        raise AssertionError(f"unexpected query: {query}")


def test_symbols_union_of_held_and_recent_sorted():
    """Regression for the production 500.

    This logic first lived inline in the /api/risk/status route and referenced
    `_dt`, which that module never defined at module scope. No test touched the
    route, so the suite was green while the endpoint returned 500.
    """
    db = _FakeDB(held=["MSFT", "AAPL"], recent=["TSLA", "AAPL"])
    held, symbols = risk_status.held_and_recent_symbols(db, now=NOW)
    assert held == ["MSFT", "AAPL"]
    assert symbols == ["AAPL", "MSFT", "TSLA"]


def test_cutoff_is_bound_as_a_parameter_not_backend_sql():
    """`now() - interval '7 days'` is PostgreSQL-only (FAL-11)."""
    db = _FakeDB(held=[], recent=[])
    risk_status.held_and_recent_symbols(db, now=NOW, days=7)
    orders_query, params, _ = db.calls[1]
    assert "interval" not in orders_query.lower() and "now()" not in orders_query.lower()
    assert params == (NOW - dt.timedelta(days=7),)


def test_empty_book_returns_empty_lists():
    db = _FakeDB(held=[], recent=[])
    assert risk_status.held_and_recent_symbols(db, now=NOW) == ([], [])


def test_null_db_results_are_tolerated():
    class _NoneDB(_FakeDB):
        def execute(self, *a, **k):
            super().execute(*a, **k)
            return None
    assert risk_status.held_and_recent_symbols(_NoneDB([], []), now=NOW) == ([], [])
