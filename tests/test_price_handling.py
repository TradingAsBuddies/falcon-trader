"""Price arithmetic and monitor reporting in the trading path.

Two live failures on 2026-10-05, after the orchestrator was finally wired to
the real database:

* every position raised `unsupported operand type(s) for -: 'float' and
  'decimal.Decimal'`, because entry_price comes from PostgreSQL as Decimal and
  the fetched price is a float — so no stop and no target was evaluated for a
  13-position book;
* the monitor returned rows only for SELLs, so a book that was monitored and
  held came back empty and the caller printed "No open positions to monitor"
  while holding thirteen.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from falcon_core import prices
from falcon_trader.orchestrator.engines.base_engine import BaseStrategyEngine


class _Engine(BaseStrategyEngine):
    def __init__(self, cash=10_000, risk=75):
        self.config = {"risk": {"per_trade_dollars": risk}}
        self.strategy_name = "test"
        self._cash = cash

    def get_account_balance(self):
        return self._cash

    def generate_signal(self, symbol, market_data):  # pragma: no cover
        raise NotImplementedError

    def backtest_signal(self, symbol, historical_data):  # pragma: no cover
        raise NotImplementedError


def _position(stop_loss=None, profit_target=None, entry_price=Decimal("17.35")):
    return SimpleNamespace(symbol="AMRX", quantity=164, entry_price=entry_price,
                           stop_loss=stop_loss, profit_target=profit_target)


# ── stop and target comparisons ─────────────────────────────────────────

def test_a_decimal_stop_compares_against_a_float_price():
    """The mixed-type pair that raised in production."""
    engine = _Engine()
    assert engine.check_stop_loss(_position(stop_loss=Decimal("16.48")), 16.47) is True
    assert engine.check_stop_loss(_position(stop_loss=Decimal("16.48")), 16.49) is False


def test_a_price_exactly_at_the_stop_triggers():
    assert _Engine().check_stop_loss(_position(stop_loss=Decimal("16.48")), 16.48) is True


@pytest.mark.parametrize("noisy,clean", [
    (16.480000000000001, 16.48),
    (16.479999999999997, 16.48),
    (1.3450000000000002, 1.345),
])
def test_float_noise_does_not_change_the_decision(noisy, clean):
    """Noise in the seventeenth digit must not decide an exit either way.

    Both quantize to the same integer, so both give the same answer as the
    clean price — here that answer is "at the stop, so exit".
    """
    engine = _Engine()
    position = _position(stop_loss=Decimal(str(clean)))
    assert engine.check_stop_loss(position, noisy) == engine.check_stop_loss(position, clean)


def test_no_stop_set_never_triggers():
    for value in (None, 0, Decimal("0")):
        assert _Engine().check_stop_loss(_position(stop_loss=value), 1.0) is False


def test_a_decimal_target_compares_against_a_float_price():
    engine = _Engine()
    assert engine.check_profit_target(_position(profit_target=Decimal("20.00")), 20.01) is True
    assert engine.check_profit_target(_position(profit_target=Decimal("20.00")), 19.99) is False


def test_no_target_set_never_triggers():
    for value in (None, 0, Decimal("0")):
        assert _Engine().check_profit_target(_position(profit_target=value), 1e9) is False


def test_a_sub_dollar_stop_is_compared_in_thousandths():
    """CDXS at $1.345: a cents-only comparison would be half a percent out."""
    engine = _Engine()
    assert engine.check_stop_loss(_position(stop_loss=Decimal("1.345")), 1.344) is True
    assert engine.check_stop_loss(_position(stop_loss=Decimal("1.345")), 1.346) is False


# ── sizing on quantized prices ──────────────────────────────────────────

def test_sizing_quantizes_the_stop_it_measures_from():
    """A stop of 32.17649999 implies a risk per share that is not a real number."""
    engine = _Engine(risk=75)
    noisy = engine.calculate_position_size("NKE", 33.87, 1.0, stop_loss=32.176499999)
    clean = engine.calculate_position_size("NKE", 33.87, 1.0, stop_loss=32.176)
    assert noisy == clean


def test_risk_is_still_one_r():
    engine = _Engine(risk=75)
    quantity = engine.calculate_position_size("QSR", 69.91, 1.0, stop_loss=66.41)
    assert quantity * (69.91 - 66.41) <= 75


# ── what the monitor reports ────────────────────────────────────────────

class _FakeExecutor:
    """monitor_positions' reporting contract, without the engine machinery."""

    def __init__(self, actions):
        self._actions = actions

    def monitor_positions(self):
        return self._actions


def test_held_positions_are_reported_not_silently_dropped(capsys):
    """The regression: only SELLs were collected, so a held book read as empty."""
    from falcon_trader import run_orchestrator

    executor = _FakeExecutor([
        {"symbol": "AMRX", "action": "HOLD", "current_price": 20.48, "pnl_pct": 18.04},
        {"symbol": "BMBL", "action": "HOLD", "current_price": 2.54, "pnl_pct": -16.17},
    ])
    run_orchestrator.monitor_positions(executor, tracker=None)
    out = capsys.readouterr().out
    assert "Monitored 2 positions" in out
    # $20.480: under $52, so the price shows the thousandth it is computed in.
    assert "[HOLD] AMRX: $20.480 (+18.04%)" in out
    assert "No open positions to monitor" not in out


def test_a_position_that_could_not_be_evaluated_is_named(capsys):
    """An unmonitored position has no stop in force; that is not a quiet hold."""
    from falcon_trader import run_orchestrator

    executor = _FakeExecutor([
        {"symbol": "AMRX", "action": "ERROR", "reason": "boom"},
        {"symbol": "BMBL", "action": "HOLD", "current_price": 2.54, "pnl_pct": -16.2},
    ])
    run_orchestrator.monitor_positions(executor, tracker=None)
    out = capsys.readouterr().out
    assert "1 could not be evaluated" in out
    assert "[UNMONITORED] AMRX: boom" in out


def test_an_empty_book_still_says_so(capsys):
    from falcon_trader import run_orchestrator
    run_orchestrator.monitor_positions(_FakeExecutor([]), tracker=None)
    assert "No open positions to monitor" in capsys.readouterr().out


def test_a_sub_dollar_hold_prints_three_decimals(capsys):
    from falcon_trader import run_orchestrator
    run_orchestrator.monitor_positions(
        _FakeExecutor([{"symbol": "CDXS", "action": "HOLD",
                        "current_price": 1.3449, "pnl_pct": -7.2}]), tracker=None)
    assert "$1.345" in capsys.readouterr().out


# ── what a fill prints ──────────────────────────────────────────────────

def test_a_fill_prints_its_real_quantity_and_price(capsys):
    """Read from the top level these were absent: every fill printed BUY 0 @ $0.00."""
    from falcon_trader import run_orchestrator

    summary = {
        "total_stocks": 1, "processed": 1, "trades_executed": 1, "skipped": 0,
        "errors": 0,
        "details": [{
            "symbol": "QSR", "success": True, "action": "BUY",
            "details": {"execution": {"quantity": 21, "price": 69.24}},
        }],
    }

    class _Executor:
        def process_recommendations(self, recs):
            return summary

    class _DB:
        def execute(self, *a, **k):
            return [{"run_timestamp": None, "run_data": {"recommendations": [{"ticker": "QSR"}]}}]

    run_orchestrator.process_screener_results(_Executor(), None, _DB())
    assert "[OK] QSR: BUY 21 @ $69.24" in capsys.readouterr().out


def test_prices_module_is_the_one_in_falcon_core():
    """The convention lives in one place for all three repos."""
    assert prices.MILLS_PER_DOLLAR == 1000
    assert prices.FINE_TICK_BELOW == Decimal("52")
