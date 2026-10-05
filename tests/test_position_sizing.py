"""Risk-based position sizing, and the end-of-day flatten switch.

Sizing was 20-25% of cash per trade regardless of where the stop sat, so a
tight stop and a wide one risked wildly different amounts on the same account.
The operator trades in R units: one R is a fixed dollar risk, and quantity
follows from the stop distance.
"""

import datetime as dt

import pytest

from falcon_trader import run_orchestrator
from falcon_trader.orchestrator.engines.base_engine import BaseStrategyEngine


class _Engine(BaseStrategyEngine):
    """BaseStrategyEngine without its database or kill switch."""

    def __init__(self, config, cash):
        self.config = config
        self.strategy_name = "test"
        self._cash = cash

    def get_account_balance(self):
        return self._cash

    def generate_signal(self, symbol, market_data):  # pragma: no cover - abstract
        raise NotImplementedError

    def backtest_signal(self, symbol, historical_data):  # pragma: no cover
        raise NotImplementedError


def _engine(cash=7835.16, risk=75):
    config = {"risk": {"per_trade_dollars": risk}} if risk is not None else {}
    return _Engine(config, cash)


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("FALCON_RISK_PER_TRADE", raising=False)
    monkeypatch.delenv("FALCON_EOD_FLATTEN", raising=False)


# ── risk-based sizing ───────────────────────────────────────────────────

def test_quantity_follows_the_stop_distance():
    """$75 risked, $1.50 of risk per share -> 50 shares."""
    assert _engine().calculate_position_size("X", 26.20, 0.25, stop_loss=24.70) == 50


def test_a_tighter_stop_buys_more_shares_for_the_same_risk():
    engine = _engine()
    wide = engine.calculate_position_size("X", 100.0, 1.0, stop_loss=95.0)   # $5/share
    tight = engine.calculate_position_size("X", 100.0, 1.0, stop_loss=99.0)  # $1/share
    assert wide == 15 and tight == 75
    for qty, stop in ((wide, 95.0), (tight, 99.0)):
        assert qty * (100.0 - stop) <= 75


def test_risk_is_the_same_whatever_the_price():
    engine = _engine()
    cheap = engine.calculate_position_size("X", 3.00, 1.0, stop_loss=2.70)
    rich = engine.calculate_position_size("X", 300.0, 1.0, stop_loss=270.0)
    assert cheap * 0.30 <= 75 and rich * 30.0 <= 75


def test_percentage_ceiling_caps_a_very_tight_stop():
    """A stop 0.1% away would otherwise buy the whole account in one name."""
    engine = _engine(cash=10_000)
    qty = engine.calculate_position_size("X", 100.0, 0.20, stop_loss=99.9)
    assert qty == 20  # 20% of 10,000 / 100, not 75 / 0.10 = 750


def test_nothing_exceeds_available_cash():
    engine = _engine(cash=500)
    qty = engine.calculate_position_size("X", 100.0, 1.0, stop_loss=99.0)
    assert qty * 100.0 <= 500


# ── falling back ────────────────────────────────────────────────────────

def test_without_a_stop_it_sizes_by_percentage():
    assert _engine(cash=10_000).calculate_position_size("X", 100.0, 0.25) == 25


@pytest.mark.parametrize("stop", [0, -5, 100.0, 150.0])
def test_an_unusable_stop_falls_back_to_percentage(stop):
    """A stop at or above entry has no risk distance to size from."""
    assert _engine(cash=10_000).calculate_position_size("X", 100.0, 0.25, stop_loss=stop) == 25


def test_no_configured_risk_means_percentage_sizing():
    assert _engine(cash=10_000, risk=None).calculate_position_size(
        "X", 100.0, 0.25, stop_loss=95.0) == 25


@pytest.mark.parametrize("price", [0, -1])
def test_a_non_positive_price_buys_nothing(price):
    assert _engine().calculate_position_size("X", price, 0.25, stop_loss=1.0) == 0


def test_risk_too_small_for_one_share_buys_nothing():
    engine = _engine(cash=10_000, risk=1)
    assert engine.calculate_position_size("X", 100.0, 0.25, stop_loss=90.0) == 0


# ── the risk budget ─────────────────────────────────────────────────────

def test_env_overrides_the_configured_risk(monkeypatch):
    monkeypatch.setenv("FALCON_RISK_PER_TRADE", "150")
    # Cash well above 150 shares * $100 so the cash cap is not what is measured.
    assert _engine(cash=100_000).calculate_position_size(
        "X", 100.0, 1.0, stop_loss=99.0) == 150


@pytest.mark.parametrize("raw", ["", "abc", "0", "-5", None])
def test_unusable_risk_values_disable_risk_sizing(raw, monkeypatch):
    if raw is not None:
        monkeypatch.setenv("FALCON_RISK_PER_TRADE", raw)
    engine = _engine(cash=10_000, risk=raw)
    assert engine.risk_per_trade() is None


def test_configured_risk_is_the_operators_sim_1r():
    """The shipped config matches the operator's SIM 1R of $75."""
    import yaml
    from pathlib import Path
    import falcon_trader
    path = Path(falcon_trader.__file__).parent / "orchestrator" / "orchestrator_config.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["risk"]["per_trade_dollars"] == 75


# ── end-of-day flatten ──────────────────────────────────────────────────

def test_flatten_is_off_in_the_shipped_config():
    """Swing behaviour: these strategies hold for max_hold_days (12-20)."""
    import yaml
    from pathlib import Path
    import falcon_trader
    path = Path(falcon_trader.__file__).parent / "orchestrator" / "orchestrator_config.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["session"]["eod_flatten"] is False
    assert run_orchestrator.eod_flatten_enabled(config) is False


@pytest.mark.parametrize("config", [None, {}, {"session": {}}])
def test_flatten_defaults_off_when_unconfigured(config):
    assert run_orchestrator.eod_flatten_enabled(config) is False


def test_flatten_can_be_turned_on_in_config():
    assert run_orchestrator.eod_flatten_enabled({"session": {"eod_flatten": True}}) is True


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False),
])
def test_env_controls_flatten(raw, expected, monkeypatch):
    monkeypatch.setenv("FALCON_EOD_FLATTEN", raw)
    assert run_orchestrator.eod_flatten_enabled({"session": {"eod_flatten": False}}) is expected


def test_flatten_window_itself_is_unchanged():
    """The switch gates the flatten; it does not move the window."""
    from falcon_trader import trading_guards
    assert trading_guards.DEFAULT_FLATTEN_AT == dt.time(15, 55)
