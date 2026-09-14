"""Tests for falcon_trader.risk_limits and symbol_state (falcon-trader#24, #26)."""

import datetime as dt

import pytest

from falcon_trader.risk_limits import (
    KillSwitch,
    RiskLimits,
    check_pre_trade,
    reconcile,
)
from falcon_trader.symbol_state import build_symbol_state

LIMITS = RiskLimits()


def order(**kw):
    base = dict(
        symbol="AAPL", side="buy", quantity=10, price=100.0, limits=LIMITS,
        total_value=10000.0, cash=10000.0, open_positions=(), positions_value=0.0,
    )
    base.update(kw)
    return check_pre_trade(**base)


# --------------------------------------------------------------------------
# kill switch
# --------------------------------------------------------------------------

def test_kill_switch_enabled_by_default(tmp_path):
    ks = KillSwitch(halt_file=tmp_path / "halt", env={})
    assert ks.is_trading_enabled() is True
    assert ks.reason() is None


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE"])
def test_env_flag_halts(tmp_path, raw):
    ks = KillSwitch(halt_file=tmp_path / "halt",
                    env={"FALCON_TRADING_ENABLED": raw})
    assert ks.is_trading_enabled() is False


def test_halt_file_halts(tmp_path):
    halt = tmp_path / "halt"
    ks = KillSwitch(halt_file=halt, env={})
    assert ks.is_trading_enabled() is True
    halt.write_text("stopped")
    assert ks.is_trading_enabled() is False
    assert "Halt file present" in ks.reason()


def test_in_process_halt_writes_the_file(tmp_path):
    halt = tmp_path / "halt"
    ks = KillSwitch(halt_file=halt, env={})
    ks.halt("daily loss limit breached")
    assert ks.is_trading_enabled() is False
    assert halt.exists()
    assert "daily loss limit breached" in halt.read_text()


def test_resume_clears_the_halt(tmp_path):
    halt = tmp_path / "halt"
    ks = KillSwitch(halt_file=halt, env={})
    ks.halt("test")
    ks.resume()
    assert ks.is_trading_enabled() is True
    assert not halt.exists()


def test_resume_cannot_override_the_env_flag(tmp_path):
    """A deploy-time halt must not be clearable from inside the process."""
    ks = KillSwitch(halt_file=tmp_path / "halt",
                    env={"FALCON_TRADING_ENABLED": "0"})
    ks.resume()
    assert ks.is_trading_enabled() is False


def test_unreadable_halt_file_fails_closed(tmp_path):
    """A kill switch that fails open is not a kill switch."""
    class Exploding:
        def exists(self):
            raise OSError("no")
    ks = KillSwitch(halt_file=tmp_path / "halt", env={})
    ks.halt_file = Exploding()
    assert ks.is_trading_enabled() is False


def test_halted_orders_are_blocked(tmp_path):
    ks = KillSwitch(halt_file=tmp_path / "halt", env={"FALCON_TRADING_ENABLED": "0"})
    result = order(kill_switch=ks)
    assert result.allowed is False
    assert result.reason == "trading_halted"


def test_kill_switch_blocks_sells_too(tmp_path):
    """A full halt means no routing at all, in either direction."""
    ks = KillSwitch(halt_file=tmp_path / "halt", env={"FALCON_TRADING_ENABLED": "0"})
    assert order(side="sell", kill_switch=ks).allowed is False


# --------------------------------------------------------------------------
# pre-trade limits
# --------------------------------------------------------------------------

def test_a_reasonable_order_passes():
    assert order().allowed is True


def test_penny_stock_is_refused():
    """SNYR at $0.13 and AEON at $0.30 were live recommendations."""
    result = order(symbol="SNYR", price=0.13, quantity=1000)
    assert result.allowed is False
    assert result.reason == "below_min_price"


def test_thin_liquidity_is_refused():
    result = order(avg_dollar_volume=100_000.0)
    assert result.allowed is False
    assert result.reason == "insufficient_liquidity"


def test_position_concentration_limit():
    """25% of a $10k account is $2500; $4000 must be refused."""
    result = order(quantity=40, price=100.0)
    assert result.allowed is False
    assert result.reason == "max_position_size"


def test_concentration_counts_the_existing_holding():
    """Scaling in must not sneak past the cap in two steps."""
    result = order(
        quantity=15, price=100.0,
        open_positions=[{"symbol": "AAPL", "quantity": 15, "avgPrice": 100.0}],
    )
    assert result.allowed is False
    assert result.reason == "max_position_size"


def test_gross_exposure_limit():
    """Live account: $20.6k of positions on a $10k account, no limit anywhere."""
    result = order(
        quantity=5, price=100.0,
        total_value=10000.0, cash=10000.0, positions_value=9900.0,
    )
    assert result.allowed is False
    assert result.reason == "max_gross_exposure"


def test_max_open_positions():
    positions = [{"symbol": f"S{i}", "quantity": 1, "avgPrice": 10.0}
                 for i in range(10)]
    result = order(open_positions=positions, positions_value=100.0)
    assert result.allowed is False
    assert result.reason == "max_open_positions"


def test_adding_to_an_existing_position_ignores_the_count_limit():
    positions = [{"symbol": f"S{i}", "quantity": 1, "avgPrice": 10.0}
                 for i in range(10)]
    positions[0]["symbol"] = "AAPL"
    result = order(quantity=1, price=100.0, open_positions=positions,
                   positions_value=100.0)
    assert result.allowed is True


def test_daily_loss_halt():
    result = order(daily_pnl=-350.0, total_value=10000.0)
    assert result.allowed is False
    assert result.reason == "max_daily_loss"


def test_daily_gain_does_not_halt():
    assert order(daily_pnl=+350.0).allowed is True


def test_insufficient_cash():
    result = order(quantity=10, price=100.0, cash=500.0)
    assert result.allowed is False
    assert result.reason == "insufficient_cash"


def test_blacklist():
    limits = RiskLimits(blacklist=frozenset({"CDXS"}))
    result = order(symbol="CDXS", limits=limits)
    assert result.reason == "blacklisted"


def test_halted_symbol_is_refused():
    result = order(is_halted_symbol=True)
    assert result.reason == "symbol_halted"


def test_pdt_limit():
    result = order(day_trades_used=3)
    assert result.reason == "pdt_limit"


def test_pdt_check_can_be_disabled():
    limits = RiskLimits(max_day_trades=None)
    assert order(limits=limits, day_trades_used=99).allowed is True


def test_sells_bypass_exposure_limits():
    """Refusing to reduce risk because exposure is high would be backwards."""
    result = order(side="sell", positions_value=99999.0, daily_pnl=-9999.0,
                   price=1.0)
    assert result.allowed is True


def test_sell_of_a_halted_symbol_is_still_refused():
    assert order(side="sell", is_halted_symbol=True).allowed is False


def test_limits_from_config():
    limits = RiskLimits.from_config({
        "max_position_pct": 0.1,
        "max_gross_exposure_pct": 0.5,
        "max_daily_loss_pct": 0.02,
        "max_open_positions": 5,
        "min_price": 10.0,
        "min_dollar_volume": 1e7,
        "blacklist": ["snyr"],
    })
    assert limits.max_position_pct == 0.1
    assert limits.max_open_positions == 5
    assert limits.blacklist == frozenset({"SNYR"})


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------

def test_matching_books_reconcile():
    report = reconcile(
        [{"symbol": "AAPL", "quantity": 10}], 5000.0,
        [{"symbol": "AAPL", "quantity": 10}], 5000.0,
    )
    assert report.matched is True


def test_quantity_drift_is_detected():
    report = reconcile(
        [{"symbol": "AAPL", "quantity": 10}], 5000.0,
        [{"symbol": "AAPL", "quantity": 8}], 5000.0,
    )
    assert report.matched is False
    assert report.position_differences[0]["difference"] == 2


def test_cash_drift_is_detected():
    report = reconcile([], 5000.0, [], 4900.0)
    assert report.matched is False
    assert report.cash_difference == pytest.approx(100.0)


def test_position_only_at_broker_is_flagged():
    report = reconcile([], 0.0, [{"symbol": "TSLA", "quantity": 5}], 0.0)
    assert report.missing_locally == ["TSLA"]
    assert report.matched is False


def test_position_only_local_is_flagged():
    report = reconcile([{"symbol": "TSLA", "quantity": 5}], 0.0, [], 0.0)
    assert report.missing_at_broker == ["TSLA"]


def test_reconciliation_summary_is_human_readable():
    report = reconcile([{"symbol": "A", "quantity": 1}], 100.0, [], 50.0)
    assert "cash off by" in report.summary()


# --------------------------------------------------------------------------
# symbol state (#24)
# --------------------------------------------------------------------------

def test_no_history_yields_an_empty_state():
    state = build_symbol_state("NEW", [])
    assert state.last_exit_at is None
    assert state.loss_streak == 0
    assert state.round_trips_today == 0


def test_loss_streak_counts_consecutive_losses():
    orders = [
        {"side": "sell", "timestamp": "2026-09-01T10:00:00", "pnl": -10.0},
        {"side": "sell", "timestamp": "2026-09-02T10:00:00", "pnl": -20.0},
    ]
    assert build_symbol_state("CDXS", orders).loss_streak == 2


def test_a_win_clears_the_streak():
    orders = [
        {"side": "sell", "timestamp": "2026-09-01T10:00:00", "pnl": -10.0},
        {"side": "sell", "timestamp": "2026-09-02T10:00:00", "pnl": -20.0},
        {"side": "sell", "timestamp": "2026-09-03T10:00:00", "pnl": +30.0},
    ]
    assert build_symbol_state("CDXS", orders).loss_streak == 0


def test_buys_do_not_count_as_round_trips():
    orders = [
        {"side": "buy",  "timestamp": "2026-09-01T10:00:00", "pnl": 0.0},
        {"side": "buy",  "timestamp": "2026-09-02T10:00:00", "pnl": 0.0},
        {"side": "sell", "timestamp": "2026-09-03T10:00:00", "pnl": -5.0},
    ]
    state = build_symbol_state("CDXS", orders)
    assert state.total_round_trips == 1


def test_round_trips_today_counts_only_today():
    now = dt.datetime(2026, 9, 9, 15, 0)
    orders = [
        {"side": "sell", "timestamp": "2026-09-08T16:50:00", "pnl": -277.92},
        {"side": "sell", "timestamp": "2026-09-09T10:37:00", "pnl": -90.72},
    ]
    state = build_symbol_state("CDXS", orders, now=now)
    assert state.round_trips_today == 1
    assert state.total_round_trips == 2


def test_last_exit_is_the_most_recent():
    orders = [
        {"side": "sell", "timestamp": "2026-09-09T10:37:00", "pnl": -90.72},
        {"side": "sell", "timestamp": "2026-09-08T16:50:00", "pnl": -277.92},
    ]
    state = build_symbol_state("CDXS", orders)
    assert state.last_exit_at.date() == dt.date(2026, 9, 9)
    assert state.last_exit_pnl == pytest.approx(-90.72)


def test_unparseable_timestamps_are_skipped_not_fatal():
    orders = [
        {"side": "sell", "timestamp": "not a date", "pnl": -1.0},
        {"side": "sell", "timestamp": "2026-09-09T10:00:00", "pnl": -2.0},
    ]
    state = build_symbol_state("X", orders)
    assert state.total_round_trips == 1


def test_null_pnl_is_treated_as_flat():
    orders = [{"side": "sell", "timestamp": "2026-09-09T10:00:00", "pnl": None}]
    state = build_symbol_state("X", orders)
    assert state.loss_streak == 0
