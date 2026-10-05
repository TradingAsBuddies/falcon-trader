"""Values coming back out of the database, coerced at one boundary.

A row does not come back the way it went in, and it differs by backend.
PostgreSQL returns numeric as Decimal and timestamp as datetime; SQLite
returns floats and ISO strings. Two failures from that, both on the same day,
both hidden by a per-position except that printed one line and moved on:

    TypeError: fromisoformat: argument must be str
    TypeError: unsupported operand type(s) for -: 'float' and 'decimal.Decimal'

Either one stopped get_position from returning, so no stop-loss and no profit
target was evaluated for any of 13 positions while the orchestrator reported
itself healthy.
"""

import datetime as dt
from decimal import Decimal
from types import SimpleNamespace

import pytest

from falcon_trader.orchestrator.engines.base_engine import BaseStrategyEngine
from falcon_trader.orchestrator.utils import db_values


# ── as_datetime ─────────────────────────────────────────────────────────

def test_a_postgres_datetime_passes_through():
    """The value that raised: psycopg2 already parsed it."""
    value = dt.datetime(2026, 10, 5, 14, 20, 27)
    assert db_values.as_datetime(value) is value


def test_a_sqlite_iso_string_is_parsed():
    assert db_values.as_datetime("2026-10-05T14:20:27") == dt.datetime(2026, 10, 5, 14, 20, 27)


def test_a_date_widens_to_midnight():
    assert db_values.as_datetime(dt.date(2026, 10, 5)) == dt.datetime(2026, 10, 5, 0, 0)


@pytest.mark.parametrize("value", [None, "", "not a date", 42, object()])
def test_an_unusable_timestamp_is_none_not_an_exception(value):
    """The caller is monitoring a position; the timestamp is not why it exits."""
    assert db_values.as_datetime(value) is None


# ── as_price ────────────────────────────────────────────────────────────

def test_a_postgres_decimal_becomes_a_float():
    value = db_values.as_price(Decimal("69.24"))
    assert type(value) is float and value == 69.24


def test_a_price_is_quantized_to_its_tick():
    assert db_values.as_price(Decimal("1.3449")) == 1.345     # under $52
    assert db_values.as_price(Decimal("69.2449")) == 69.24    # at or above


@pytest.mark.parametrize("value", [None, "", "junk", object()])
def test_an_unusable_price_is_the_default(value):
    assert db_values.as_price(value) == 0.0
    assert db_values.as_price(value, default=1.5) == 1.5


def test_a_quantity_can_be_fractional():
    assert db_values.as_quantity(Decimal("21.0000")) == 21.0
    assert db_values.as_quantity(Decimal("2.5")) == 2.5
    assert db_values.as_quantity(None) == 0.0


# ── get_position, the method that never returned ────────────────────────

class _Engine(BaseStrategyEngine):
    def __init__(self, row):
        self.config = {}
        self.strategy_name = "rsi_mean_reversion"
        self.db = SimpleNamespace(execute=lambda *a, **k: row)

    def generate_signal(self, symbol, market_data):  # pragma: no cover
        raise NotImplementedError

    def backtest_signal(self, symbol, historical_data):  # pragma: no cover
        raise NotImplementedError


POSTGRES_ROW = {
    "symbol": "QSR", "quantity": Decimal("21.0000"),
    "entry_price": Decimal("69.24"), "stop_loss": Decimal("65.77"),
    "profit_target": Decimal("70.98"), "strategy": "rsi",
    "entry_date": dt.datetime(2026, 10, 5, 14, 20, 27),
}

SQLITE_ROW = {
    "symbol": "QSR", "quantity": 21, "entry_price": 69.24, "stop_loss": 65.77,
    "profit_target": 70.98, "strategy": "rsi",
    "entry_date": "2026-10-05T14:20:27",
}


@pytest.mark.parametrize("row", [POSTGRES_ROW, SQLITE_ROW], ids=["postgres", "sqlite"])
def test_get_position_works_on_either_backend(row):
    position = _Engine(row).get_position("QSR")
    assert position is not None
    assert position.symbol == "QSR"
    assert position.entry_timestamp == dt.datetime(2026, 10, 5, 14, 20, 27)


@pytest.mark.parametrize("row", [POSTGRES_ROW, SQLITE_ROW], ids=["postgres", "sqlite"])
def test_a_position_carries_no_decimals_into_the_engines(row):
    """Engines compute against float config values; a Decimal raises on contact."""
    position = _Engine(row).get_position("QSR")
    for field in ("quantity", "entry_price", "current_price", "stop_loss", "profit_target"):
        assert type(getattr(position, field)) is float, field


def test_position_prices_are_quantized():
    row = dict(POSTGRES_ROW, entry_price=Decimal("69.2449"), stop_loss=Decimal("1.3449"))
    position = _Engine(row).get_position("QSR")
    assert position.entry_price == 69.24
    assert position.stop_loss == 1.345


def test_a_missing_entry_date_does_not_stop_the_position_loading():
    """Better a position with an approximate age than no monitoring at all."""
    position = _Engine(dict(POSTGRES_ROW, entry_date=None)).get_position("QSR")
    assert position is not None
    assert position.entry_timestamp is not None


def test_no_row_still_means_no_position():
    assert _Engine(None).get_position("QSR") is None


def test_the_position_then_survives_a_price_update():
    """update_current_price does cents arithmetic; a Decimal entry broke it."""
    position = _Engine(POSTGRES_ROW).get_position("QSR")
    position.update_current_price(70.00)
    assert position.unrealized_pnl == pytest.approx(15.96, abs=0.01)
