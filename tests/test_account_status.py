"""Account status printing, on both backends' row shapes.

It did `cash, total_value = account`, which unpacks the *key names* from a
PostgreSQL mapping row, and selected a total_value column that `account` does
not have:

    [ERROR] Cycle 170 failed: UndefinedColumn: column "total_value" does not exist

It runs every tenth cycle, so account status had not printed once since the
orchestrator moved to PostgreSQL. The cycle guard caught it each time, which
is why it cost the report and nothing else.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from falcon_trader import run_orchestrator as orch


class _DB:
    """Returns mapping rows, as RealDictCursor does. Records the SQL."""

    def __init__(self, cash=Decimal("24755.67"), positions=None):
        self.cash = cash
        self.positions = positions if positions is not None else []
        self.queries = []

    def execute(self, query, params=None, fetch=None):
        self.queries.append(" ".join(query.split()))
        if "FROM account" in query:
            return {"cash": self.cash}
        if "FROM positions" in query:
            return list(self.positions)
        raise AssertionError(f"unexpected query: {query}")


def _position(symbol, quantity, entry, current):
    return {"symbol": symbol, "quantity": Decimal(str(quantity)),
            "entry_price": Decimal(str(entry)),
            "current_price": Decimal(str(current))}


def _status(db):
    orch.show_account_status(SimpleNamespace(db=db))


def test_it_no_longer_selects_a_column_that_does_not_exist(capsys):
    """The regression: account has id, cash, last_updated, initial_balance."""
    db = _DB()
    _status(db)
    account_query = next(q for q in db.queries if "FROM account" in q)
    assert "total_value" not in account_query


def test_cash_prints_its_value_not_the_column_name(capsys):
    """Unpacking a mapping row gave the string 'cash'."""
    _status(_DB(cash=Decimal("24755.67")))
    out = capsys.readouterr().out
    assert "Cash: $24,755.67" in out


def test_total_value_is_cash_plus_marked_positions(capsys):
    db = _DB(cash=Decimal("1000.00"), positions=[
        # 70.515 quantizes to 70.52 (at/above $52, so cents) *before* the
        # multiply -- price to its tick first, then multiply, as a till does.
        _position("QSR", 21, "69.24", "70.515"),     # 21 * 70.52 = 1480.92
        _position("SHC", 79, "18.855", "18.560"),    # 79 * 18.56  = 1466.24
    ])
    _status(db)
    out = capsys.readouterr().out
    assert "Positions: $2,947.16" in out
    assert "Total Value: $3,947.16" in out


def test_each_position_prints_with_its_own_tick(capsys):
    db = _DB(positions=[
        _position("QSR", 21, "69.24", "70.515"),   # at/above $52 -> cents
        _position("CDXS", 1588, "1.45", "1.345"),  # below $52 -> thousandths
    ])
    _status(db)
    out = capsys.readouterr().out
    assert "QSR: 21 @ $69.24 -> $70.52" in out
    assert "CDXS: 1588 @ $1.450 -> $1.345" in out


def test_percentages_are_computed_not_taken_from_sql(capsys):
    """The old query divided in SQL, which returned Decimal and mixed types."""
    _status(_DB(positions=[_position("SHC", 79, "18.855", "19.326")]))
    assert "+2.50%" in capsys.readouterr().out


def test_an_unmarked_position_falls_back_to_its_entry(capsys):
    """current_price is NULL until the first monitoring pass writes one."""
    db = _DB(positions=[{"symbol": "NEW", "quantity": Decimal("10"),
                         "entry_price": Decimal("5.00"), "current_price": None}])
    _status(db)
    out = capsys.readouterr().out
    assert "NEW: 10 @ $5.000 -> $5.000 (+0.00%)" in out
    assert "Positions: $50.00" in out


def test_an_empty_book_prints_zero(capsys):
    _status(_DB(positions=[]))
    out = capsys.readouterr().out
    assert "Open Positions: 0" in out
    assert "Total Value: $24,755.67" in out


def test_no_account_row_does_not_raise(capsys):
    class _NoAccount(_DB):
        def execute(self, query, params=None, fetch=None):
            self.queries.append(query)
            return None if "FROM account" in query else []

    _status(_NoAccount())
    assert "Cash: $0.00" in capsys.readouterr().out
