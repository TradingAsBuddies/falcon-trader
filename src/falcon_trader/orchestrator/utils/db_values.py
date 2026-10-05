"""Coercion for values read back out of the database.

A row does not come back the way it went in, and it differs by backend:
PostgreSQL returns ``numeric`` as Decimal and ``timestamp`` as datetime, while
SQLite returns floats and ISO strings. Code written against one backend breaks
on the other, in the quiet way that matters:

* ``datetime.fromisoformat(result['entry_date'])`` raised "argument must be
  str" for every position, so get_position never returned and no stop or
  target was evaluated for the whole book;
* Decimal entry prices met float market prices in the same subtraction and
  raised TypeError, for the same result.

Both failures were caught by a per-position ``except`` that printed one line
and moved on, so the orchestrator looked healthy while monitoring nothing.

Prices are normalised through falcon_core.prices, so a value that reaches a
strategy engine is a quantized float at the tick its size deserves.
"""

import datetime as _dt
from typing import Any, Optional

from falcon_core import prices


def as_datetime(value: Any) -> Optional[_dt.datetime]:
    """A datetime from whatever the driver returned, or None.

    PostgreSQL gives a datetime, SQLite an ISO string; a date is widened to
    midnight. Anything unparseable is None rather than an exception, because
    the caller is usually monitoring a position and the timestamp is not the
    reason it would exit.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime.combine(value, _dt.time())
    if isinstance(value, str):
        try:
            return _dt.datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def as_price(value: Any, default: float = 0.0) -> float:
    """A quantized float price, whatever the driver returned.

    Engines compute in floats against config values that are floats; feeding
    them a Decimal raises as soon as the two meet. Quantized so the float that
    does arrive carries no digits the price does not have.
    """
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        # An empty cell is an absent price, not zero. falcon_core.prices reads
        # "" as Decimal(0) by design; at this boundary absent is the honest
        # reading, and a zero stop would mean "no stop" to the comparison.
        return default
    try:
        return prices.to_float(value)
    except (TypeError, ValueError, ArithmeticError):
        return default


def as_quantity(value: Any, default: float = 0.0) -> float:
    """A share count as a float; fractional quantities are real."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
