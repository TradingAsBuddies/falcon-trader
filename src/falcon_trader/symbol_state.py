"""Per-symbol trading history used by the churn guards (falcon-trader#24).

The cooldown guards need three facts per symbol: when it was last exited, how
many consecutive losing round-trips it has had, and how many round-trips it has
already had today.

These are **derived from the existing ``orders`` table** rather than stored in
new columns. The alternative -- ``last_exit_at`` / ``loss_streak`` columns on
``positions`` -- needs a migration, and worse, it is a second copy of something
the order history already knows. A denormalized counter that drifts from the
fills it summarizes is exactly the class of bug that produced the cash-accounting
issue (#22). Deriving costs one indexed query per entry decision, on a loop that
runs every five minutes.

``build_symbol_state`` is pure -- it takes order rows and returns the facts --
so the streak logic is testable without a database.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

from falcon_trader.trading_guards import EASTERN

__all__ = ["SymbolState", "build_symbol_state", "load_symbol_state", "SYMBOL_STATE_SQL"]


@dataclass(frozen=True)
class SymbolState:
    """What the churn guards need to know about one symbol."""

    symbol: str
    last_exit_at: Optional[_dt.datetime] = None
    last_exit_pnl: Optional[float] = None
    loss_streak: int = 0
    round_trips_today: int = 0
    total_round_trips: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "lastExitAt": self.last_exit_at.isoformat() if self.last_exit_at else None,
            "lastExitPnl": self.last_exit_pnl,
            "lossStreak": self.loss_streak,
            "roundTripsToday": self.round_trips_today,
            "totalRoundTrips": self.total_round_trips,
        }


#: Orders for one symbol, oldest first. `pnl` is non-zero only on exits.
SYMBOL_STATE_SQL = """
    SELECT side, quantity, price, timestamp, pnl
      FROM orders
     WHERE symbol = %s
     ORDER BY timestamp ASC
"""


def _parse_ts(value) -> Optional[_dt.datetime]:
    """Coerce a stored timestamp to an Eastern-aware datetime."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = _dt.datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=EASTERN)
    return moment.astimezone(EASTERN)


def build_symbol_state(
    symbol: str,
    orders: Iterable[Mapping],
    now: Optional[_dt.datetime] = None,
) -> SymbolState:
    """Derive the churn facts for `symbol` from its order history.

    A "round-trip" is a sell that realized P&L. The streak counts *consecutive*
    losing exits working backwards from the most recent one, so a single winner
    clears it -- which is the behavior you want: the lockout is for a symbol
    that keeps costing money, not one that once did.
    """
    now = now or _dt.datetime.now(EASTERN)
    if now.tzinfo is None:
        now = now.replace(tzinfo=EASTERN)
    today = now.date()

    exits: List[tuple] = []
    for row in orders:
        if str(row.get("side", "")).lower() != "sell":
            continue
        moment = _parse_ts(row.get("timestamp"))
        if moment is None:
            continue
        try:
            pnl = float(row.get("pnl") or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        exits.append((moment, pnl))

    if not exits:
        return SymbolState(symbol=symbol)

    exits.sort(key=lambda pair: pair[0])

    last_moment, last_pnl = exits[-1]

    streak = 0
    for _, pnl in reversed(exits):
        if pnl < 0:
            streak += 1
        else:
            break

    round_trips_today = sum(1 for moment, _ in exits if moment.date() == today)

    return SymbolState(
        symbol=symbol,
        last_exit_at=last_moment,
        last_exit_pnl=last_pnl,
        loss_streak=streak,
        round_trips_today=round_trips_today,
        total_round_trips=len(exits),
    )


def load_symbol_state(db, symbol: str, now=None) -> SymbolState:
    """Read `symbol`'s order history and derive its state.

    Never raises: a state lookup that fails must not take down a trading loop.
    A default state is permissive, so the caller's other guards still apply.
    """
    try:
        rows = db.execute(SYMBOL_STATE_SQL, (symbol,), fetch="all") or []
    except Exception:
        return SymbolState(symbol=symbol)
    return build_symbol_state(symbol, rows, now=now)
