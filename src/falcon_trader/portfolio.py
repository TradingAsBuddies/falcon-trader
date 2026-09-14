"""Position and cash arithmetic (falcon-trader#21, #22).

Pulled out of ``paper_trading_bot`` and ``dashboard_server`` so the money math
can be tested without a database, a Flask app, or a market data key. The live
account had drifted to::

    cash 7358.48 + positions 20648.00 = 28006.48   but totalValue 28368.22

on a $10k paper account with +$478 of realized P&L. Three separate defects fed
that: two different price sources for the same quantity, sells of shares the
account did not hold crediting cash anyway, and unsynchronized read-modify-write
updates to a single-row ``account`` table.

The functions here are deliberately dumb and total: they take numbers and return
numbers or a verdict, and they never touch I/O. The callers keep the SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

__all__ = [
    "SellCheck",
    "PositionMark",
    "AccountValuation",
    "InvariantReport",
    "validate_sell",
    "weighted_average_entry",
    "reduce_position",
    "mark_position",
    "value_account",
    "check_invariant",
]

#: Tolerance for float comparison of dollar amounts. A cent is the smallest
#: unit anyone cares about; anything under a tenth of a cent is float noise.
CENT = 0.001


@dataclass(frozen=True)
class SellCheck:
    """Verdict on a proposed sell."""

    ok: bool
    quantity: int = 0
    reason: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class PositionMark:
    """A position valued at a current price, or honestly marked stale."""

    symbol: str
    quantity: int
    avg_price: float
    current_price: Optional[float]
    stale: bool
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avgPrice": self.avg_price,
            # None, not avgPrice. A caller that cannot tell a real mark from a
            # fallback will compute an unrealized P&L of exactly $0 forever and
            # never fire a stop-loss (falcon-trader#21).
            "currentPrice": self.current_price,
            "stale": self.stale,
            "marketValue": self.market_value,
            "unrealizedPnl": self.unrealized_pnl,
            "unrealizedPnlPct": self.unrealized_pnl_pct,
        }


@dataclass(frozen=True)
class AccountValuation:
    """One consistent valuation of the whole account."""

    cash: float
    initial_balance: float
    positions_value: float
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    stale_symbols: tuple = ()

    @property
    def has_stale_marks(self) -> bool:
        return bool(self.stale_symbols)

    def to_dict(self) -> Dict:
        return {
            "cash": self.cash,
            "initialBalance": self.initial_balance,
            "positionsValue": self.positions_value,
            "totalValue": self.total_value,
            "realizedPnl": self.realized_pnl,
            "unrealizedPnl": self.unrealized_pnl,
            "staleSymbols": list(self.stale_symbols),
        }


@dataclass
class InvariantReport:
    """Result of checking the account identity."""

    ok: bool
    expected: float
    actual: float
    difference: float
    details: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# sells
# --------------------------------------------------------------------------

def validate_sell(held_quantity: Optional[int], requested_quantity: int) -> SellCheck:
    """Decide whether a sell may proceed, before anything is mutated.

    ``place_order`` used to credit cash for every sell unconditionally while
    ``_update_position`` quietly did nothing when there was no position, and
    deleted the row (still crediting the full notional) when the requested
    quantity exceeded what was held. Both minted cash from nothing. Shorting is
    not supported by this account model, so the answer is to refuse.
    """
    if requested_quantity is None or requested_quantity <= 0:
        return SellCheck(
            ok=False,
            reason="invalid_quantity",
            message="Sell quantity must be a positive number",
        )

    held = held_quantity or 0
    if held <= 0:
        return SellCheck(
            ok=False,
            reason="no_position",
            message="Cannot sell a symbol with no open position (shorting is not supported)",
        )

    if requested_quantity > held:
        return SellCheck(
            ok=False,
            reason="insufficient_shares",
            message=(
                f"Cannot sell {requested_quantity} shares; only {held} held"
            ),
        )

    return SellCheck(ok=True, quantity=requested_quantity)


def reduce_position(held_quantity: int, sold_quantity: int) -> int:
    """Remaining quantity after a validated sell. Never negative."""
    return max(held_quantity - sold_quantity, 0)


# --------------------------------------------------------------------------
# buys
# --------------------------------------------------------------------------

def weighted_average_entry(
    held_quantity: int,
    held_avg_price: float,
    added_quantity: int,
    added_price: float,
) -> float:
    """Blended entry price after scaling into a position.

    ``base_engine`` did ``quantity = quantity + %s`` without recomputing
    ``entry_price``, so every scale-in left ``avgPrice`` reporting the *first*
    fill and every subsequent P&L calculation was wrong.
    """
    held_quantity = max(held_quantity or 0, 0)
    added_quantity = max(added_quantity or 0, 0)
    total = held_quantity + added_quantity
    if total <= 0:
        return 0.0
    if held_quantity == 0:
        return float(added_price)
    return ((held_avg_price * held_quantity) + (added_price * added_quantity)) / total


# --------------------------------------------------------------------------
# marking
# --------------------------------------------------------------------------

def mark_position(
    symbol: str,
    quantity: int,
    avg_price: float,
    price_map: Mapping[str, Optional[float]],
) -> PositionMark:
    """Value one position against a price map.

    A symbol missing from the map -- or present with a null or non-positive
    price -- is marked stale rather than silently falling back to ``avg_price``.
    """
    raw = price_map.get(symbol)
    try:
        price = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        price = None
    if price is not None and price <= 0:
        price = None

    stale = price is None
    quantity = int(quantity or 0)
    avg_price = float(avg_price or 0.0)

    # A stale position is still worth *something*; carrying it at cost keeps the
    # account total finite, and `stale` plus `staleSymbols` says not to trust it.
    effective = avg_price if stale else price
    market_value = effective * quantity
    unrealized = (effective - avg_price) * quantity
    pct = ((effective / avg_price) - 1.0) * 100.0 if avg_price > 0 else 0.0

    return PositionMark(
        symbol=symbol,
        quantity=quantity,
        avg_price=avg_price,
        current_price=price,
        stale=stale,
        market_value=market_value,
        unrealized_pnl=0.0 if stale else unrealized,
        unrealized_pnl_pct=0.0 if stale else pct,
    )


def value_account(
    cash: float,
    initial_balance: float,
    positions: Iterable[Mapping],
    price_map: Mapping[str, Optional[float]],
    realized_pnl: float = 0.0,
) -> AccountValuation:
    """Value the whole account from ONE price map.

    ``totalValue`` used to come from the bot's own per-position ``get_quote``
    calls while ``positionsValue`` was recomputed in the dashboard from a
    different price source that fell back to ``avgPrice``. The two answers
    disagreed by exactly that divergence -- the $361.74 gap. Both numbers are
    now derived here, from the same map, in one place.
    """
    positions_value = 0.0
    unrealized = 0.0
    stale = []

    for row in positions:
        mark = mark_position(
            row["symbol"],
            row.get("quantity", 0),
            row.get("avgPrice", row.get("entry_price", 0.0)),
            price_map,
        )
        positions_value += mark.market_value
        unrealized += mark.unrealized_pnl
        if mark.stale:
            stale.append(mark.symbol)

    return AccountValuation(
        cash=float(cash),
        initial_balance=float(initial_balance),
        positions_value=positions_value,
        total_value=float(cash) + positions_value,
        realized_pnl=float(realized_pnl),
        unrealized_pnl=unrealized,
        stale_symbols=tuple(stale),
    )


# --------------------------------------------------------------------------
# the invariant
# --------------------------------------------------------------------------

def check_invariant(
    valuation: AccountValuation,
    tolerance: float = CENT,
) -> InvariantReport:
    """Assert ``cash + positions == initial + realized + unrealized``.

    This is the identity the live account violated. Checking it costs a
    subtraction and turns a slow, silent drift into something a sentinel can
    alert on.
    """
    actual = valuation.cash + valuation.positions_value
    expected = (
        valuation.initial_balance
        + valuation.realized_pnl
        + valuation.unrealized_pnl
    )
    difference = actual - expected

    return InvariantReport(
        ok=abs(difference) <= tolerance,
        expected=expected,
        actual=actual,
        difference=difference,
        details={
            "cash": valuation.cash,
            "positionsValue": valuation.positions_value,
            "initialBalance": valuation.initial_balance,
            "realizedPnl": valuation.realized_pnl,
            "unrealizedPnl": valuation.unrealized_pnl,
            "staleSymbols": list(valuation.stale_symbols),
        },
    )


def replay(
    initial_balance: float,
    fills: Sequence[Mapping],
) -> Dict:
    """Replay a fill sequence and return the resulting book.

    Used by the invariant tests: any sequence of validated fills must leave the
    account satisfying :func:`check_invariant`. Rejected sells must leave both
    cash and positions untouched.

    Each fill is ``{"symbol", "side", "quantity", "price"}``.
    """
    cash = float(initial_balance)
    realized = 0.0
    positions: Dict[str, Dict] = {}
    rejected = []

    for fill in fills:
        symbol = fill["symbol"]
        side = str(fill["side"]).lower()
        qty = int(fill["quantity"])
        price = float(fill["price"])
        held = positions.get(symbol)

        if side == "buy":
            cost = price * qty
            if cost > cash + CENT:
                rejected.append({**dict(fill), "reason": "insufficient_funds"})
                continue
            if held:
                held["avgPrice"] = weighted_average_entry(
                    held["quantity"], held["avgPrice"], qty, price,
                )
                held["quantity"] += qty
            else:
                positions[symbol] = {
                    "symbol": symbol, "quantity": qty, "avgPrice": price,
                }
            cash -= cost

        elif side == "sell":
            check = validate_sell(held["quantity"] if held else 0, qty)
            if not check.ok:
                rejected.append({**dict(fill), "reason": check.reason})
                continue
            realized += (price - held["avgPrice"]) * qty
            remaining = reduce_position(held["quantity"], qty)
            if remaining == 0:
                del positions[symbol]
            else:
                held["quantity"] = remaining
            cash += price * qty

        else:
            rejected.append({**dict(fill), "reason": "unknown_side"})

    return {
        "cash": cash,
        "realized_pnl": realized,
        "positions": list(positions.values()),
        "rejected": rejected,
    }
