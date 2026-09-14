"""
Retail-method financial arithmetic.

All prices are converted to whole-number cents for intermediate math,
then converted back to dollars-and-cents for storage and display.
This eliminates floating-point drift in financial calculations.

Usage:
    from falcon_trader.orchestrator.utils.cents import to_cents, to_dollars, calc_cost, calc_pnl
"""


def to_cents(price: float) -> int:
    """Convert a dollar price to whole cents."""
    return round(price * 100)


def to_dollars(cents: int) -> float:
    """Convert whole cents back to dollars and cents."""
    return cents / 100


def calc_cost(quantity: int, price: float) -> float:
    """Calculate total cost in dollars: quantity * price, via cents."""
    return to_dollars(quantity * to_cents(price))


def calc_pnl(sell_price: float, entry_price: float, quantity: int) -> float:
    """Calculate P&L in dollars: (sell - entry) * qty, via cents."""
    return to_dollars((to_cents(sell_price) - to_cents(entry_price)) * quantity)


def calc_avg_price(old_price: float, old_qty: int,
                   new_price: float, new_qty: int) -> float:
    """Calculate weighted-average price via cents."""
    total_cents = to_cents(old_price) * old_qty + to_cents(new_price) * new_qty
    total_qty = old_qty + new_qty
    if total_qty == 0:
        return 0.0
    return to_dollars(total_cents // total_qty)
