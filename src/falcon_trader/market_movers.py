"""Top gainers / losers from a Polygon snapshot, as the market page shows them.

Pure functions over the snapshot's ticker dicts, so the logic is testable
without Flask or the network. dashboard_server.api_market is the adapter.
"""

from typing import Any, Dict, Iterable, List, Optional

# Nasdaq's fifth-character identifiers for instruments that are not the
# common listing itself: W = warrant, R = right, U = unit, and Z, which Nasdaq
# defines as "miscellaneous" and issuers use for an additional warrant series.
#
# Z was added after W/R/U was measured against the live snapshot on 2026-09-17
# (42 unique tickers, each checked against Polygon's reference API):
#
#   W/R/U     kept OPENZ, a WARRANT (a second series beside OPENW)
#   W/R/U/Z   no derivative kept, no common listing dropped
#
# The residual risk is a genuine five-letter listing whose fifth character is
# Z; none appeared, and a "miscellaneous" issue is rarely what a movers list is
# for. If one is ever hidden, this set is where to look.
_NASDAQ_DERIVATIVE_FIFTH = frozenset("WRUZ")

# Dot-suffixed forms some feeds use for the same instruments. Class shares
# (BRK.A, BRK.B) are common stock and are deliberately absent.
_DOT_DERIVATIVE_SUFFIXES = frozenset({"W", "WS", "WT", "R", "RT", "U"})


def is_common_listing(ticker: Optional[str]) -> bool:
    """False for warrants, rights and units; True for the listings people trade.

    The market page's movers were led by instruments like DAICW (+328%, 400
    shares, $0.02) that no one means when they ask what is moving. Verified
    against Polygon's reference API on 2026-09-17:

        DAICW, REVBW, ONFOW, OPENW, OPENZ -> WARRANT      RIVr -> RIGHT
        DAIC, GNRC, CTNT, GOOGL -> CS       SNOW -> CS      ARKW -> ETF

    SNOW and ARKW are why the rule is not "ends in W": the fifth-character
    convention only applies to five-letter Nasdaq symbols, so a four-letter
    ticker ending in W is an ordinary listing.

    This is a convention, not a lookup. Querying the reference API per ticker
    would be authoritative but costs a request for each of ~20 movers on every
    page load, on an endpoint that already takes seconds.
    """
    if not ticker:
        return False

    # Polygon encodes NYSE share classes and instrument types in lowercase:
    # RIVr is a right. No common-stock symbol contains a lowercase letter.
    if any(c.islower() for c in ticker):
        return False

    if "." in ticker:
        return ticker.rsplit(".", 1)[1] not in _DOT_DERIVATIVE_SUFFIXES

    if len(ticker) >= 5 and ticker[-1] in _NASDAQ_DERIVATIVE_FIFTH:
        return False

    return True


def _first_positive(*values: Any) -> float:
    for v in values:
        if isinstance(v, (int, float)) and v > 0:
            return v
    return 0


def snapshot_price(row: Dict[str, Any]) -> float:
    """Latest price for a snapshot row, including outside regular hours.

    This used to be ``row['day'].get('c', lastTrade.p)``. Before the open
    Polygon sends a ``day`` object whose close is 0, so ``.get`` returned the
    0 — the key exists — and never reached the fallback. The fallback was dead
    anyway: this plan's snapshot carries no ``lastTrade`` field at all. The
    result was a price of 0 for every mover until 09:30 ET.

    ``min.c`` is the close of the most recent minute bar, which is populated
    pre- and post-market. ``prevDay.c`` is deliberately not a fallback:
    presenting yesterday's close as the current price would be quietly wrong
    in a way 0 at least is not.
    """
    return _first_positive(
        (row.get("day") or {}).get("c"),
        (row.get("min") or {}).get("c"),
        (row.get("lastTrade") or {}).get("p"),
    )


def snapshot_volume(row: Dict[str, Any]) -> float:
    """Shares traded today, including extended hours.

    ``day.v`` is 0 before the open, for the same reason as the price.
    ``min.av`` is the accumulated volume for the day so far.
    """
    return _first_positive(
        (row.get("day") or {}).get("v"),
        (row.get("min") or {}).get("av"),
    )


def build_movers(rows: Iterable[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """The first ``limit`` common listings from a snapshot, in snapshot order.

    Filters *before* truncating. The snapshot returns about twenty rows; taking
    ten and then dropping warrants would show a short list whenever a
    derivative ranked in the top ten, which on 2026-09-17 was most of them.
    """
    movers: List[Dict[str, Any]] = []
    for row in rows or []:
        symbol = row.get("ticker", "")
        if not is_common_listing(symbol):
            continue
        movers.append({
            "symbol": symbol,
            "price": snapshot_price(row),
            "change_pct": round(row.get("todaysChangePerc", 0) or 0, 2),
            "volume": snapshot_volume(row),
        })
        if len(movers) >= limit:
            break
    return movers
