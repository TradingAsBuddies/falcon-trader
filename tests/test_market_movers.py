"""Tests for the market page's gainers/losers.

Fixtures are the shapes Polygon's snapshot actually returned before the open on
2026-09-17: a `day` object present with close and volume both 0, the real
price and volume in `min`, and no `lastTrade` key at all.
"""

import pytest

from falcon_trader.market_movers import (
    build_movers,
    is_common_listing,
    snapshot_price,
    snapshot_volume,
)


def _row(ticker, *, day_c=0, day_v=0, min_c=None, min_av=None, prev_c=None,
         chg=0.0, last_trade=None):
    row = {
        "ticker": ticker,
        "day": {"c": day_c, "v": day_v},
        "prevDay": {"c": prev_c} if prev_c is not None else {},
        "todaysChangePerc": chg,
    }
    if min_c is not None or min_av is not None:
        row["min"] = {"c": min_c, "av": min_av}
    if last_trade is not None:
        row["lastTrade"] = {"p": last_trade}
    return row


# Real pre-market rows, 2026-09-17.
CTNT = _row("CTNT", min_c=0.1098, min_av=510480309, prev_c=0.0417, chg=162.1)
GNRC = _row("GNRC", min_c=235.99, min_av=42715, prev_c=175.11, chg=34.8)
DAICW = _row("DAICW", min_c=0.02, min_av=400, prev_c=0.007, chg=185.7)
RIVR = _row("RIVr", min_c=0.0064, min_av=1500, prev_c=0.004, chg=60.0)


# ── is_common_listing ───────────────────────────────────────────────────

@pytest.mark.parametrize("ticker", ["DAICW", "REVBW", "ONFOW", "OPENW"])
def test_nasdaq_warrants_are_excluded(ticker):
    """All confirmed WARRANT by Polygon's reference API."""
    assert not is_common_listing(ticker)


def test_z_suffix_additional_warrant_series_is_excluded():
    """OPENZ is a WARRANT — a second series beside OPENW.

    The first version of this rule used W/R/U and kept it. It stayed off the
    page only because it ranked 13th, below the top-ten cut: excluded by luck,
    not by the filter.
    """
    assert not is_common_listing("OPENZ")


def test_lowercase_suffix_right_is_excluded():
    """RIVr confirmed RIGHT; the lowercase letter is the tell."""
    assert not is_common_listing("RIVr")


@pytest.mark.parametrize("ticker", ["DAIC", "GNRC", "CTNT", "GOOGL"])
def test_common_stock_is_kept(ticker):
    assert is_common_listing(ticker)


@pytest.mark.parametrize("ticker", ["SNOW", "ARKW"])
def test_four_letter_symbols_ending_in_w_are_kept(ticker):
    """The reason the rule is not "ends in W".

    SNOW is common stock and ARKW is an ETF. The fifth-character convention
    only applies to five-letter symbols.
    """
    assert is_common_listing(ticker)


@pytest.mark.parametrize("ticker", ["ABCDU", "ABCDR"])
def test_nasdaq_units_and_rights_are_excluded(ticker):
    assert not is_common_listing(ticker)


@pytest.mark.parametrize("ticker,expected", [
    ("BRK.B", True),    # class B shares are common stock
    ("BRK.A", True),
    ("XYZ.WS", False),  # warrant
    ("XYZ.U", False),   # unit
    ("XYZ.RT", False),  # right
])
def test_dot_suffixes(ticker, expected):
    assert is_common_listing(ticker) is expected


@pytest.mark.parametrize("ticker", ["", None])
def test_missing_ticker_is_not_a_listing(ticker):
    assert not is_common_listing(ticker)


# ── snapshot_price / snapshot_volume ────────────────────────────────────

def test_premarket_price_comes_from_the_minute_bar():
    """The regression: `day.c` is 0 before the open and was returned as-is."""
    assert snapshot_price(CTNT) == 0.1098
    assert snapshot_price(GNRC) == 235.99


def test_premarket_volume_comes_from_accumulated_minute_volume():
    assert snapshot_volume(CTNT) == 510480309
    assert snapshot_volume(GNRC) == 42715


def test_regular_session_prefers_the_day_bar():
    row = _row("GNRC", day_c=236.50, day_v=1_200_000, min_c=236.40, min_av=1_199_000)
    assert snapshot_price(row) == 236.50
    assert snapshot_volume(row) == 1_200_000


def test_last_trade_is_used_when_present():
    """Not on this plan's snapshot today, but honoured if a feed supplies it."""
    row = {"ticker": "X", "day": {"c": 0, "v": 0}, "lastTrade": {"p": 12.34}}
    assert snapshot_price(row) == 12.34


def test_previous_close_is_never_presented_as_current_price():
    """Yesterday's close shown as today's price would be quietly wrong."""
    row = _row("X", prev_c=50.0)
    assert snapshot_price(row) == 0


def test_rows_missing_nested_objects_do_not_raise():
    assert snapshot_price({"ticker": "X"}) == 0
    assert snapshot_volume({"ticker": "X", "day": None, "min": None}) == 0


# ── build_movers ────────────────────────────────────────────────────────

def test_movers_carry_real_prices_and_volumes():
    movers = build_movers([CTNT, GNRC])
    assert movers == [
        {"symbol": "CTNT", "price": 0.1098, "change_pct": 162.1, "volume": 510480309},
        {"symbol": "GNRC", "price": 235.99, "change_pct": 34.8, "volume": 42715},
    ]


def test_derivatives_are_dropped_and_order_is_preserved():
    movers = build_movers([DAICW, CTNT, RIVR, GNRC])
    assert [m["symbol"] for m in movers] == ["CTNT", "GNRC"]


def test_filters_before_truncating():
    """Taking ten then filtering would return a short list.

    Here the first four rows are derivatives; the page should still get ten
    common listings when the snapshot has them.
    """
    derivatives = [DAICW, RIVR, _row("REVBW", min_c=1, min_av=1), _row("ONFOW", min_c=1, min_av=1)]
    commons = [_row(f"C{i:03d}", min_c=1.0 + i, min_av=100 + i) for i in range(15)]
    movers = build_movers(derivatives + commons, limit=10)
    assert len(movers) == 10
    assert movers[0]["symbol"] == "C000"


def test_empty_snapshot():
    assert build_movers([]) == []
    assert build_movers(None) == []
