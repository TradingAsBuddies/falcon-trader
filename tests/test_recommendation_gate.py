"""Which recommendations may be presented as actionable.

The case these exist for is the live payload from 2026-09-30 06:50 ET: the
screener had just produced 13 picks from a Finviz screen run minutes earlier,
the intraday scanner had 5 setups from the previous session's flat file, and
the page showed nothing at all. The scanner's "STALE (EOD flat-file, as of
2026-09-29)" was applied to every merged row.
"""

import datetime as dt

import pytest

from falcon_trader import recommendation_gate as gate
from falcon_trader.orchestrator.utils.timezone import ET

NOW = dt.datetime(2026, 9, 30, 6, 50, tzinfo=ET)
SCAN_STALE = "STALE (EOD flat-file, as of 2026-09-29)"
SCAN_LIVE = "LIVE (minute bars, 1m behind)"


def _screener(ticker="CAPR", **over):
    """A screener pick: no recency label of its own, no validity window."""
    rec = {"ticker": ticker, "_theme": "trapped_shorts",
           "_profile_source": "3rd Day Setup (Trapped Shorts)",
           "confidence_score": 8, "data_recency": None, "valid_until": None}
    rec.update(over)
    return rec


def _intraday(ticker="RIOT", **over):
    rec = {"ticker": ticker, "_theme": "intraday_setup",
           "_profile_source": "intraday_scan", "edge_score": 2.1,
           "data_recency": SCAN_STALE,
           "valid_until": "2026-09-29T16:00:00-04:00"}
    rec.update(over)
    return rec


# ── the regression ──────────────────────────────────────────────────────

def test_fresh_screener_picks_survive_a_stale_scan():
    """The bug: a flat-file scan's staleness withheld every screener pick."""
    live, withheld = gate.mark_actionable(
        [_screener("CAPR"), _screener("INTC"), _intraday()], SCAN_STALE, NOW)
    assert [r["ticker"] for r in live] == ["CAPR", "INTC"]
    assert [r["ticker"] for r in withheld] == ["RIOT"]


def test_stale_scan_still_withholds_its_own_setups():
    """Window still open, data stale: withheld for staleness, not expiry."""
    setup = _intraday(valid_until="2026-09-30T16:00:00-04:00")
    live, withheld = gate.mark_actionable([setup], SCAN_STALE, NOW)
    assert live == []
    assert withheld[0]["expired"] is False
    assert "stale data" in withheld[0]["withheld_reason"]
    assert withheld[0]["actionable"] is False


def test_expiry_is_reported_ahead_of_staleness():
    """Yesterday's setup on yesterday's data: the closed window is the reason."""
    live, withheld = gate.mark_actionable([_intraday()], SCAN_STALE, NOW)
    assert live == []
    assert withheld[0]["withheld_reason"] == "past its validity window"


def test_intraday_setups_are_actionable_when_the_scan_is_live():
    setup = _intraday(data_recency=SCAN_LIVE,
                      valid_until="2026-09-30T16:00:00-04:00")
    live, withheld = gate.mark_actionable([setup], SCAN_LIVE, NOW)
    assert [r["ticker"] for r in live] == ["RIOT"]
    assert withheld == []


def test_a_screener_row_with_its_own_stale_label_is_withheld():
    """Per-row labels still count; only the scan's label is scoped."""
    live, withheld = gate.mark_actionable(
        [_screener(data_recency="STALE (something)")], SCAN_LIVE, NOW)
    assert live == []
    assert "stale data" in withheld[0]["withheld_reason"]


# ── expiry ──────────────────────────────────────────────────────────────

def test_expired_window_is_withheld_even_on_live_data():
    rec = _screener(valid_until="2026-09-30T06:00:00-04:00")
    live, withheld = gate.mark_actionable([rec], SCAN_LIVE, NOW)
    assert live == []
    assert withheld[0]["expired"] is True
    assert withheld[0]["withheld_reason"] == "past its validity window"


def test_window_still_open_is_actionable():
    rec = _screener(valid_until="2026-09-30T16:00:00-04:00")
    live, _ = gate.mark_actionable([rec], SCAN_LIVE, NOW)
    assert [r["ticker"] for r in live] == ["CAPR"]


def test_naive_valid_until_is_not_compared():
    """`now` is Eastern and the containers run UTC; a naive value is 4h out.

    Comparing it would expire setups four hours early, in the safe-looking
    direction, which is why it would go unnoticed.
    """
    rec = _screener(valid_until="2026-09-30T09:00:00")
    live, _ = gate.mark_actionable([rec], SCAN_LIVE, NOW)
    assert [r["ticker"] for r in live] == ["CAPR"]


@pytest.mark.parametrize("raw", [None, "", "not a date", 0])
def test_unparseable_window_means_no_window(raw):
    assert gate.is_expired({"valid_until": raw}, NOW) is False


# ── nothing is dropped ──────────────────────────────────────────────────

def test_every_row_is_returned_somewhere():
    recs = [_screener("A"), _intraday("B"), _screener("C", valid_until="2026-09-30T06:00:00-04:00")]
    live, withheld = gate.mark_actionable(recs, SCAN_STALE, NOW)
    assert len(live) + len(withheld) == 3
    assert {r["ticker"] for r in live} | {r["ticker"] for r in withheld} == {"A", "B", "C"}


def test_actionable_rows_carry_no_withheld_reason():
    live, _ = gate.mark_actionable([_screener(withheld_reason="left over")], SCAN_LIVE, NOW)
    assert "withheld_reason" not in live[0]


# ── source classification ───────────────────────────────────────────────

@pytest.mark.parametrize("rec,expected", [
    ({"_theme": "intraday_setup"}, True),
    ({"_profile_source": "intraday_scan"}, True),
    ({"_theme": "momentum", "_profile_source": "Momentum Breakouts"}, False),
    ({}, False),
])
def test_is_from_intraday_scan(rec, expected):
    assert gate.is_from_intraday_scan(rec) is expected


# ── the message ─────────────────────────────────────────────────────────

def test_message_is_none_when_something_is_actionable():
    assert gate.summary_message([_screener()], [_intraday()]) is None


def test_message_counts_the_reasons_separately():
    withheld = [_intraday(), _screener(expired=True)]
    withheld[1]["expired"] = True
    msg = gate.summary_message([], withheld)
    assert "1 past their validity window" in msg
    assert "1 on stale data" in msg
    assert "Do NOT trade these" in msg


def test_message_when_there_is_nothing_at_all():
    assert gate.summary_message([], []) == "No screening results available yet"
