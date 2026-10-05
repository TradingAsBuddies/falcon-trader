"""The screener -> orchestrator handoff, read from the database.

The regression: the orchestrator read 'screened_stocks.json' as a relative
path. The screener writes that file into its own container's volume, and the
orchestrator runs in /app, so it was never found. Every cycle printed "Screener
file not found" and 24 days passed with no trade and no alarm.
"""

import datetime as dt

import pytest

from falcon_trader import screener_feed


class _FakeDB:
    """Answers the one query the feed makes; records what it was asked."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(self, query, params=None, fetch=None):
        self.calls.append((query, params, fetch))
        return self.rows


NOW = dt.datetime(2026, 10, 5, 12, 0)  # naive UTC, as the column holds


def _row(stamp, recs):
    return {"run_timestamp": stamp, "run_data": {"recommendations": recs}}


def _rec(ticker, confidence=5, **over):
    rec = {"ticker": ticker, "confidence_score": confidence,
           "stop_loss": "$24.70", "target_price": "$29.50"}
    rec.update(over)
    return rec


# ── reading the screen ──────────────────────────────────────────────────

def test_recommendations_come_back_with_the_newest_timestamp():
    morning = dt.datetime(2026, 10, 5, 8, 0)
    db = _FakeDB([_row(morning, [_rec("APLD", 7), _rec("NXL", 6)])])
    recs, newest = screener_feed.latest_recommendations(db, now=NOW)
    assert [r["ticker"] for r in recs] == ["APLD", "NXL"]
    assert newest == morning


def test_results_are_ordered_by_confidence():
    db = _FakeDB([_row(NOW, [_rec("LOW", 3), _rec("HIGH", 9), _rec("MID", 6)])])
    recs, _ = screener_feed.latest_recommendations(db, now=NOW)
    assert [r["ticker"] for r in recs] == ["HIGH", "MID", "LOW"]


def test_a_ticker_from_two_profiles_appears_once_at_its_best_confidence():
    """Profiles overlap; the orchestrator should consider a name once."""
    db = _FakeDB([
        _row(dt.datetime(2026, 10, 5, 8, 0), [_rec("GME", 5, reasoning="seasonal")]),
        _row(dt.datetime(2026, 10, 4, 23, 0), [_rec("GME", 8, reasoning="momentum")]),
    ])
    recs, _ = screener_feed.latest_recommendations(db, now=NOW)
    assert len(recs) == 1
    assert recs[0]["confidence_score"] == 8
    assert recs[0]["reasoning"] == "momentum"


def test_symbol_is_accepted_in_place_of_ticker_and_upcased():
    db = _FakeDB([_row(NOW, [{"symbol": "aapl", "confidence_score": 4}])])
    recs, _ = screener_feed.latest_recommendations(db, now=NOW)
    assert recs[0]["ticker"] == "AAPL"


def test_run_timestamp_is_attached_to_each_recommendation():
    stamp = dt.datetime(2026, 10, 5, 8, 0)
    db = _FakeDB([_row(stamp, [_rec("APLD")])])
    recs, _ = screener_feed.latest_recommendations(db, now=NOW)
    assert recs[0]["_run_timestamp"] == stamp.isoformat()


# ── the age window ──────────────────────────────────────────────────────

def test_cutoff_is_bound_as_a_parameter_in_utc():
    """The column is naive UTC; `now() - interval` is also PostgreSQL-only."""
    db = _FakeDB([])
    screener_feed.latest_recommendations(db, now=NOW, max_age=dt.timedelta(hours=24))
    query, params, _ = db.calls[0]
    assert "interval" not in query.lower() and "now()" not in query.lower()
    assert params == (NOW - dt.timedelta(hours=24),)


def test_default_window_is_a_day():
    assert screener_feed.DEFAULT_MAX_AGE == dt.timedelta(hours=24)


def test_utc_now_naive_has_no_tzinfo_and_is_utc():
    """A naive local clock would be four hours off the column in these containers."""
    now = screener_feed.utc_now_naive()
    assert now.tzinfo is None
    reference = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    assert abs((now - reference).total_seconds()) < 5


# ── nothing to act on ───────────────────────────────────────────────────

def test_no_runs_returns_empty_and_no_timestamp():
    assert screener_feed.latest_recommendations(_FakeDB([]), now=NOW) == ([], None)


def test_none_from_the_database_is_tolerated():
    assert screener_feed.latest_recommendations(_FakeDB(None), now=NOW) == ([], None)


@pytest.mark.parametrize("run_data", [None, {}, {"recommendations": None},
                                      {"recommendations": "nope"}, "junk", 7])
def test_malformed_run_data_yields_no_recommendations(run_data):
    db = _FakeDB([{"run_timestamp": NOW, "run_data": run_data}])
    recs, newest = screener_feed.latest_recommendations(db, now=NOW)
    assert recs == []
    assert newest == NOW  # the run still happened


def test_rows_without_a_ticker_are_skipped():
    db = _FakeDB([_row(NOW, [{"confidence_score": 9}, {"ticker": "  "}, _rec("OK")])])
    recs, _ = screener_feed.latest_recommendations(db, now=NOW)
    assert [r["ticker"] for r in recs] == ["OK"]


def test_unparseable_confidence_does_not_raise():
    db = _FakeDB([_row(NOW, [_rec("A", "high"), _rec("B", 2)])])
    recs, _ = screener_feed.latest_recommendations(db, now=NOW)
    assert {r["ticker"] for r in recs} == {"A", "B"}
