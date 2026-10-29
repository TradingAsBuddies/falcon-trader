"""The Strategy Manager: the roster, and moving a strategy through it.

The dashboard advertised a Strategy Manager whose card linked to
/api/strategy -- the deployed strategy's *source code* as JSON -- and whose
page was never built. The endpoints that looked like one read and wrote
`active_strategies`, a table that has never existed in this database, so
/api/strategies/active always answered "none" while the roster held nine, and
pause/resume reported success for rows they never touched.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

from falcon_trader import strategy_roster_api as api

flask = pytest.importorskip("flask")

NOW = dt.datetime(2026, 10, 29, 9, 30)


# ── the lifecycle ───────────────────────────────────────────────────────

@pytest.mark.parametrize("current,target,move", [
    ("backtest", "paper_trading", "promote"),
    ("paper_trading", "backtest", "demote"),
    ("backtest", "retired", "retire"),
    ("paper_trading", "retired", "retire"),
    ("retired", "backtest", "reinstate"),
])
def test_allowed_moves_are_named(current, target, move):
    assert api.transition_name(current, target) == move


def test_a_retired_strategy_cannot_jump_straight_to_paper_trading():
    """It has to be backtested again first."""
    with pytest.raises(api.TransitionError) as exc:
        api.transition_name("retired", "paper_trading")
    assert "cannot go from retired to paper_trading" in str(exc.value)
    assert "backtest" in str(exc.value)


def test_moving_to_the_same_status_is_refused():
    with pytest.raises(api.TransitionError) as exc:
        api.transition_name("backtest", "backtest")
    assert "already backtest" in str(exc.value)


@pytest.mark.parametrize("target", ["", "live", "paused", "deleted", "BACKTEST"])
def test_an_unknown_status_is_refused(target):
    with pytest.raises(api.TransitionError) as exc:
        api.transition_name("backtest", target)
    assert "unknown status" in str(exc.value)


# ── the note ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "", "   ", 42, "\n\t "])
def test_a_note_is_required(raw):
    with pytest.raises(ValueError) as exc:
        api.clean_note(raw)
    assert "note is required" in str(exc.value)


def test_a_note_is_collapsed_to_one_line():
    assert api.clean_note(" zero\ntrades\tin 3 cycles ") == "zero trades in 3 cycles"


def test_a_note_is_bounded():
    api.clean_note("x" * api.MAX_NOTE_LENGTH)
    with pytest.raises(ValueError):
        api.clean_note("x" * (api.MAX_NOTE_LENGTH + 1))


# ── the roster view ─────────────────────────────────────────────────────

def _row(name="atr_breakout", status="backtest", **over):
    row = {
        "id": 2, "strategy_name": name, "status": status, "interval": "5m",
        "symbols": ["AMD", "NVDA"], "params": {"max_price": 500.0, "min_price": 1.0},
        "review_notes": None, "promoted_at": None, "demoted_at": None,
        "last_backtest_at": dt.datetime(2026, 10, 22, 15, 35),
        "backtest_sharpe": -10.75, "backtest_win_rate": 0.1212,
        "backtest_profit_factor": 0.4, "backtest_total_return": -0.1274,
        "paper_sharpe": None, "paper_win_rate": None, "paper_profit_factor": None,
    }
    row.update(over)
    return row


def test_a_strategy_view_carries_what_the_page_shows():
    view = api.strategy_view(_row(), {"runs": 326, "recent_runs": 14,
                                      "signals": 78, "trades": 30})
    assert view["name"] == "atr_breakout"
    assert view["symbols"] == ["AMD", "NVDA"]
    assert view["backtest"]["sharpe"] == pytest.approx(-10.75)
    assert (view["runs"], view["signals"], view["trades"]) == (326, 78, 30)
    assert view["available_transitions"] == ["paper_trading", "retired"]


def test_signals_and_trades_default_to_zero_without_runs():
    view = api.strategy_view(_row())
    assert (view["runs"], view["recent_runs"], view["signals"], view["trades"]) == (0, 0, 0, 0)


def test_activity_is_windowed_so_the_two_figures_share_an_era():
    """signals_count only exists for runs since falcon-core#42 (2026-10-16).

    Summed all time against total_trades, which goes back to August, it
    produced nonsense: opening_range_breakout read 307 trades, 0 signals.
    """
    view = api.strategy_view(_row(), {"runs": 93, "recent_runs": 0,
                                      "signals": 0, "trades": 0})
    assert view["runs"] == 93 and view["recent_runs"] == 0
    assert view["activity_window_days"] == api.ACTIVITY_WINDOW_DAYS


def test_the_aggregate_query_bounds_the_window_as_a_parameter():
    """No `now()` or `interval` in the SQL: this runs on SQLite too."""
    class _Recorder(_FakeDB):
        def __init__(self):
            super().__init__()
            self.queries = []

        def execute(self, query, params=None, fetch=None):
            self.queries.append((" ".join(query.split()), params))
            return super().execute(query, params, fetch)

    db = _Recorder()
    api.roster(db, now=NOW)
    agg = next(q for q, _ in db.queries if "GROUP BY strategy_name" in q)
    assert "interval" not in agg.lower() and "now()" not in agg.lower()
    assert "FILTER" not in agg.upper()          # not portable to older SQLite
    params = next(p for q, p in db.queries if "GROUP BY strategy_name" in q)
    assert params == (NOW - dt.timedelta(days=api.ACTIVITY_WINDOW_DAYS),) * 3


def test_json_string_columns_are_tolerated():
    """jsonb returns objects; an older row may hold a JSON string."""
    view = api.strategy_view(_row(symbols='["SPY", "QQQ"]', params='{"min_price": 1}'))
    assert view["symbols"] == ["SPY", "QQQ"]
    assert view["params"] == {"min_price": 1}


def test_a_retired_strategy_offers_only_reinstatement():
    assert api.strategy_view(_row(status="retired"))["available_transitions"] == ["backtest"]


# ── applying a transition ───────────────────────────────────────────────

class _FakeDB:
    def __init__(self, rows=None, aggregates=None):
        self.rows = rows if rows is not None else [_row()]
        self.aggregates = aggregates or []
        self.writes = []

    def execute(self, query, params=None, fetch=None):
        flat = " ".join(query.split())
        if flat.startswith("UPDATE"):
            self.writes.append((flat, params))
            return 1
        if "FROM backtest_runs" in flat and "GROUP BY" in flat:
            return self.aggregates
        if "FROM backtest_runs" in flat:
            return []
        if "WHERE strategy_name" in flat:
            return next((r for r in self.rows if r["strategy_name"] == params[0]), None)
        return self.rows


def test_a_promotion_records_the_note_and_the_time():
    db = _FakeDB()
    result = api.apply_transition(db, "atr_breakout", "paper_trading", "sharpe recovered", NOW)
    assert result["transition"] == "promote"
    assert result["from"] == "backtest" and result["to"] == "paper_trading"
    query, params = db.writes[0]
    assert "promoted_at" in query
    assert params[0] == "paper_trading"
    assert "promote: backtest -> paper_trading. sharpe recovered" in params[1]
    assert "[2026-10-29 09:30]" in params[1]


def test_notes_accumulate_rather_than_replace():
    """The history of a strategy's judgement has to survive."""
    db = _FakeDB([_row(review_notes="[2026-08-13] retire: zero trades")])
    api.apply_transition(db, "atr_breakout", "retired", "still nothing", NOW)
    _, params = db.writes[0]
    assert "[2026-08-13] retire: zero trades" in params[1]
    assert "still nothing" in params[1]


def test_a_demotion_stamps_demoted_at():
    db = _FakeDB([_row(status="paper_trading")])
    api.apply_transition(db, "atr_breakout", "backtest", "losing money", NOW)
    query, _ = db.writes[0]
    assert "demoted_at" in query


def test_retiring_stamps_demoted_at():
    db = _FakeDB()
    api.apply_transition(db, "atr_breakout", "retired", "done", NOW)
    assert "demoted_at" in db.writes[0][0]


def test_reinstating_stamps_neither():
    db = _FakeDB([_row(status="retired")])
    api.apply_transition(db, "atr_breakout", "backtest", "worth another look", NOW)
    query, _ = db.writes[0]
    assert "promoted_at" not in query and "demoted_at" not in query


def test_an_unknown_strategy_raises_lookup():
    with pytest.raises(LookupError):
        api.apply_transition(_FakeDB([]), "nope", "retired", "x", NOW)


def test_a_refused_move_writes_nothing():
    db = _FakeDB([_row(status="retired")])
    with pytest.raises(api.TransitionError):
        api.apply_transition(db, "atr_breakout", "paper_trading", "x", NOW)
    assert db.writes == []


# ── through the routes ──────────────────────────────────────────────────

def _app(db):
    app = flask.Flask(__name__)
    app.register_blueprint(api.create_blueprint(get_db=lambda: db))
    return app


def test_the_roster_route_groups_running_strategies_first():
    db = _FakeDB([_row("atr_breakout", "backtest"),
                  _row("opening_range_breakout", "paper_trading"),
                  _row("one_candle_rule", "retired")],
                 aggregates=[{"strategy_name": "atr_breakout", "runs": 326,
                              "recent_runs": 14, "signals": 78, "trades": 30}])
    body = _app(db).test_client().get("/api/roster").get_json()
    assert [s["status"] for s in body["strategies"]] == [
        "paper_trading", "backtest", "retired"]
    assert body["statuses"] == list(api.STATUSES)


def test_the_status_route_promotes():
    db = _FakeDB()
    response = _app(db).test_client().post(
        "/api/roster/atr_breakout/status",
        json={"target": "paper_trading", "note": "sharpe recovered"})
    assert response.status_code == 200
    assert response.get_json()["transition"] == "promote"
    assert db.writes


def test_the_status_route_requires_a_note():
    db = _FakeDB()
    response = _app(db).test_client().post(
        "/api/roster/atr_breakout/status", json={"target": "retired"})
    assert response.status_code == 400
    assert "note is required" in response.get_json()["error"]
    assert db.writes == []


def test_the_status_route_refuses_an_illegal_move_with_409():
    db = _FakeDB([_row(status="retired")])
    response = _app(db).test_client().post(
        "/api/roster/atr_breakout/status",
        json={"target": "paper_trading", "note": "skip the queue"})
    assert response.status_code == 409
    assert db.writes == []


def test_the_status_route_404s_on_an_unknown_strategy():
    response = _app(_FakeDB([])).test_client().post(
        "/api/roster/ghost/status", json={"target": "retired", "note": "x"})
    assert response.status_code == 404


@pytest.mark.parametrize("payload", [None, [], "retire", 7])
def test_non_object_json_is_a_400_not_a_500(payload):
    response = _app(_FakeDB()).test_client().post(
        "/api/roster/atr_breakout/status", json=payload)
    assert response.status_code == 400


def test_the_backtests_route_returns_runs():
    body = _app(_FakeDB()).test_client().get("/api/roster/atr_breakout/backtests").get_json()
    assert body["strategy"] == "atr_breakout"
    assert body["runs"] == []
