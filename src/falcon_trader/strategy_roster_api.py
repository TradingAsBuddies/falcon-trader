"""The strategy roster, and its lifecycle, for the Strategy Manager page.

The dashboard advertised a Strategy Manager that did not exist: the card linked
to /api/strategy, which returns the deployed strategy's *source code* as JSON,
and the page it implied was never built. The endpoints that looked like a
manager -- /api/strategies/active, and pause/resume -- read and wrote a table
called `active_strategies` that has never existed in this database, so active
always came back empty and pause/resume silently did nothing.

The real roster is `strategy_roster`: nine strategies moving through a
lifecycle of backtest -> paper_trading -> retired, with backtest and paper
metrics recorded against each.

Scope is deliberate (operator's decision, 2026-10-23): read the roster and its
metrics, and move a strategy between states with a note. Strategy code is not
served here and deploy/rollback are not wired -- those endpoints execute
caller-supplied Python and stay behind FALCON_ALLOW_DEPLOY.

Every transition records *why*. A roster whose statuses change without reasons
is a roster nobody can audit, and "why is this one retired?" was already
unanswerable for one of the two retired strategies.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The lifecycle. A strategy is backtested, promoted to paper trading, and
#: retired when it stops earning its place.
BACKTEST = "backtest"
PAPER = "paper_trading"
RETIRED = "retired"

STATUSES = (BACKTEST, PAPER, RETIRED)

#: Allowed moves, and what each one is called. Anything absent is refused:
#: a strategy cannot go straight from retired to paper trading without being
#: backtested again.
TRANSITIONS: Dict[Tuple[str, str], str] = {
    (BACKTEST, PAPER): "promote",
    (PAPER, BACKTEST): "demote",
    (BACKTEST, RETIRED): "retire",
    (PAPER, RETIRED): "retire",
    (RETIRED, BACKTEST): "reinstate",
}

#: Longest accepted note, after whitespace is collapsed.
MAX_NOTE_LENGTH = 500

#: Window for the signals and trades figures.
#:
#: backtest_runs.signals_count was written from an attribute no result carried
#: until falcon-core#42 (2026-10-16), so every row before that reads 0 while
#: total_trades has accumulated since August. Summing all time mixes the two
#: eras and produces nonsense -- opening_range_breakout showed 307 trades and
#: 0 signals. Counting a recent window instead means both figures come from
#: runs that recorded both.
ACTIVITY_WINDOW_DAYS = 7


class TransitionError(ValueError):
    """The requested lifecycle move is not allowed."""


def clean_note(raw: Any) -> str:
    """A single-line, bounded note. Raises ValueError when unusable.

    Required, not optional: the whole point of recording a transition is the
    reason for it.
    """
    if not isinstance(raw, str):
        raise ValueError("a note is required to change a strategy's status")
    note = " ".join(raw.split())
    if not note:
        raise ValueError("a note is required to change a strategy's status")
    if len(note) > MAX_NOTE_LENGTH:
        raise ValueError(f"note must be at most {MAX_NOTE_LENGTH} characters")
    return note


def transition_name(current: str, target: str) -> str:
    """What the move from `current` to `target` is called, or raise."""
    if target not in STATUSES:
        raise TransitionError(
            f"unknown status '{target}'; expected one of {', '.join(STATUSES)}")
    if current == target:
        raise TransitionError(f"already {target}")
    name = TRANSITIONS.get((current, target))
    if name is None:
        allowed = sorted(to for (frm, to) in TRANSITIONS if frm == current)
        raise TransitionError(
            f"cannot go from {current} to {target}; from {current} a strategy "
            f"may become {', '.join(allowed) or 'nothing'}")
    return name


def _as_list(value: Any) -> List[Any]:
    """jsonb columns come back as lists; older rows may hold a JSON string."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def strategy_view(row: Any, runs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One roster row, shaped for the page.

    `runs` carries the aggregate from backtest_runs: how many runs all time,
    and how many runs, signals and trades inside ACTIVITY_WINDOW_DAYS. Signals
    beside trades is the distinction that was invisible until falcon-core#42: a
    strategy generating signals and trading none is failing differently from
    one that generates nothing.
    """
    runs = runs or {}
    status = row["status"]
    return {
        "id": row["id"],
        "name": row["strategy_name"],
        "status": status,
        "interval": row["interval"],
        "symbols": _as_list(row["symbols"]),
        "params": _as_dict(row["params"]),
        "review_notes": row["review_notes"],
        "promoted_at": str(row["promoted_at"]) if row.get("promoted_at") else None,
        "demoted_at": str(row["demoted_at"]) if row.get("demoted_at") else None,
        "last_backtest_at": (str(row["last_backtest_at"])
                             if row.get("last_backtest_at") else None),
        "backtest": {
            "sharpe": _number(row.get("backtest_sharpe")),
            "win_rate": _number(row.get("backtest_win_rate")),
            "profit_factor": _number(row.get("backtest_profit_factor")),
            "total_return": _number(row.get("backtest_total_return")),
        },
        "paper": {
            "sharpe": _number(row.get("paper_sharpe")),
            "win_rate": _number(row.get("paper_win_rate")),
            "profit_factor": _number(row.get("paper_profit_factor")),
        },
        "runs": int(runs.get("runs") or 0),
        "recent_runs": int(runs.get("recent_runs") or 0),
        "signals": int(runs.get("signals") or 0),
        "trades": int(runs.get("trades") or 0),
        "activity_window_days": ACTIVITY_WINDOW_DAYS,
        "available_transitions": sorted(
            to for (frm, to) in TRANSITIONS if frm == status),
    }


def roster(db, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Every strategy, running ones first, with its backtest aggregate."""
    rows = db.execute(
        "SELECT id, strategy_name, status, interval, symbols, params, "
        "review_notes, promoted_at, demoted_at, last_backtest_at, "
        "backtest_sharpe, backtest_win_rate, backtest_profit_factor, "
        "backtest_total_return, paper_sharpe, paper_win_rate, "
        "paper_profit_factor FROM strategy_roster ORDER BY strategy_name",
        fetch='all') or []

    # CASE WHEN rather than FILTER: this has to run on SQLite as well.
    cutoff = (now or datetime.now()) - timedelta(days=ACTIVITY_WINDOW_DAYS)
    aggregates = db.execute(
        "SELECT strategy_name, count(*) AS runs, "
        "sum(CASE WHEN created_at > %s THEN 1 ELSE 0 END) AS recent_runs, "
        "sum(CASE WHEN created_at > %s THEN coalesce(signals_count, 0) ELSE 0 END) AS signals, "
        "sum(CASE WHEN created_at > %s THEN coalesce(total_trades, 0) ELSE 0 END) AS trades "
        "FROM backtest_runs GROUP BY strategy_name",
        (cutoff, cutoff, cutoff), fetch='all') or []
    by_name = {a["strategy_name"]: a for a in aggregates}

    order = {PAPER: 0, BACKTEST: 1, RETIRED: 2}
    view = [strategy_view(row, by_name.get(row["strategy_name"])) for row in rows]
    view.sort(key=lambda s: (order.get(s["status"], 9), s["name"]))
    return view


def backtests_for(db, name: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Recent backtest runs for one strategy."""
    rows = db.execute(
        "SELECT symbol, trading_date, interval, total_return, total_trades, "
        "signals_count, sharpe_ratio, win_rate, created_at FROM backtest_runs "
        "WHERE strategy_name = %s ORDER BY created_at DESC LIMIT %s",
        (name, int(limit)), fetch='all') or []
    return [{
        "symbol": r["symbol"],
        "trading_date": str(r["trading_date"]) if r["trading_date"] else None,
        "interval": r["interval"],
        "total_return": _number(r["total_return"]),
        "total_trades": int(r["total_trades"] or 0),
        "signals_count": int(r["signals_count"] or 0),
        "sharpe_ratio": _number(r["sharpe_ratio"]),
        "win_rate": _number(r["win_rate"]),
        "created_at": str(r["created_at"]) if r["created_at"] else None,
    } for r in rows]


def apply_transition(db, name: str, target: str, note: str,
                     now: Optional[datetime] = None) -> Dict[str, Any]:
    """Move a strategy to `target`, recording the note. Returns the new state.

    Raises LookupError when the strategy is unknown and TransitionError when
    the move is not allowed. The note is appended to review_notes rather than
    replacing it, so the history of a strategy's judgement survives.
    """
    now = now or datetime.now()
    row = db.execute(
        "SELECT id, strategy_name, status, review_notes FROM strategy_roster "
        "WHERE strategy_name = %s", (name,), fetch='one')
    if not row:
        raise LookupError(f"no strategy named '{name}'")

    current = row["status"]
    move = transition_name(current, target)

    stamp = now.strftime("%Y-%m-%d %H:%M")
    entry = f"[{stamp}] {move}: {current} -> {target}. {note}"
    history = f"{row['review_notes']}\n{entry}" if row["review_notes"] else entry

    # promoted_at and demoted_at are the roster's own record of the last move
    # in each direction; retire counts as a demotion.
    if target == PAPER:
        db.execute(
            "UPDATE strategy_roster SET status = %s, review_notes = %s, "
            "promoted_at = %s, updated_at = %s WHERE strategy_name = %s",
            (target, history, now, now, name))
    elif target == RETIRED or (current == PAPER and target == BACKTEST):
        db.execute(
            "UPDATE strategy_roster SET status = %s, review_notes = %s, "
            "demoted_at = %s, updated_at = %s WHERE strategy_name = %s",
            (target, history, now, now, name))
    else:
        db.execute(
            "UPDATE strategy_roster SET status = %s, review_notes = %s, "
            "updated_at = %s WHERE strategy_name = %s",
            (target, history, now, name))

    logger.warning("Strategy %s: %s (%s -> %s). %s", name, move, current, target, note)
    return {"name": name, "from": current, "to": target,
            "transition": move, "note": note, "at": now.isoformat()}


def create_blueprint(get_db):
    """The /api/roster routes.

    Authentication is not here: auth.install gates every route on the app,
    blueprint routes included, and a status change is a POST, so it also
    passes the Origin check.
    """
    from flask import Blueprint, jsonify, request

    bp = Blueprint("strategy_roster", __name__)

    @bp.route("/api/roster")
    def roster_route():
        return jsonify({
            "status": "success",
            "strategies": roster(get_db()),
            "statuses": list(STATUSES),
        })

    @bp.route("/api/roster/<name>/backtests")
    def backtests_route(name):
        return jsonify({"status": "success", "strategy": name,
                        "runs": backtests_for(get_db(), name)})

    @bp.route("/api/roster/<name>/status", methods=["POST"])
    def transition_route(name):
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        try:
            note = clean_note(data.get("note"))
        except ValueError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400

        try:
            result = apply_transition(get_db(), name, str(data.get("target", "")), note)
        except LookupError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 404
        except TransitionError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 409

        return jsonify({"status": "success", **result})

    return bp
