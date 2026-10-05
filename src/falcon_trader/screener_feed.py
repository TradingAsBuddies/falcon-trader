"""The screener's recommendations, read from the database.

The orchestrator used to read `screened_stocks.json` as a *relative* path. The
screener writes that file into its own container's volume; the orchestrator runs
with the working directory /app, where no such file exists. So every cycle
printed "Screener file not found" and the orchestrator placed no trades at all
while reporting itself healthy -- 24 days with an empty book.

A file on a per-container volume is the wrong channel between two containers.
Both already share PostgreSQL, and the screener writes every run to
profile_runs, which is where the dashboard reads them from.

Timestamps: profile_runs.run_timestamp is a naive column holding UTC (the
containers run on UTC), so the cutoff here is built in UTC and compared naive.
It is bound as a parameter rather than written as `now() - interval`, which is
PostgreSQL-only.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: How old a screen may be and still be acted on. A morning screen runs at
#: 04:00 ET and must stay usable for that session; beyond a day it is not a
#: view of today's market.
DEFAULT_MAX_AGE = timedelta(hours=24)


def utc_now_naive() -> datetime:
    """Now, as the naive UTC the database columns hold."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _recommendations_of(run_data: Any) -> List[Dict[str, Any]]:
    """The recommendations list inside one profile_runs.run_data value."""
    if not isinstance(run_data, dict):
        return []
    recs = run_data.get("recommendations")
    return [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []


def _confidence(rec: Dict[str, Any]) -> float:
    try:
        return float(rec.get("confidence_score") or 0)
    except (TypeError, ValueError):
        return 0.0


def latest_recommendations(
    db,
    now: Optional[datetime] = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> Tuple[List[Dict[str, Any]], Optional[datetime]]:
    """Recommendations from every screener run inside the window.

    Returns (recommendations, newest_run_timestamp). One ticker appears once,
    keeping the highest confidence, because several profiles can surface the
    same name and the orchestrator should consider it once.
    """
    now = now or utc_now_naive()
    cutoff = now - max_age

    rows = db.execute(
        "SELECT run_timestamp, run_data FROM profile_runs "
        "WHERE run_timestamp > %s ORDER BY run_timestamp DESC",
        (cutoff,),
        fetch='all',
    ) or []

    by_ticker: Dict[str, Dict[str, Any]] = {}
    newest: Optional[datetime] = None

    for row in rows:
        stamp = row['run_timestamp']
        if stamp and (newest is None or stamp > newest):
            newest = stamp
        for rec in _recommendations_of(row['run_data']):
            ticker = str(rec.get('ticker') or rec.get('symbol') or '').strip().upper()
            if not ticker:
                continue
            kept = by_ticker.get(ticker)
            if kept is None or _confidence(rec) > _confidence(kept):
                enriched = dict(rec)
                enriched['ticker'] = ticker
                enriched.setdefault('_run_timestamp', stamp.isoformat() if stamp else None)
                by_ticker[ticker] = enriched

    recommendations = sorted(by_ticker.values(), key=_confidence, reverse=True)
    logger.info("[FEED] %d recommendation(s) from %d run(s) since %s",
                len(recommendations), len(rows), cutoff)
    return recommendations, newest
