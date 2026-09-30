"""Which recommendations are safe to present as actionable.

/api/recommendations merges two sources with different freshness:

* the **screener**, whose rows come from a Finviz screen run minutes ago
* the **intraday scanner**, whose rows are computed from flat-file minute bars
  and are only as fresh as the last published session

The rule that matters: never present an expired or stale setup as actionable.
The rule that was wrong: the scanner's recency was applied to *every* merged
row, so before the open -- when the newest flat file is yesterday's -- a fresh
screener pick was withheld as "STALE (EOD flat-file)" even though no flat file
was involved in producing it.

Nothing is dropped. Withheld rows are returned separately, each carrying the
reason it was withheld, so the page can show them as dead rather than pretend
they do not exist.
"""

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Marks a row produced by the intraday flat-file scan.
INTRADAY_THEME = "intraday_setup"
INTRADAY_SOURCE = "intraday_scan"

#: Recency labels that mean "do not trade this".
STALE_LABEL = "STALE"


def is_from_intraday_scan(rec: Dict[str, Any]) -> bool:
    """True when this row came from the flat-file scan, not the screener."""
    return (rec.get("_theme") == INTRADAY_THEME
            or rec.get("_profile_source") == INTRADAY_SOURCE)


def is_stale_label(label: Any) -> bool:
    return STALE_LABEL in str(label or "")


def is_expired(rec: Dict[str, Any], now: datetime) -> bool:
    """True when the row's validity window has closed.

    A naive valid_until is not compared: the containers run on UTC and `now` is
    Eastern, so a naive value would be four hours out and would expire setups
    early. Unparseable or absent means "no window", not "expired".
    """
    raw = rec.get("valid_until")
    if not raw:
        return False
    try:
        end = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return False
    if end.tzinfo is None:
        return False
    return now > end


def mark_actionable(recs: Iterable[Dict[str, Any]], scan_recency: Any,
                    now: datetime) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into (actionable, withheld), stamping each with the reason.

    ``scan_recency`` is the intraday scan's own recency label. It applies only
    to rows from that scan; a screener row is judged by its own label.
    """
    scan_is_stale = is_stale_label(scan_recency)
    live: List[Dict[str, Any]] = []
    withheld: List[Dict[str, Any]] = []

    for rec in recs:
        expired = is_expired(rec, now)
        stale = is_stale_label(rec.get("data_recency")) or (
            scan_is_stale and is_from_intraday_scan(rec))

        rec["expired"] = expired
        rec["actionable"] = not (expired or stale)
        if expired:
            rec["withheld_reason"] = "past its validity window"
        elif stale:
            rec["withheld_reason"] = "stale data: {}".format(
                rec.get("data_recency") or scan_recency)
        else:
            rec.pop("withheld_reason", None)
            live.append(rec)
            continue
        withheld.append(rec)

    return live, withheld


def summary_message(live: List[Dict[str, Any]],
                    withheld: List[Dict[str, Any]]) -> Optional[str]:
    """One line for the page when nothing is actionable, else None."""
    if live:
        return None
    if not withheld:
        return "No screening results available yet"
    expired = sum(1 for r in withheld if r.get("expired"))
    stale = len(withheld) - expired
    parts = []
    if expired:
        parts.append(f"{expired} past their validity window")
    if stale:
        parts.append(f"{stale} on stale data")
    return "No live setups — " + " and ".join(parts) + ". Do NOT trade these."
