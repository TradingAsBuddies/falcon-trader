"""
Timezone utilities for Falcon Trader

All trading logic operates in US/Eastern (market time). This module provides
centralized helpers to ensure every timestamp comparison uses the same
timezone, preventing the class of bugs where server time is compared
against Eastern Time strategy parameters or screener data.

Convention for US equities:
  - "Market time" means US/Eastern and is the default for this system.
  - Naive datetimes (no tzinfo) are assumed to already be market time (ET).
  - Only label a source as UTC when its documentation explicitly says so.

Known source timezones:
  1. Polygon.io — epoch milliseconds (tz-agnostic); daily bar dates are
     market-calendar dates (ET). Treat as ET unless docs say otherwise.
  2. Database — we write ET-aware ISO strings via now_et().
  3. Screener — outputs US/Eastern (ISO 8601 with offset).
  4. Strategy parameters (trade windows, max-hold) — Eastern.
  5. All datetime.now() calls in the trader MUST go through now_et().
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def now_et() -> datetime:
    """Return the current time as a timezone-aware datetime in US/Eastern."""
    return datetime.now(ET)


def ensure_et(dt: datetime) -> datetime:
    """
    Normalize *dt* to US/Eastern (market time).

    - If *dt* is naive (no tzinfo), it is assumed to already be market
      time (ET) and localized in place.
    - If *dt* is already timezone-aware, it is converted to Eastern.
    """
    if dt.tzinfo is None:
        # Naive → assume market time (ET)
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def ensure_et_from_utc(dt: datetime) -> datetime:
    """
    Convert a datetime that is known to be UTC into US/Eastern.

    Use this only for sources whose documentation explicitly states UTC
    (e.g. a raw epoch conversion). For everything else use ensure_et().
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def parse_timestamp(ts_string: str) -> datetime:
    """
    Parse a timestamp string and return a timezone-aware Eastern datetime.

    Handles:
      - ISO 8601 with offset  ("2026-01-05T18:11:24.043177-05:00")
      - ISO 8601 naive         ("2026-01-08T05:01:23")
      - Simple format          ("2026-01-08 05:01:23")

    Naive strings are assumed to be market time (ET).
    """
    from dateutil import parser as dateutil_parser

    dt = dateutil_parser.parse(ts_string)
    return ensure_et(dt)
