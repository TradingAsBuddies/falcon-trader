#!/usr/bin/env python3
"""
squawk_feed.py — BENZINGA-STYLE LIVE SQUAWK (falcon-trader #8).

A SHORT, ranked, de-duped, reverse-chronological stream of REAL breaking
market-moving headlines surfaced on the PRIVATE dashboard, the way Benzinga
Pro's news feed works — so an intraday trader sees catalysts the moment they
hit, gated to the in-play / watchlist universe and high-impact catalyst types.

SOURCE = Polygon REST news endpoint GET /v2/reference/news (verified live):
    {results[].{id, publisher.name, title, author, published_utc (ISO8601 Z),
                article_url, tickers[], description, keywords[],
                insights[]:{ticker, sentiment, sentiment_reasoning}}}

HARD CONSTRAINTS honored:
  * News is the ONE sanctioned Polygon REST use. published_utc is the publisher's
    publish-time (effectively real-time-ish), NOT the 15-min Polygon bars/quotes
    lag — so this does NOT violate the DAS-over-Polygon rule (that rule is about
    BARS/QUOTES, never news). intraday_scanner stays flat-files-only; this module
    never touches bars/quotes or flat files.
  * PRIVATE: this is a dashboard-internal surface. NOTHING here writes to Slack /
    Notion / any public channel. Massive/Polygon news must never be republished.
  * API key read via os.getenv('POLYGON_API_KEY') with MASSIVE_API_KEY fallback —
    NEVER hardcoded.
  * Carver/Bellafiore lens: a SHORT ranked de-duped stream beats a firehose.
    We gate to the in-play universe, classify + drop NOISE/sponsored PR, dedup by
    Polygon id AND a (ticker+event-type+time-bucket) fingerprint, rank by an
    explicit squawk_score, and HARD-CAP <= 30 items.
  * RESILIENCE: all network is wrapped; on 429/error we return the last-good
    cached payload with a degraded status and NEVER raise. 120s TTL cache mirrors
    intraday_scanner._CACHE / _CACHE_TTL_SECONDS.

Universe is SHARED CONTEXT with the scanner, not a forked list: prefer
FALCON_DASHBOARD_SYMBOLS, union with tickers from the latest scan / merged recs
when available, and fall back to intraday_scanner.DEFAULT_UNIVERSE /
FALCON_INTRADAY_UNIVERSE only when both are empty.
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as _urlrequest
from urllib import parse as _urlparse
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

UTC = timezone.utc
ET = ZoneInfo("America/New_York")

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

HARD_CAP = 30                      # Bellafiore: a short ranked stream, not a firehose
DEFAULT_LOOKBACK_MIN = 90          # session window; older items are muted/expired
NEW_BADGE_MAX_AGE_SEC = 120        # items younger than this render a NEW badge
PER_TICKER_LIMIT = 5               # small page per ticker — low-latency, low-cost
UNIVERSE_CAP = 40                  # bound the number of tickers we query
SCORE_FLOOR = 0.20                 # drop items ranked below this
HTTP_TIMEOUT_SEC = 6

# TTL cache for the lazy in-process path (mirrors intraday_scanner).
_CACHE_TTL_SECONDS = 120
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# ---------------------------------------------------------------------------
# Catalyst classification — keyword + publisher rules over the verbatim title.
# Order matters: the FIRST matching tier wins (tier-1 catalysts checked first).
# ---------------------------------------------------------------------------

# tier -> weight. TIER-1 catalysts on an in-play ticker are tagged HIGH.
TIER_WEIGHT = {1: 3.0, 2: 1.6, 3: 1.0}
CATALYST_TIER = {
    "HALT": 1, "M&A": 1, "FDA": 1, "GUIDANCE": 1,
    "EARNINGS": 2, "OFFERING": 2, "SEC-8K": 2, "RATING": 2,
    "EXEC": 3, "MACRO": 3,
    "NOISE": 3,
}

# Regex keyword rules per catalyst type (lowercased title match).
_CATALYST_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    ("HALT", re.compile(
        r"\b(halt(ed|s)?|trading halt|circuit breaker|resume(d|s)? trading|"
        r"volatility pause)\b")),
    ("M&A", re.compile(
        r"\b(acquir(e|es|ed|ing|ition)|merge(r|s|d)?|to be acquired|takeover|"
        r"buyout|tender offer|definitive agreement|to acquire|stake in|"
        r"go private|going private|all-cash)\b")),
    ("FDA", re.compile(
        r"\b(fda|phase\s?(1|2|3|i{1,3})|clinical (trial|data)|topline|"
        r"breakthrough therapy|nda|bla|pdufa|crl|complete response|"
        r"approval|approves?|emergency use|510\(k\)|primary endpoint)\b")),
    ("GUIDANCE", re.compile(
        r"\b(guidance|raises? (full[- ]?year|fy|q[1-4]|outlook|forecast)|"
        r"cuts? (guidance|outlook|forecast)|lowers? (guidance|outlook|forecast)|"
        r"reaffirms?|preliminary results|warns?|profit warning|"
        r"above|below) (consensus|estimates|expectations)\b")),
    ("EARNINGS", re.compile(
        r"\b(q[1-4]|first|second|third|fourth)[- ]?(quarter|qtr)\b|"
        r"\b(earnings|eps|revenue|reports? (results|earnings)|beats?|misses?|"
        r"tops? (estimates|views)|posts? (a )?(loss|profit)|"
        r"results? (for|exceed))\b")),
    ("OFFERING", re.compile(
        r"\b(offering|prices? \$?\d|priced? (public )?offering|"
        r"registered direct|atm|at[- ]the[- ]market|shelf|s-1|s-3|"
        r"dilut(e|ion|ive)|private placement|convertible notes?|"
        r"warrants?|reverse split|stock split|capital raise|raises? \$)\b")),
    ("SEC-8K", re.compile(
        r"\b(8-?k|10-?k|10-?q|sec (filing|investigation|subpoena|charges)|"
        r"form 4|insider (buy|sell|sale|purchase)|13d|13g|"
        r"restatement|going concern|delist)\b")),
    ("RATING", re.compile(
        r"\b(upgrade(s|d)?|downgrade(s|d)?|initiate(s|d)? (coverage)?|"
        r"price target|raises? (pt|target)|cuts? (pt|target)|"
        r"reiterat(e|es|ed)|overweight|underweight|outperform|underperform|"
        r"buy rating|sell rating|neutral rating|analyst)\b")),
    ("EXEC", re.compile(
        r"\b(ceo|cfo|coo|president|chief executive|steps? down|resign(s|ed|ation)?|"
        r"appoints?|names? (new )?(ceo|cfo)|departure|board of directors|"
        r"management change)\b")),
    ("MACRO", re.compile(
        r"\b(fed|fomc|interest rate|cpi|ppi|inflation|jobs report|"
        r"nonfarm|unemployment|gdp|treasury|powell|tariff(s)?|"
        r"jerome powell)\b")),
]

# Routine-PR / sponsored noise — classified NOISE and dropped from the stream.
_NOISE_RULES = re.compile(
    r"\b(sponsored|webinar|conference call (details|reminder)|to (present|webcast)|"
    r"to participate in|announces? (date|time) of|to host|to attend|"
    r"investor (day|conference|presentation)|annual meeting|"
    r"why .* (could|might|may) (be a|move)|things to know|"
    r"\d+ (stocks|reasons)|here'?s why|motley fool|zacks rank|"
    r"dividend (declaration|announcement)|declares (quarterly )?dividend|"
    r"ex-dividend|appoints? .* to (its )?board|"
    r"recognized|award(ed)?|certified|partnership with|collaborat)\b",
    re.IGNORECASE,
)

# Publisher quality tiers (lowercased substring match on publisher.name).
# primary newswire / exchange = high; aggregator = med; PR wire = low.
_PUBLISHER_QUALITY = {
    "high": ["dow jones", "reuters", "bloomberg", "the wall street journal",
             "marketwatch", "barron", "cnbc", "associated press", "financial times",
             "nasdaq", "nyse"],
    "med":  ["benzinga", "seeking alpha", "investing.com", "thefly", "the fly",
             "yahoo", "investor", "247wallst", "schaeffer"],
    "low":  ["globenewswire", "business wire", "businesswire", "pr newswire",
             "prnewswire", "accesswire", "newsfile", "globe newswire",
             "ein presswire", "ace news", "motley fool", "zacks"],
}
_PUBLISHER_WEIGHT = {"high": 1.4, "med": 1.1, "low": 0.85, "unknown": 1.0}


# ---------------------------------------------------------------------------
# Universe resolution — SHARED CONTEXT with the scanner (no forked list)
# ---------------------------------------------------------------------------

def resolve_universe(extra_tickers: Optional[List[str]] = None) -> List[str]:
    """Resolve the in-play universe the squawk filters to.

    Prefer FALCON_DASHBOARD_SYMBOLS (today's real watchlist), union with any
    tickers from the latest scan / merged recs (passed in via extra_tickers),
    and fall back to intraday_scanner.DEFAULT_UNIVERSE / FALCON_INTRADAY_UNIVERSE
    ONLY when both are empty. Bounded set — never the full market.
    """
    syms: List[str] = []

    dash = os.getenv("FALCON_DASHBOARD_SYMBOLS", "")
    if dash:
        syms.extend(s.strip().upper() for s in dash.split(",") if s.strip())

    if extra_tickers:
        syms.extend(str(s).strip().upper() for s in extra_tickers if str(s).strip())

    if not syms:
        # Fallback: reuse the scanner's universe / env, never a private copy.
        intra_env = os.getenv("FALCON_INTRADAY_UNIVERSE", "")
        if intra_env:
            syms.extend(s.strip().upper() for s in intra_env.split(",") if s.strip())
        else:
            try:
                from falcon_trader import intraday_scanner
                syms.extend(str(s).strip().upper()
                            for s in intraday_scanner.DEFAULT_UNIVERSE if s)
            except Exception:
                pass

    # De-dup preserving order, then bound the count.
    seen = set()
    out: List[str] = []
    for s in syms:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:UNIVERSE_CAP]


def _latest_scan_tickers() -> List[str]:
    """Best-effort: pull tickers from the latest intraday scan so the squawk
    news-enriches exactly the names that have live setups. Never raises."""
    try:
        from falcon_trader import intraday_scanner
        scan = intraday_scanner.scan_intraday_setups()
        return [str(s.get("ticker", "")).upper()
                for s in scan.get("setups", []) if s.get("ticker")]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Classification / scoring helpers
# ---------------------------------------------------------------------------

def classify_catalyst(title: str, publisher: str = "") -> str:
    """Classify catalyst_type from keyword rules over the verbatim title.

    Returns one of {EARNINGS, GUIDANCE, M&A, FDA, HALT, RATING, OFFERING,
    SEC-8K, EXEC, MACRO, NOISE}. NOISE wins only when no real catalyst matches.
    """
    t = (title or "").lower()
    for ctype, pat in _CATALYST_RULES:
        if pat.search(t):
            return ctype
    if _NOISE_RULES.search(title or ""):
        return "NOISE"
    return "NOISE"


def _publisher_quality(publisher: str) -> Tuple[str, float]:
    p = (publisher or "").lower()
    for tier in ("high", "med", "low"):
        for name in _PUBLISHER_QUALITY[tier]:
            if name in p:
                return tier, _PUBLISHER_WEIGHT[tier]
    return "unknown", _PUBLISHER_WEIGHT["unknown"]


def _parse_utc(published_utc: str) -> Optional[datetime]:
    """Parse Polygon's ISO8601 Z timestamp into an aware UTC datetime."""
    if not published_utc:
        return None
    try:
        s = published_utc.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _render_age(dt: Optional[datetime], now: datetime) -> str:
    """Human age, e.g. 'NEW', '0m12s ago', '47m ago', '2h3m ago'."""
    if dt is None:
        return "unknown"
    secs = int((now - dt).total_seconds())
    if secs < 0:
        secs = 0
    if secs < NEW_BADGE_MAX_AGE_SEC:
        return "NEW"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s ago"
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    return f"{hrs}h{mins}m ago"


def _recency_decay(dt: Optional[datetime], now: datetime,
                   lookback_min: int) -> float:
    """Linear-ish decay from 1.0 (just published) toward ~0.2 at the window edge."""
    if dt is None:
        return 0.3
    age_min = max(0.0, (now - dt).total_seconds() / 60.0)
    if age_min >= lookback_min:
        return 0.1
    return max(0.1, 1.0 - 0.8 * (age_min / float(lookback_min)))


def _select_sentiment(insights: List[Dict[str, Any]],
                      ticker: str) -> Tuple[str, str]:
    """Select the insights[] entry where entry.ticker == ticker (NOT insights[0]).

    An article tagging multiple tickers carries a per-ticker insight; attributing
    insights[0] would mislabel the queried ticker's sentiment. Returns
    (sentiment, reasoning); ('no sentiment', '') when no matching insight exists.
    """
    if not insights:
        return "no sentiment", ""
    for ins in insights:
        if str(ins.get("ticker", "")).upper() == ticker.upper():
            return (str(ins.get("sentiment", "no sentiment") or "no sentiment"),
                    str(ins.get("sentiment_reasoning", "") or ""))
    return "no sentiment", ""


def _fingerprint(ticker: str, catalyst_type: str,
                 dt: Optional[datetime]) -> str:
    """Normalized (ticker + event-type + time-bucket) dedup key.

    Collapses the same event reported by N publishers within a 10-min bucket
    into one cluster so the stream isn't a triplicate firehose.
    """
    if dt is not None:
        bucket = int(dt.timestamp() // 600)  # 10-minute buckets
    else:
        bucket = 0
    return f"{ticker.upper()}|{catalyst_type}|{bucket}"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _api_key() -> Optional[str]:
    return os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")


def _fetch_ticker_news(ticker: str, gte_iso: str, api_key: str,
                       limit: int = PER_TICKER_LIMIT) -> List[Dict[str, Any]]:
    """GET /v2/reference/news for one ticker. Returns results[] or [] on error.
    Never raises — resilience is the caller's degraded-status contract."""
    params = {
        "ticker": ticker,
        "order": "desc",
        "sort": "published_utc",
        "published_utc.gte": gte_iso,
        "limit": str(limit),
        "apiKey": api_key,
    }
    url = f"{POLYGON_NEWS_URL}?{_urlparse.urlencode(params)}"
    req = _urlrequest.Request(url, headers={"User-Agent": "falcon-squawk/1.0"})
    try:
        with _urlrequest.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("status") not in ("OK", "DELAYED", None):
            return []
        return payload.get("results", []) or []
    except HTTPError as e:
        # 429 (rate limit) or other HTTP error — signal up via exception sentinel.
        raise
    except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
        logger.warning("squawk: news fetch failed for %s: %s", ticker, e)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_squawk(universe: Optional[List[str]] = None,
                 lookback_min: int = DEFAULT_LOOKBACK_MIN,
                 use_cache: bool = True) -> Dict[str, Any]:
    """Fetch, classify, dedup, rank and cap the live squawk stream.

    Returns:
        {
          "status": "success" | "no_data" | "degraded" | "error",
          "items": [row, ...],          # <= HARD_CAP, reverse-chronological-ish
          "universe": [tickers...],
          "fetched_at": ISO8601 Z,
          "data_recency": "LIVE (Polygon news, publisher-time)",
          "source": "polygon_news",
          "message": optional str,
        }
    NEVER raises. On 429/error returns last-good cached payload (degraded) or an
    empty error envelope — it never crashes the dashboard.
    """
    if universe is None:
        universe = resolve_universe(extra_tickers=_latest_scan_tickers())
    universe = [s.strip().upper() for s in (universe or []) if s and s.strip()]

    now = datetime.now(UTC)
    cache_key = f"{','.join(universe)}|{lookback_min}"

    if use_cache:
        hit = _CACHE.get(cache_key)
        if hit and (now.timestamp() - hit[0]) < _CACHE_TTL_SECONDS:
            return hit[1]

    api_key = _api_key()
    if not api_key:
        return {
            "status": "error",
            "items": [],
            "universe": universe,
            "fetched_at": now.isoformat().replace("+00:00", "Z"),
            "data_recency": "UNAVAILABLE (POLYGON_API_KEY not set)",
            "source": "polygon_news",
            "message": "No Polygon API key (POLYGON_API_KEY / MASSIVE_API_KEY).",
        }

    if not universe:
        return {
            "status": "no_data",
            "items": [],
            "universe": [],
            "fetched_at": now.isoformat().replace("+00:00", "Z"),
            "data_recency": "LIVE (Polygon news, publisher-time)",
            "source": "polygon_news",
            "message": "No in-play universe resolved.",
        }

    gte_iso = (now - timedelta(minutes=lookback_min)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    # ---- fetch per ticker, mapping each article to the QUERIED ticker --------
    raw: List[Tuple[str, Dict[str, Any]]] = []   # (queried_ticker, result)
    rate_limited = False
    errors = 0
    for tk in universe:
        try:
            results = _fetch_ticker_news(tk, gte_iso, api_key)
        except HTTPError as e:
            if e.code == 429:
                rate_limited = True
                break
            errors += 1
            continue
        for r in results:
            raw.append((tk, r))

    # If we got nothing AND hit a rate-limit/error, degrade to last-good cache.
    if (rate_limited or errors) and not raw:
        hit = _CACHE.get(cache_key)
        if hit:
            cached = dict(hit[1])
            cached["status"] = "degraded"
            cached["message"] = ("Polygon rate-limited (429); serving last-good "
                                 "cached stream.") if rate_limited else \
                                ("Polygon errored; serving last-good cached stream.")
            cached["fetched_at"] = now.isoformat().replace("+00:00", "Z")
            return cached
        # No cache to fall back on — empty degraded envelope, never raise.
        return {
            "status": "degraded",
            "items": [],
            "universe": universe,
            "fetched_at": now.isoformat().replace("+00:00", "Z"),
            "data_recency": "DEGRADED (Polygon rate-limited / error, no cache)",
            "source": "polygon_news",
            "message": "Polygon 429/error and no cached stream available.",
        }

    # ---- classify + build candidate rows ------------------------------------
    candidates: List[Dict[str, Any]] = []
    for queried_ticker, r in raw:
        title = str(r.get("title", "") or "")
        if not title:
            continue
        publisher = str((r.get("publisher") or {}).get("name", "") or "")
        catalyst_type = classify_catalyst(title, publisher)
        # NOISE / sponsored PR never appears in the default in-play stream.
        if catalyst_type == "NOISE":
            continue

        dt = _parse_utc(str(r.get("published_utc", "") or ""))
        # EXPIRY: mute items older than the session window (no NEW badge).
        if dt is not None and (now - dt) > timedelta(minutes=lookback_min):
            continue

        tickers = [str(t).upper() for t in (r.get("tickers") or [])]
        in_play = queried_ticker in [s.upper() for s in universe]
        sentiment, reasoning = _select_sentiment(r.get("insights") or [],
                                                 queried_ticker)

        tier_num = CATALYST_TIER.get(catalyst_type, 3)
        tier_weight = TIER_WEIGHT.get(tier_num, 1.0)
        pub_tier, pub_weight = _publisher_quality(publisher)
        decay = _recency_decay(dt, now, lookback_min)
        in_play_bonus = 1.3 if in_play else 1.0
        squawk_score = round(
            tier_weight * pub_weight * decay * in_play_bonus, 4)

        tier_label = "HIGH" if (tier_num == 1 and in_play) else (
            "MED" if tier_num == 2 else "LOW")

        candidates.append({
            "id": str(r.get("id", "")),
            "queried_ticker": queried_ticker,
            "tickers": tickers or [queried_ticker],
            "title": title,                       # verbatim, never paraphrased
            "published_utc": str(r.get("published_utc", "") or ""),
            "_dt": dt,
            "age": _render_age(dt, now),
            "publisher": publisher,
            "publisher_tier": pub_tier,
            "article_url": str(r.get("article_url", "") or ""),
            "catalyst_type": catalyst_type,
            "tier": tier_label,
            "sentiment": sentiment,
            "sentiment_source": "model-derived (Polygon insights)",
            "sentiment_reasoning": reasoning,
            "squawk_score": squawk_score,
            "in_play": in_play,
            "source": "polygon_news",
        })

    # ---- DEDUP: collapse by Polygon id AND fingerprint, count sources -------
    by_key: Dict[str, Dict[str, Any]] = {}
    seen_ids: Dict[str, str] = {}   # polygon id -> cluster key
    for c in candidates:
        fp = _fingerprint(c["queried_ticker"], c["catalyst_type"], c["_dt"])
        # Same Polygon id already clustered -> bump source_count.
        pid = c["id"]
        key = seen_ids.get(pid) or fp
        existing = by_key.get(key)
        if existing is None:
            c["source_count"] = 1
            by_key[key] = c
            if pid:
                seen_ids[pid] = key
        else:
            existing["source_count"] = existing.get("source_count", 1) + 1
            # Keep the highest-scoring / freshest representative of the cluster.
            if (c["squawk_score"] > existing["squawk_score"] or
                    (c["_dt"] and existing["_dt"] and c["_dt"] > existing["_dt"])):
                c["source_count"] = existing["source_count"]
                by_key[key] = c
                if pid:
                    seen_ids[pid] = key

    rows = list(by_key.values())

    # ---- score floor, then sort: score desc, then strictly reverse-chrono ---
    rows = [r for r in rows if r["squawk_score"] >= SCORE_FLOOR]
    rows.sort(
        key=lambda x: (
            x["squawk_score"],
            x["_dt"].timestamp() if x["_dt"] else 0.0,
        ),
        reverse=True,
    )

    # HARD CAP <= 30.
    rows = rows[:HARD_CAP]

    # Strip the internal datetime before returning (not JSON-serializable nicely
    # and not part of the contract — published_utc + age carry freshness).
    for r in rows:
        r.pop("_dt", None)

    result = {
        "status": "success" if rows else "no_data",
        "items": rows,
        "universe": universe,
        "fetched_at": now.isoformat().replace("+00:00", "Z"),
        "data_recency": "LIVE (Polygon news, publisher-time)",
        "source": "polygon_news",
        "lookback_min": lookback_min,
        "count": len(rows),
    }
    if not rows:
        result["message"] = "No in-play catalyst headlines in the session window."

    if use_cache:
        _CACHE[cache_key] = (now.timestamp(), result)
    return result


# ---------------------------------------------------------------------------
# CLI / one-shot verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint
    out = fetch_squawk(use_cache=False)
    print("status:", out["status"], "| count:", out.get("count"),
          "| universe:", out["universe"])
    for it in out["items"][:10]:
        print(f"\n[{it['age']}] {it['queried_ticker']} "
              f"({it['catalyst_type']}/{it['tier']}) score={it['squawk_score']} "
              f"x{it['source_count']} sent={it['sentiment']}")
        print(f"   {it['title']}")
        print(f"   {it['publisher']} | {it['published_utc']} | {it['article_url']}")
