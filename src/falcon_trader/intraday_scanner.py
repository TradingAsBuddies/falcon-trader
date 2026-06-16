#!/usr/bin/env python3
"""
intraday_scanner.py — LIVE INTRADAY SETUP SCANNER (falcon-trader #6).

Problem: /api/recommendations returned {"status":"no_data"} during the trading
day because nothing produced INTRADAY/minute-bar setups — the endpoint only read
a Finviz/AI SWING screener's profile_runs, which is not even scheduled on this host.

This module surfaces REAL, RANKED, STRUCTURED day-trade setups computed from
1-MINUTE FLAT-FILE bars (Polygon flat files via Massive S3), reusing the existing
SMB strategy signal logic (bella_fade / offside_scalp / microstructure_momentum)
plus two triggers codified here (ORB and VWAP-reclaim).

FRESHNESS TIERS (falcon-trader #9 — a SUGGESTION/DISCOVERY surface, not execution):
  This is a candidate-discovery surface, so it prefers the FRESHEST AVAILABLE bar
  source and labels every payload + setup HONESTLY to the tier actually used:
      DAS live (RTH top tier)        reserved hook — DEFERRED (full DAS wiring later)
      DELAYED ~15m (Polygon REST)    RTH + POLYGON_API_KEY set + non-empty result,
                                     reached by REUSING DataFeed(source="polygon")
      STALE (EOD flat-file, as of D) pre-open / key unset / non-RTH -> flat files
      DEGRADED (Polygon -> flat-file) RTH but Polygon errored/empty -> flat fallback
  HONEST LABELING IS NON-NEGOTIABLE: a df sourced from flat files NEVER carries a
  DELAYED or LIVE tag; Polygon bars are exactly "DELAYED ~15m (Polygon REST)". The
  Polygon REST allowance is for SUGGESTION ONLY — execution still routes through DAS.
  Flat files remain the fallback/backfill tier; disabling Polygon is purely additive
  (behavior identical to the pre-#9 STALE-only scanner). This module NEVER emits a
  LIVE label — that is reserved for real DAS wiring.
  * Carver/Bellafiore lens: rank by an EDGE PROXY (R:R x confidence), apply a cost
    speed-limit + liquidity gates BEFORE ranking, weight trend setups above
    counter-trend fades, dedup by ticker keep-highest-edge, HARD CAP 5 setups.

DEPLOYMENT NOTE (boto3 placement):
  boto3 lives in the falcon-core image (extras: backtesting/data-sync), NOT in the
  falcon-trader/dashboard image. So flat files can only be READ from an image that
  has boto3. This module is therefore designed to run two ways:
    (1) IN-PROCESS (lazy, inside get_recommendations): works wherever boto3 +
        DataFeed flat-files are available; gracefully returns an empty-setups
        result (clearly labeled) where they are not — it never crashes the endpoint.
    (2) CLI / one-shot (run inside the falcon-core image, which HAS boto3):
        `python3 -m falcon_trader.intraday_scanner` runs the scan and PERSISTS the
        setups into the existing profile_runs store via ProfileManager.log_profile_run
        under a dedicated "Intraday Scanner" profile. The dashboard endpoint then
        reads them through its existing ProfileManager merge loop — zero new tables,
        and /api/recommendations/history works for free.

Strategy modules (falcon-strategies/core/*.py) are NOT pip-installed in either
image; they are plain modules that depend only on falcon_core + pandas/numpy. We
load them via importlib from a configurable directory (FALCON_STRATEGIES_CORE_DIR,
default falcon-strategies/core), so the strategy classes are reused verbatim — we
do NOT reinvent their indicators or signal semantics.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import importlib.util
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Configuration / tunables
# ---------------------------------------------------------------------------

# Bounded in-play universe — a gappers / rel-vol seed list. We do NOT scan the
# full symbol universe (overtrading + cost red flag). ~30 liquid, frequently
# in-play names. Override via FALCON_INTRADAY_UNIVERSE (comma-separated).
DEFAULT_UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "NFLX",
    "AVGO", "MU", "SMCI", "PLTR", "COIN", "MARA", "RIOT", "SOFI", "AFRM",
    "BABA", "INTC", "F", "BAC", "GME", "AMC", "DKNG", "SHOP", "UBER", "CVNA",
    "ARM", "DELL",
]

# Liquidity / price gates (applied BEFORE ranking).
MIN_PRICE = 3.0                 # avoid sub-$3 names (spread + halt risk)
MIN_DOLLAR_VOLUME = 5_000_000   # session $-volume floor (price * volume summed)
MIN_RR = 1.5                    # no setup ranked below 1.5:1 reward:risk

# Cost speed-limit. The intraday backtest tool (falcon-strategies/tools/
# backtest_intraday.py:23) documents a slippage model as a follow-up but exposes
# no importable constant, so we define a documented default here.
# TODO(unify): replace with the shared slippage model once backtest_intraday.py
# exports one (issue #1 follow-up).
SLIPPAGE_BPS = 8.0              # one-way modeled slippage in basis points
SPREAD_BPS = 5.0               # modeled half-spread in basis points (one-way)
# Round-trip cost (enter + exit) in bps: 2 * (slippage + spread).
ROUND_TRIP_COST_BPS = 2.0 * (SLIPPAGE_BPS + SPREAD_BPS)
# Expected target move must clear round-trip cost by this multiple (wide margin).
COST_SPEED_LIMIT_MULT = 3.0

# Trend setups (gap-and-go / ORB / VWAP reclaim) are weighted above counter-trend
# fades, per the decisive "structured in-play, not a raw volatility list" finding.
TREND_SETUPS = {"orb", "vwap_reclaim", "gap_and_go", "microstructure_momentum"}
FADE_SETUPS = {"bella_fade", "offside_scalp"}
TREND_WEIGHT = 1.15
FADE_WEIGHT = 0.90

HARD_CAP = 5                   # Bellafiore: a short ranked shortlist, not a firehose
SUPPRESS_LAST_MINUTES = 30     # do not emit new setups in the last 30 min of RTH

# TTL cache for the in-process lazy path.
_CACHE_TTL_SECONDS = 120
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Where the falcon-strategies core modules live (plain .py, not pip-installed).
def _strategies_core_dir() -> str:
    env = os.getenv("FALCON_STRATEGIES_CORE_DIR")
    if env and os.path.isdir(env):
        return env
    # Common locations: bind-mounted into the image, or sibling repo on host.
    candidates = [
        "/opt/falcon-strategies/core",
        "/usr/local/lib/falcon-strategies/core",
        os.path.expanduser("~/src/TradingAsBuddies/falcon-strategies/core"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "falcon-strategies", "core"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return candidates[-1]


def _load_strategy_class(module_name: str, class_name: str):
    """Import a strategy module by file path and return its class.

    Reuses the strategy verbatim (no reinvention). Returns None on failure so the
    scanner degrades gracefully rather than crashing the endpoint.
    """
    core_dir = _strategies_core_dir()
    path = os.path.join(core_dir, f"{module_name}.py")
    if not os.path.exists(path):
        logger.warning("strategy module not found: %s", path)
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_falcon_intraday_{module_name}", path)
        mod = importlib.util.module_from_spec(spec)
        # Make sibling helpers importable if any.
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, class_name, None)
    except Exception as e:
        logger.warning("failed to load %s.%s: %s", module_name, class_name, e)
        return None


# ---------------------------------------------------------------------------
# Flat-file minute bars (FLAT FILES ONLY — never REST)
# ---------------------------------------------------------------------------

def _make_datafeed():
    """Construct a falcon-core DataFeed. Returns None if unavailable."""
    try:
        from falcon_core.backtesting.data_feed import DataFeed
    except Exception as e:
        logger.warning("falcon_core DataFeed unimportable: %s", e)
        return None
    try:
        cache_dir = os.getenv("FALCON_CACHE_DIR") or None
        feed = DataFeed(cache_dir=cache_dir)
    except Exception as e:
        logger.warning("DataFeed init failed: %s", e)
        return None
    # If flat files are not wired (e.g. boto3 missing in this image), bail — we
    # NEVER fall back to the REST API for scan input.
    if getattr(feed, "flatfiles", None) is None:
        logger.warning(
            "Flat Files client unavailable in this process (boto3 missing?) — "
            "cannot scan in-process. Run the scanner CLI in the falcon-core image."
        )
        return None
    return feed


def resolve_session_date(session_date: Optional[str] = None) -> Optional[str]:
    """Resolve the most-recent available flat-file session (YYYY-MM-DD), <= today.

    Uses FlatFilesClient.list_available_dates(year, month). Returns None if the
    flat-file client is unavailable.
    """
    if session_date:
        return session_date
    try:
        from falcon_core.backtesting.flatfiles_client import FlatFilesClient
    except Exception as e:
        logger.warning("FlatFilesClient unimportable: %s", e)
        return None
    try:
        client = FlatFilesClient(
            access_key=os.getenv("MASSIVE_ACCESS_KEY"),
            secret_key=os.getenv("MASSIVE_SECRET_KEY"),
            cache_dir=os.getenv("FALCON_CACHE_DIR") or None,
        )
    except Exception as e:
        logger.warning("FlatFilesClient init failed (boto3 missing?): %s", e)
        return None

    today = date.today()
    # Look back across this month and previous month for the newest session.
    for back in range(0, 2):
        ym = today.replace(day=1) - timedelta(days=back * 28)
        try:
            dates = client.list_available_dates(ym.year, ym.month)
        except Exception as e:
            logger.warning("list_available_dates(%s,%s) failed: %s", ym.year, ym.month, e)
            dates = []
        usable = [d for d in dates if d <= today.isoformat()]
        if usable:
            return sorted(usable)[-1]
    return None


def _flatfile_minute_df(feed, symbol: str, session_date: str):
    """Pull 1-minute bars for one session via flat files ONLY. None on failure.

    This is the FALLBACK / BACKFILL tier. It is the original scan input and is
    preserved verbatim — disabling the Polygon tier yields behavior identical to
    the pre-#9 scanner.
    """
    try:
        df = feed.get_historical_data(
            symbol,
            session_date,
            session_date,
            interval="1m",
            source="flatfiles",        # explicit — flat-file backfill tier
            market_hours_only=True,
        )
    except Exception as e:
        logger.debug("no flat-file bars for %s %s: %s", symbol, session_date, e)
        return None
    if df is None or df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    return df


def _polygon_minute_df(feed, symbol: str, today_iso: str):
    """Try the Polygon REST tier (DELAYED ~15m). Returns df or None.

    Reuses DataFeed source='polygon' / _try_polygon — NO new REST client. Bars
    are ~15-min DELAYED (the API's own word) and must be labeled accordingly;
    NEVER as LIVE. Returns None on empty/error so the caller can degrade.
    """
    try:
        df = feed.get_historical_data(
            symbol,
            today_iso,
            today_iso,
            interval="1m",
            source="polygon",
            market_hours_only=True,
        )
    except Exception as e:
        logger.debug("no polygon bars for %s %s: %s", symbol, today_iso, e)
        return None
    if df is None or df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    return df


def _last_bar_iso_et(df) -> Optional[str]:
    """tz-aware America/New_York ISO-8601 of the last bar, or None."""
    try:
        bar_ts = df.index[-1]
    except Exception:
        return None
    try:
        ts = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
    except Exception:
        ts = bar_ts
    if hasattr(ts, "tzinfo"):
        if ts.tzinfo is None:
            # Polygon/flat bars are UTC; localize then convert. Flat-file bars are
            # already tz-aware after the polygon_client fix; this is defensive.
            ts = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
        else:
            ts = ts.astimezone(ET)
        return ts.isoformat()
    return str(bar_ts)


def _rth_in_progress(open_et: datetime, close_et: datetime,
                     min_after_open: int = 15) -> bool:
    """True if 'now' is within RTH and at least ~min_after_open minutes past the
    open (>= ~09:45 ET) and before the close. Pre-open / post-close -> False.

    TEST-ONLY OVERRIDE: when FALCON_SCANNER_FORCE_RTH_DATE is set (YYYY-MM-DD),
    this returns True so the off-hours acceptance harness can exercise the Polygon
    DELAYED tier against a known-good past RTH session. The env is UNSET in
    production (dashboard.container / intraday-scan.container), so the live gate is
    byte-for-byte unchanged: DELAYED stays gated on real now() in RTH + non-empty
    Polygon. This NEVER relabels a tier — it only re-opens the wall-clock gate for
    a deliberate, opt-in proof run.
    """
    if os.getenv("FALCON_SCANNER_FORCE_RTH_DATE"):
        return True
    now_et = datetime.now(tz=ET)
    earliest = open_et + timedelta(minutes=min_after_open)
    return earliest <= now_et <= close_et


def _get_minute_df(feed, symbol: str, session_date: str,
                   open_et: Optional[datetime] = None,
                   close_et: Optional[datetime] = None,
                   allow_polygon: bool = True):
    """Tier-aware minute-bar fetch. Returns (df, tier_recency, last_bar_ts).

    Freshness hierarchy (prefer the freshest AVAILABLE):
        LIVE (DAS)               reserved hook — DEFERRED, never emitted here
        DELAYED ~15m (Polygon)   RTH + key set + non-empty result
        DEGRADED (Polygon->flat) RTH + Polygon error/empty -> flat-file fallback
        STALE (EOD flat-file)    pre-open / key unset / non-RTH -> flat-file

    Returns (None, None, None) when no bars are available at any tier.
    HONEST LABELING: a df sourced from flat files NEVER gets a DELAYED/LIVE tag.
    """
    poly_eligible = (
        allow_polygon
        and bool(os.getenv("POLYGON_API_KEY"))
        and open_et is not None and close_et is not None
        and _rth_in_progress(open_et, close_et)
    )

    if poly_eligible:
        # Production: query Polygon for *today* (RTH, ~15m delayed). TEST-ONLY: when
        # FALCON_SCANNER_FORCE_RTH_DATE is set, query that past RTH date instead so
        # the off-hours acceptance harness can prove the DELAYED tier end-to-end.
        # The label string is identical either way — no tier is faked.
        poly_iso = (os.getenv("FALCON_SCANNER_FORCE_RTH_DATE")
                    or datetime.now(tz=ET).date().isoformat())
        pdf = _polygon_minute_df(feed, symbol, poly_iso)
        if pdf is not None and not pdf.empty:
            return (pdf, "DELAYED ~15m (Polygon REST)", _last_bar_iso_et(pdf))
        # RTH Polygon attempt produced nothing -> degrade to flat files, labeled
        # DEGRADED (NEVER a faked DELAYED over flat-file data).
        fdf = _flatfile_minute_df(feed, symbol, session_date)
        if fdf is not None and not fdf.empty:
            return (fdf, f"DEGRADED (Polygon error -> flat-file, as of {session_date})",
                    _last_bar_iso_et(fdf))
        return (None, None, None)

    # Pre-open / key unset / non-RTH: straight to flat files, honest STALE.
    fdf = _flatfile_minute_df(feed, symbol, session_date)
    if fdf is not None and not fdf.empty:
        return (fdf, f"STALE (EOD flat-file, as of {session_date})",
                _last_bar_iso_et(fdf))
    return (None, None, None)


# Freshness ordering for envelope "least-fresh" computation. Lower index = fresher.
# 'LIVE' reserved at top (DEFERRED — never emitted by this module).
_TIER_RANK = {"LIVE": 0, "DELAYED": 1, "DEGRADED": 2, "STALE": 3, "UNAVAILABLE": 4}


def _tier_rank(recency: Optional[str]) -> int:
    if not recency:
        return _TIER_RANK["UNAVAILABLE"]
    for key, rank in _TIER_RANK.items():
        if recency.startswith(key):
            return rank
    return _TIER_RANK["UNAVAILABLE"]


# ---------------------------------------------------------------------------
# Reused strategy signals (bella_fade / offside_scalp / microstructure_momentum)
# ---------------------------------------------------------------------------

_STRATEGY_SPECS = [
    ("bella_fade", "BellaFadeStrategy", "bella_fade"),
    ("offside_scalp", "OffsideScalpStrategy", "offside_scalp"),
    ("microstructure_momentum", "MicrostructureMomentumStrategy", "microstructure_momentum"),
]

_STRATEGY_CACHE: Dict[str, Any] = {}


def _strategy_instance(module_name: str, class_name: str):
    if module_name in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[module_name]
    cls = _load_strategy_class(module_name, class_name)
    inst = None
    if cls is not None:
        try:
            inst = cls()
        except Exception as e:
            logger.warning("failed to instantiate %s: %s", class_name, e)
            inst = None
    _STRATEGY_CACHE[module_name] = inst
    return inst


def _run_strategy_signals(strategy, df):
    """Mirror BaseStrategy.run() indicator-merge, then call generate_signals
    directly so we keep ALL entry signals (not validation-filtered). We take the
    LAST entry signal ourselves. Does not modify the strategy class.
    """
    import pandas as pd  # local import; pandas is always present
    data = strategy.preprocess_data(df.copy())
    indicators = strategy.calculate_indicators(data)
    for name, series in indicators.items():
        data[name] = series
    return strategy.generate_signals(data)


def _last_entry_signal(signals, symbol: str):
    """Return the LAST LONG/SHORT Signal of the session (the live setup), or None."""
    from importlib import import_module
    SignalType = import_module("falcon_core.backtesting.strategies.base").SignalType
    last = None
    for s in signals:
        if s.signal_type in (SignalType.LONG, SignalType.SHORT):
            last = s
    if last is not None and getattr(last, "symbol", "") in ("", None):
        last.symbol = symbol
    return last


# ---------------------------------------------------------------------------
# Codified triggers: ORB and VWAP-reclaim (Signal contract reused, not modified)
# ---------------------------------------------------------------------------

def _vwap_series(df):
    import numpy as np
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cumvol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cumvol


def _to_et(ts):
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        return ts.astimezone(ET)
    return ts


def _make_signal(ts, signal_type_name, price, symbol, confidence, stop, target, reason, metadata):
    base = __import__("falcon_core.backtesting.strategies.base",
                      fromlist=["Signal", "SignalType"])
    SignalType = base.SignalType
    Signal = base.Signal
    st = SignalType.LONG if signal_type_name == "long" else SignalType.SHORT
    return Signal(
        timestamp=ts, signal_type=st, price=float(price), symbol=symbol,
        confidence=float(confidence), stop_loss=float(stop), take_profit=float(target),
        reason=reason, metadata=metadata,
    )


def _orb_signal(df, symbol):
    """Opening Range Breakout: define 09:30-09:45 OR high/low; emit on first 1m
    close beyond OR with bar vol > rolling avg AND price on the correct VWAP side;
    stop = opposite OR side; target = 2:1.
    """
    import numpy as np
    vwap = _vwap_series(df)
    avg_vol = df["volume"].rolling(15, min_periods=5).mean()

    # Opening range window in ET.
    et_index = [_to_et(t) for t in df.index]
    or_mask = []
    for t in et_index:
        tt = t.time() if hasattr(t, "time") else None
        or_mask.append(tt is not None and tt >= datetime.strptime("09:30", "%H:%M").time()
                       and tt < datetime.strptime("09:45", "%H:%M").time())
    if not any(or_mask):
        return None
    or_idx = [i for i, m in enumerate(or_mask) if m]
    or_high = float(df["high"].iloc[or_idx].max())
    or_low = float(df["low"].iloc[or_idx].min())
    if or_high <= or_low:
        return None

    last = None
    closes = df["close"].values
    highs = df["high"].values
    vols = df["volume"].values
    for i in range(len(df)):
        t = et_index[i]
        tt = t.time() if hasattr(t, "time") else None
        if tt is None or tt < datetime.strptime("09:45", "%H:%M").time():
            continue
        if np.isnan(avg_vol.iloc[i]) or vols[i] <= avg_vol.iloc[i]:
            continue
        v = vwap.iloc[i]
        cl = closes[i]
        # Long break: close above OR high, on/above VWAP.
        if cl > or_high and not np.isnan(v) and cl >= v:
            entry = cl
            stop = or_low
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + 2.0 * risk
            conf = min(0.95, 0.6 + min((cl - or_high) / max(or_high - or_low, 1e-9), 1.0) * 0.2)
            last = _make_signal(df.index[i], "long", entry, symbol, conf, stop, target,
                                f"ORB long: close {cl:.2f} > OR high {or_high:.2f} on >avg vol, above VWAP",
                                {"or_high": or_high, "or_low": or_low, "vwap": float(v)})
        # Short break: close below OR low, on/below VWAP.
        elif cl < or_low and not np.isnan(v) and cl <= v:
            entry = cl
            stop = or_high
            risk = stop - entry
            if risk <= 0:
                continue
            target = entry - 2.0 * risk
            conf = min(0.95, 0.6 + min((or_low - cl) / max(or_high - or_low, 1e-9), 1.0) * 0.2)
            last = _make_signal(df.index[i], "short", entry, symbol, conf, stop, target,
                                f"ORB short: close {cl:.2f} < OR low {or_low:.2f} on >avg vol, below VWAP",
                                {"or_high": or_high, "or_low": or_low, "vwap": float(v)})
    return last


def _vwap_reclaim_signal(df, symbol):
    """VWAP reclaim: >=3 consecutive bars below VWAP, then a reclaim (close back
    above VWAP) on vol >= 1.2x avg; target 2:1; stop on close back below VWAP
    (modeled as the reclaim bar's low). Long-only reclaim.
    """
    import numpy as np
    vwap = _vwap_series(df)
    avg_vol = df["volume"].rolling(15, min_periods=5).mean()
    closes = df["close"].values
    lows = df["low"].values
    vols = df["volume"].values

    below_run = 0
    last = None
    for i in range(len(df)):
        v = vwap.iloc[i]
        if np.isnan(v):
            continue
        cl = closes[i]
        if cl < v:
            below_run += 1
            continue
        # cl >= v : potential reclaim
        if below_run >= 3 and not np.isnan(avg_vol.iloc[i]) and vols[i] >= 1.2 * avg_vol.iloc[i]:
            entry = cl
            stop = min(lows[i], float(v) * 0.999)  # close back below VWAP / reclaim-bar low
            risk = entry - stop
            if risk > 0:
                target = entry + 2.0 * risk
                conf = min(0.92, 0.58 + min(below_run / 6.0, 1.0) * 0.2
                           + min(vols[i] / max(avg_vol.iloc[i], 1e-9) / 3.0, 0.15))
                last = _make_signal(df.index[i], "long", entry, symbol, conf, stop, target,
                                    f"VWAP reclaim: {below_run} bars below then reclaim on "
                                    f"{vols[i]/max(avg_vol.iloc[i],1e-9):.1f}x vol",
                                    {"vwap": float(v), "below_run": int(below_run)})
        below_run = 0
    return last


# ---------------------------------------------------------------------------
# Gates, ranking, mapping to recommendation dict
# ---------------------------------------------------------------------------

def _risk_level(rr: float) -> str:
    if rr >= 2.5:
        return "Low"
    if rr >= 1.8:
        return "Medium"
    return "High"


def _passes_cost_speed_limit(entry: float, target: float) -> bool:
    """Expected move must clear round-trip cost by a wide margin."""
    if entry <= 0:
        return False
    expected_move_bps = abs(target - entry) / entry * 10_000.0
    return expected_move_bps >= ROUND_TRIP_COST_BPS * COST_SPEED_LIMIT_MULT


def _signal_to_rec(signal, setup_type: str, session_date: str, data_recency: str,
                   valid_until: str, dollar_volume: float,
                   last_bar_ts: Optional[str] = None) -> Optional[Dict[str, Any]]:
    entry = float(signal.price)
    stop = signal.stop_loss
    target = signal.take_profit
    if stop is None or target is None or entry <= 0:
        return None
    # Direction-aware R:R.
    is_long = (target >= entry)
    if is_long:
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0 or reward <= 0:
        return None
    rr = float(reward / risk)
    confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
    edge_score = float(rr * confidence)

    # Cast all numerics to native Python floats/ints — numpy scalars are not JSON
    # serializable by Flask jsonify / ProfileManager run_data serialization.
    return {
        # --- structured intraday-setup fields (new contract) ---
        "ticker": str(signal.symbol),
        "setup_type": setup_type,
        "trigger_detail": str(signal.reason),
        "direction": "long" if is_long else "short",
        "entry": round(float(entry), 4),
        "stop": round(float(stop), 4),
        "target": round(float(target), 4),
        "rr": round(rr, 2),
        "edge_score": round(edge_score, 4),
        "valid_until": valid_until,
        "data_recency": data_recency,
        "last_bar_ts": last_bar_ts,
        "session_date": session_date,
        "dollar_volume": round(float(dollar_volume), 2),
        # --- backward-compatible fields the existing www table consumes ---
        "entry_price_range": f"{float(entry) * 0.999:.2f}-{float(entry) * 1.001:.2f}",
        "target_price": round(float(target), 2),
        "stop_loss": round(float(stop), 2),
        "risk_level": _risk_level(rr),
        "confidence_score": int(round(confidence * 10)),
        "reasoning": str(signal.reason),
    }


def _session_window(session_date: str) -> Tuple[str, datetime, datetime, datetime]:
    """Return (session_close_iso, open_et, close_et, suppress_after_et)."""
    d = datetime.strptime(session_date, "%Y-%m-%d").date()
    open_et = datetime.combine(d, datetime.strptime("09:30", "%H:%M").time(), tzinfo=ET)
    close_et = datetime.combine(d, datetime.strptime("16:00", "%H:%M").time(), tzinfo=ET)
    suppress_after = close_et - timedelta(minutes=SUPPRESS_LAST_MINUTES)
    return close_et.isoformat(), open_et, close_et, suppress_after


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_intraday_setups(session_date: Optional[str] = None,
                         universe: Optional[List[str]] = None,
                         use_cache: bool = True) -> Dict[str, Any]:
    """Scan the most-recent available flat-file session for ranked intraday setups.

    Returns:
        {
          "session_date": "YYYY-MM-DD" or None,
          "last_bar_ts": ISO ts or None,
          "data_source": "flatfiles",
          "data_recency": "STALE (EOD flat-file, as of <session_date>)" or status,
          "setups": [rec_dict, ...],   # <= HARD_CAP, ranked by edge_score, unique tickers
        }
    Never raises — degrades to {"setups": []} with a clear data_recency on any
    failure (e.g. boto3/flat files unavailable in this image).
    """
    cache_key = session_date or "__latest__"
    if use_cache:
        hit = _CACHE.get(cache_key)
        if hit and (time.time() - hit[0]) < _CACHE_TTL_SECONDS:
            return hit[1]

    result: Dict[str, Any] = {
        "session_date": None,
        "last_bar_ts": None,
        "data_source": "flatfiles",
        "data_recency": "UNAVAILABLE (flat-file client not available in this process)",
        "setups": [],
    }

    feed = _make_datafeed()
    if feed is None:
        if use_cache:
            _CACHE[cache_key] = (time.time(), result)
        return result

    resolved = resolve_session_date(session_date)
    if not resolved:
        result["data_recency"] = "UNAVAILABLE (no flat-file session resolved)"
        if use_cache:
            _CACHE[cache_key] = (time.time(), result)
        return result

    data_recency = f"STALE (EOD flat-file, as of {resolved})"
    result["session_date"] = resolved
    result["data_recency"] = data_recency

    close_iso, open_et, close_et, suppress_after = _session_window(resolved)

    uni = universe or (os.getenv("FALCON_INTRADAY_UNIVERSE", "").split(",")
                       if os.getenv("FALCON_INTRADAY_UNIVERSE") else DEFAULT_UNIVERSE)
    uni = [s.strip().upper() for s in uni if s and s.strip()]

    candidates: List[Dict[str, Any]] = []
    last_bar_ts = None

    for sym in uni:
        # Tier-aware fetch: Polygon REST (DELAYED) during RTH if keyed, else
        # flat-file (STALE / DEGRADED). Per-symbol recency travels with each setup.
        df, sym_recency, sym_last_bar = _get_minute_df(
            feed, sym, resolved, open_et=open_et, close_et=close_et
        )
        if df is None or len(df) < 30:
            continue
        # Per-symbol recency falls back to the envelope STALE label defensively;
        # NEVER upgrade a flat-file df to DELAYED.
        if not sym_recency:
            sym_recency = data_recency
        if sym_last_bar and (last_bar_ts is None or sym_last_bar > last_bar_ts):
            last_bar_ts = sym_last_bar

        # Liquidity gate: session dollar-volume.
        try:
            dollar_volume = float((df["close"] * df["volume"]).sum())
            session_close_px = float(df["close"].iloc[-1])
        except Exception:
            continue
        if session_close_px < MIN_PRICE or dollar_volume < MIN_DOLLAR_VOLUME:
            continue

        # Collect the LAST entry signal from each reused strategy + each new trigger.
        per_symbol: List[Tuple[str, Any]] = []
        for module_name, class_name, setup_type in _STRATEGY_SPECS:
            strat = _strategy_instance(module_name, class_name)
            if strat is None:
                continue
            try:
                signals = _run_strategy_signals(strat, df)
            except Exception as e:
                logger.debug("%s failed on %s: %s", module_name, sym, e)
                continue
            sig = _last_entry_signal(signals, sym)
            if sig is not None:
                per_symbol.append((setup_type, sig))

        try:
            orb = _orb_signal(df, sym)
            if orb is not None:
                per_symbol.append(("orb", orb))
        except Exception as e:
            logger.debug("ORB failed on %s: %s", sym, e)
        try:
            vr = _vwap_reclaim_signal(df, sym)
            if vr is not None:
                per_symbol.append(("vwap_reclaim", vr))
        except Exception as e:
            logger.debug("VWAP-reclaim failed on %s: %s", sym, e)

        for setup_type, sig in per_symbol:
            # Suppress setups that triggered in the last 30 min of RTH.
            try:
                sig_et = _to_et(sig.timestamp)
                if hasattr(sig_et, "tzinfo") and sig_et.tzinfo is not None and sig_et >= suppress_after:
                    continue
            except Exception:
                pass

            rec = _signal_to_rec(sig, setup_type, resolved, sym_recency, close_iso,
                                 dollar_volume, last_bar_ts=sym_last_bar)
            if rec is None:
                continue
            # GATES (before ranking): price, R:R, cost speed-limit.
            if rec["entry"] < MIN_PRICE:
                continue
            if rec["rr"] < MIN_RR:
                continue
            if not _passes_cost_speed_limit(rec["entry"], rec["target"]):
                continue
            # Trend vs fade weighting baked into the ranked edge_score.
            weight = TREND_WEIGHT if setup_type in TREND_SETUPS else FADE_WEIGHT
            rec["edge_score"] = round(rec["edge_score"] * weight, 4)
            candidates.append(rec)

    # Dedup by ticker, keep highest edge_score (fallback confidence_score).
    ticker_map: Dict[str, Dict[str, Any]] = {}
    for rec in candidates:
        t = rec.get("ticker", "")
        if not t:
            continue
        cur = ticker_map.get(t)
        if not cur or rec.get("edge_score", 0) > cur.get("edge_score", 0) or (
            rec.get("edge_score", 0) == cur.get("edge_score", 0)
            and rec.get("confidence_score", 0) > cur.get("confidence_score", 0)
        ):
            ticker_map[t] = rec

    ranked = sorted(
        ticker_map.values(),
        key=lambda x: (x.get("edge_score", 0), x.get("confidence_score", 0)),
        reverse=True,
    )[:HARD_CAP]
    for i, rec in enumerate(ranked, start=1):
        rec["rank"] = i

    result["setups"] = ranked
    result["last_bar_ts"] = last_bar_ts

    # Envelope data_recency = the LEAST-FRESH (worst) tier present among returned
    # setups. data_source follows the worst tier. With zero setups, keep the
    # resolved STALE/UNAVAILABLE label already set above (never fabricate DELAYED).
    if ranked:
        worst = max(ranked, key=lambda r: _tier_rank(r.get("data_recency")))
        result["data_recency"] = worst.get("data_recency") or data_recency
        result["data_source"] = (
            "polygon" if result["data_recency"].startswith("DELAYED") else "flatfiles"
        )

    if use_cache:
        _CACHE[cache_key] = (time.time(), result)
    return result


# ---------------------------------------------------------------------------
# CLI / one-shot: run the scan and PERSIST setups into profile_runs so the
# dashboard endpoint (which lacks boto3) can serve them. Run in falcon-core image.
# ---------------------------------------------------------------------------

INTRADAY_PROFILE_NAME = "Intraday Scanner"
INTRADAY_PROFILE_THEME = "intraday_setup"


def _ensure_intraday_profile(manager):
    """Get-or-create the dedicated intraday scanner profile. Returns profile id."""
    existing = manager.get_profile_by_name(INTRADAY_PROFILE_NAME)
    if existing is not None:
        return existing.id
    from falcon_screener.profile_manager import ScreenerProfile
    prof = ScreenerProfile(
        name=INTRADAY_PROFILE_NAME,
        theme=INTRADAY_PROFILE_THEME,
        description="Live intraday setup scanner (flat-file minute bars, EOD/stale).",
        enabled=True,
        schedule={"morning": False, "midday": True, "evening": False},
    )
    return manager.create_profile(prof)


def persist_scan(scan: Dict[str, Any]) -> int:
    """Persist a scan result into profile_runs via ProfileManager.log_profile_run.
    Returns the run id (or -1 on failure). Stamps recency into run_data.
    """
    try:
        from falcon_core import get_db_manager
        from falcon_screener.profile_manager import ProfileManager
    except Exception as e:
        logger.error("cannot import ProfileManager/get_db_manager: %s", e)
        return -1
    db = get_db_manager()
    manager = ProfileManager(db)
    profile_id = _ensure_intraday_profile(manager)
    setups = scan.get("setups", [])
    run_data = {
        "recommendations": setups,
        "data_source": scan.get("data_source"),
        "session_date": scan.get("session_date"),
        "last_bar_ts": scan.get("last_bar_ts"),
        "data_recency": scan.get("data_recency"),
    }
    return manager.log_profile_run(
        profile_id=profile_id,
        run_type="intraday_scan",
        stocks_found=len(setups),
        recommendations=len(setups),
        ai_agent="intraday_scanner",
        run_data=run_data,
    )


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Falcon intraday setup scanner (flat files).")
    p.add_argument("--session-date", default=None, help="YYYY-MM-DD (default: newest available)")
    p.add_argument("--no-persist", action="store_true", help="print only, do not write to DB")
    p.add_argument("--json", action="store_true", help="print full scan JSON")
    args = p.parse_args(argv)

    scan = scan_intraday_setups(session_date=args.session_date, use_cache=False)
    print(f"session_date={scan['session_date']} recency={scan['data_recency']} "
          f"setups={len(scan['setups'])}")
    for rec in scan["setups"]:
        print(f"  #{rec.get('rank')} {rec['ticker']:6} {rec['setup_type']:22} "
              f"{rec['direction']:5} entry={rec['entry']} stop={rec['stop']} "
              f"target={rec['target']} rr={rec['rr']} edge={rec['edge_score']}")
    if args.json:
        print(json.dumps(scan, indent=2, default=str))

    if not args.no_persist and scan["setups"]:
        run_id = persist_scan(scan)
        print(f"persisted run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
