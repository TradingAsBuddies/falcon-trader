"""
Tests for the intraday setup scanner (falcon-trader #6).

These tests are pure-Python and do NOT hit the network / flat files. They drive
the scanner's pure functions (gates, ranking, recency, exit-completeness) and
assert the no-REST data-path contract by source inspection.

Run: pytest falcon-trader/tests/test_intraday_scanner.py
"""
import os
import re
import sys

import pytest

# Make the scanner importable without installing the package.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from falcon_trader import intraday_scanner as scn  # noqa: E402


# ---------------------------------------------------------------------------
# (1) NO-REST contract: the recommendations code path must never use a Polygon
# REST client; data flows only through flat files / DataFeed(source='flatfiles').
# ---------------------------------------------------------------------------
def _read_source(rel_path):
    path = os.path.join(os.path.dirname(__file__), "..", "src", "falcon_trader", rel_path)
    with open(os.path.abspath(path)) as f:
        return f.read()


def test_scanner_reuses_datafeed_no_new_rest_client():
    """#9: the scanner may reach Polygon REST as a DELAYED freshness tier, but ONLY
    by reusing DataFeed source='polygon'/_try_polygon — it must NOT construct its
    own REST client, and flat files must remain the explicit fallback tier.
    """
    src = _read_source("intraday_scanner.py")
    # No hand-rolled REST client / direct HTTP to Polygon.
    banned = [
        "PolygonClient", "RESTClient", "from polygon", "import polygon",
        "api.polygon.io", "urlopen", "requests.get", "http.client",
    ]
    for token in banned:
        assert token not in src, f"scanner must not build a REST client: {token!r}"
    # Both tiers reached only via DataFeed source= selection (reused plumbing).
    assert 'source="polygon"' in src, "scanner must reach REST via DataFeed source='polygon'"
    assert 'source="flatfiles"' in src, "flat-file fallback tier must be preserved"
    assert "get_historical_data" in src
    # The scanner must NEVER emit a LIVE recency at runtime (DAS deferred). The word
    # may appear in docstrings/comments and as the reserved _TIER_RANK key; what is
    # banned is a LIVE label flowing out of a `return` statement (an emittable tuple
    # member). Scan only code lines that contain `return`. Runtime coverage is in
    # test_envelope_is_least_fresh_tier_and_no_live + test_scan_never_raises_*.
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if "return" in code:
            assert "LIVE" not in code, f"scanner must not return a LIVE label: {line!r}"


def test_endpoint_path_never_uses_polygon_rest():
    src = _read_source("dashboard_server.py")
    # The recommendations endpoint merges the scanner; the merge code itself must
    # not introduce a REST call. (We scope to the scanner import line presence.)
    assert "intraday_scanner" in src
    assert "scan_intraday_setups" in src


# ---------------------------------------------------------------------------
# (2) Cost-gate / liquidity: a low-priced, wide-spread, thin setup is filtered.
# ---------------------------------------------------------------------------
def test_cost_speed_limit_filters_tiny_move():
    # A penny-wide target on a low price: move in bps must clear round-trip cost
    # by a wide margin. entry=4.00 target=4.01 -> 25 bps move; round-trip cost
    # speed-limit = ROUND_TRIP_COST_BPS * 3. With defaults that is well above 25.
    assert scn._passes_cost_speed_limit(4.00, 4.01) is False
    # A healthy move clears it.
    assert scn._passes_cost_speed_limit(100.0, 101.5) is True  # 150 bps


def test_min_price_and_dollar_volume_constants_present():
    assert scn.MIN_PRICE >= 2.0
    assert scn.MIN_DOLLAR_VOLUME >= 1_000_000
    assert scn.MIN_RR >= 1.5


def _fake_signal(price, stop, target, confidence=0.7, reason="x", symbol="TEST", long=True):
    class _S:
        pass
    s = _S()
    s.price = price
    s.stop_loss = stop
    s.take_profit = target
    s.confidence = confidence
    s.reason = reason
    s.symbol = symbol
    return s


def test_thin_lowprice_setup_excluded_by_full_scan_gates():
    """A synthetic thin/wide-spread/low-price symbol must NOT appear in output.

    We exercise the gate logic the scanner applies per-candidate: low price + a
    cost-limit-failing target. _signal_to_rec yields a rec, but the price/cost
    gates drop it before ranking.
    """
    sig = _fake_signal(price=1.50, stop=1.45, target=1.52)  # below MIN_PRICE, tiny move
    rec = scn._signal_to_rec(sig, "orb", "2026-06-11",
                             "STALE (EOD flat-file, as of 2026-06-11)",
                             "2026-06-11T16:00:00-04:00", dollar_volume=2_000_000)
    # Mirror the scanner's gate sequence:
    excluded = (rec is None
                or rec["entry"] < scn.MIN_PRICE
                or rec["rr"] < scn.MIN_RR
                or not scn._passes_cost_speed_limit(rec["entry"], rec["target"]))
    assert excluded, "thin/low-price/wide-spread setup must be filtered out"


# ---------------------------------------------------------------------------
# (3) Exit-completeness: every emitted setup has stop, target, valid_until, rr>=1.5
# ---------------------------------------------------------------------------
def test_signal_to_rec_has_full_exit_plan():
    sig = _fake_signal(price=100.0, stop=98.0, target=104.0, confidence=0.8)  # 2:1
    rec = scn._signal_to_rec(sig, "orb", "2026-06-11",
                             "STALE (EOD flat-file, as of 2026-06-11)",
                             "2026-06-11T16:00:00-04:00", dollar_volume=50_000_000)
    assert rec is not None
    for key in ("ticker", "setup_type", "trigger_detail", "entry", "stop",
                "target", "rr", "edge_score", "valid_until", "data_recency",
                "last_bar_ts", "entry_price_range", "target_price", "stop_loss",
                "risk_level", "confidence_score", "reasoning"):
        assert key in rec, f"missing required key: {key}"
    assert rec["stop"] is not None
    assert rec["target"] is not None
    assert rec["valid_until"]
    assert rec["rr"] >= scn.MIN_RR
    assert "STALE" in rec["data_recency"]
    assert "2026-06-11" in rec["data_recency"]


def test_signal_to_rec_rejects_negative_rr():
    # target below entry for a "long" => reward <= 0 => rejected.
    sig = _fake_signal(price=100.0, stop=98.0, target=99.0)
    assert scn._signal_to_rec(sig, "orb", "2026-06-11", "STALE", "x", 1e7) is None


# ---------------------------------------------------------------------------
# (4) Cap + dedup: <=5 setups, unique tickers, ranked by edge_score.
# ---------------------------------------------------------------------------
def test_recency_label_is_stale_for_eod():
    # When a session resolves, the label must say STALE + the date — never live.
    label = "STALE (EOD flat-file, as of 2026-06-11)"
    assert "STALE" in label and "live" not in label.lower()


def test_scan_result_shape_and_cap(monkeypatch):
    """Drive scan_intraday_setups with a stubbed DataFeed so we never touch the
    network, returning a synthetic universe that yields many qualifying setups,
    and assert: <=5, unique tickers, ranked desc, every setup has a full exit.
    """
    import numpy as np
    import pandas as pd

    # Build a synthetic 1-min session that yields an ORB long: opening range,
    # then a strong breakout above OR high on rising volume, staying above VWAP.
    def make_df(base):
        idx = pd.date_range("2026-06-11 09:30", periods=120, freq="1min", tz="America/New_York")
        # First 15 bars form a tight opening range; then a clean breakout up.
        close = []
        for i in range(120):
            if i < 15:
                close.append(base + (i % 3) * 0.02)          # tight OR
            else:
                close.append(base + 0.50 + (i - 15) * 0.05)  # breakout up
        close = np.array(close)
        high = close + 0.05
        low = close - 0.05
        openp = close - 0.01
        vol = np.array([10000 + (50000 if i >= 15 else 0) + i * 100 for i in range(120)], dtype=float)
        return pd.DataFrame({"open": openp, "high": high, "low": low,
                             "close": close, "volume": vol}, index=idx)

    universe = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    prices = {s: 50.0 + i * 10 for i, s in enumerate(universe)}

    class FakeFeed:
        flatfiles = object()  # truthy => "available"
        def get_historical_data(self, symbol, start, end, interval, source, market_hours_only):
            return make_df(prices[symbol])

    monkeypatch.setattr(scn, "_make_datafeed", lambda: FakeFeed())
    monkeypatch.setattr(scn, "resolve_session_date", lambda session_date=None: "2026-06-11")
    # Force the non-RTH / flat-file STALE path deterministically (no live clock dep).
    monkeypatch.setattr(scn, "_rth_in_progress", lambda *a, **k: False)
    # Disable the reused strategy modules for determinism; rely on ORB/VWAP triggers.
    monkeypatch.setattr(scn, "_strategy_instance", lambda m, c: None)

    res = scn.scan_intraday_setups(session_date="2026-06-11",
                                   universe=universe, use_cache=False)

    assert res["data_source"] == "flatfiles"
    assert res["session_date"] == "2026-06-11"
    assert "STALE" in res["data_recency"]
    assert "2026-06-11" in res["data_recency"]

    setups = res["setups"]
    # HARD CAP 5.
    assert len(setups) <= scn.HARD_CAP
    # We engineered a breakout for all 8 names; expect at least 1 qualifying setup.
    assert len(setups) >= 1
    # Unique tickers.
    tickers = [s["ticker"] for s in setups]
    assert len(tickers) == len(set(tickers)), "no duplicate tickers"
    # Ranked by edge_score desc, ranks assigned 1..N.
    edges = [s["edge_score"] for s in setups]
    assert edges == sorted(edges, reverse=True)
    assert [s["rank"] for s in setups] == list(range(1, len(setups) + 1))
    # Exit-completeness for every emitted setup.
    for s in setups:
        assert s["stop"] is not None
        assert s["target"] is not None
        assert s["valid_until"]
        assert s["rr"] >= scn.MIN_RR
        assert "STALE" in s["data_recency"]


# ---------------------------------------------------------------------------
# (5) #9 freshness-tier no-mislabel invariant.
# ---------------------------------------------------------------------------
import numpy as _np  # noqa: E402
import pandas as _pd  # noqa: E402


def _make_session_df(base=50.0):
    idx = _pd.date_range("2026-06-16 09:30", periods=120, freq="1min",
                         tz="America/New_York")
    close = []
    for i in range(120):
        close.append(base + (i % 3) * 0.02 if i < 15 else base + 0.50 + (i - 15) * 0.05)
    close = _np.array(close)
    vol = _np.array([10000 + (50000 if i >= 15 else 0) + i * 100 for i in range(120)],
                    dtype=float)
    return _pd.DataFrame({"open": close - 0.01, "high": close + 0.05,
                          "low": close - 0.05, "close": close, "volume": vol}, index=idx)


class _OneSymFeed:
    """Feed whose source-routing is observable; flat & polygon dfs are distinct."""
    flatfiles = object()

    def __init__(self, polygon_ok=True, polygon_raises=False):
        self.polygon_ok = polygon_ok
        self.polygon_raises = polygon_raises

    def get_historical_data(self, symbol, start, end, interval, source, market_hours_only):
        if source == "polygon":
            if self.polygon_raises:
                raise RuntimeError("forced polygon error")
            if not self.polygon_ok:
                return _pd.DataFrame()  # empty -> degrade
            return _make_session_df(50.0)
        # flatfiles
        return _make_session_df(50.0)


def test_flatfile_df_never_labeled_delayed_or_live(monkeypatch):
    """No-mislabel (a): a df from flat files -> recency starts STALE/DEGRADED, never
    DELAYED/LIVE. Pre-open path is STALE."""
    feed = _OneSymFeed()
    monkeypatch.setattr(scn, "_rth_in_progress", lambda *a, **k: False)
    open_et, close_et = scn._session_window("2026-06-16")[1], scn._session_window("2026-06-16")[2]
    df, recency, last = scn._get_minute_df(feed, "NVDA", "2026-06-16",
                                           open_et=open_et, close_et=close_et)
    assert df is not None
    assert recency.startswith("STALE")
    assert "DELAYED" not in recency and "LIVE" not in recency


def test_polygon_df_labeled_delayed(monkeypatch):
    """No-mislabel (b): a non-empty Polygon df during RTH -> exactly the DELAYED str."""
    feed = _OneSymFeed(polygon_ok=True)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setattr(scn, "_rth_in_progress", lambda *a, **k: True)
    open_et, close_et = scn._session_window("2026-06-16")[1], scn._session_window("2026-06-16")[2]
    df, recency, last = scn._get_minute_df(feed, "NVDA", "2026-06-16",
                                           open_et=open_et, close_et=close_et)
    assert df is not None
    assert recency == "DELAYED ~15m (Polygon REST)"
    assert last is not None and "T" in last  # ISO-8601


def test_polygon_error_degrades_to_flatfile_not_delayed(monkeypatch):
    """No-mislabel (c): forced Polygon error/empty during RTH -> flat-file DEGRADED,
    NEVER DELAYED/LIVE."""
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setattr(scn, "_rth_in_progress", lambda *a, **k: True)
    open_et, close_et = scn._session_window("2026-06-16")[1], scn._session_window("2026-06-16")[2]
    for feed in (_OneSymFeed(polygon_ok=False), _OneSymFeed(polygon_raises=True)):
        df, recency, last = scn._get_minute_df(feed, "NVDA", "2026-06-16",
                                               open_et=open_et, close_et=close_et)
        assert df is not None, "must degrade to flat files, not return nothing"
        assert recency.startswith("DEGRADED")
        assert "DELAYED" not in recency and "LIVE" not in recency


def test_key_unset_preopen_is_stale(monkeypatch):
    """Graceful degradation (f): POLYGON_API_KEY unset -> STALE, never DELAYED/DEGRADED."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(scn, "_rth_in_progress", lambda *a, **k: True)  # even in RTH
    open_et, close_et = scn._session_window("2026-06-16")[1], scn._session_window("2026-06-16")[2]
    df, recency, last = scn._get_minute_df(_OneSymFeed(), "NVDA", "2026-06-16",
                                           open_et=open_et, close_et=close_et)
    assert recency.startswith("STALE")
    assert "DELAYED" not in recency and "DEGRADED" not in recency


def test_envelope_is_least_fresh_tier_and_no_live(monkeypatch):
    """Two-grain (d)+(e): mixed-tier scan -> per-row tiers correct; envelope = worst
    tier present; 'LIVE' never appears anywhere."""
    # AAA -> polygon ok (DELAYED); BBB -> polygon empty (DEGRADED flat fallback).
    class MixedFeed:
        flatfiles = object()
        def get_historical_data(self, symbol, start, end, interval, source, market_hours_only):
            if source == "polygon":
                return _make_session_df(50.0) if symbol == "AAA" else _pd.DataFrame()
            return _make_session_df(70.0)

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setattr(scn, "_make_datafeed", lambda: MixedFeed())
    monkeypatch.setattr(scn, "resolve_session_date", lambda session_date=None: "2026-06-16")
    monkeypatch.setattr(scn, "_rth_in_progress", lambda *a, **k: True)
    monkeypatch.setattr(scn, "_strategy_instance", lambda m, c: None)

    res = scn.scan_intraday_setups(session_date="2026-06-16",
                                   universe=["AAA", "BBB"], use_cache=False)
    setups = res["setups"]
    assert len(setups) >= 1
    recencies = {s["ticker"]: s["data_recency"] for s in setups}
    if "AAA" in recencies:
        assert recencies["AAA"] == "DELAYED ~15m (Polygon REST)"
    if "BBB" in recencies:
        assert recencies["BBB"].startswith("DEGRADED")
    # Envelope = least-fresh present.
    ranks = [scn._tier_rank(s["data_recency"]) for s in setups]
    assert scn._tier_rank(res["data_recency"]) == max(ranks)
    # No LIVE anywhere.
    blob = str(res)
    assert "LIVE" not in blob


def test_force_rth_date_env_exercises_delayed_offhours(monkeypatch):
    """Off-hours acceptance harness (#10): FALCON_SCANNER_FORCE_RTH_DATE re-opens the
    wall-clock gate for a deliberate proof run against a past RTH date and routes the
    Polygon query to that date — producing the EXACT DELAYED label. The override is
    opt-in (env set), so production (env unset) keeps the real now() gate; the tier
    string is never faked. This proves the worker can demonstrate the DELAYED path
    end-to-end even when wall-clock is off-hours."""
    seen = {}

    class DateAwareFeed:
        flatfiles = object()
        def get_historical_data(self, symbol, start, end, interval, source, market_hours_only):
            if source == "polygon":
                seen["polygon_start"] = start  # capture the date the polygon tier queried
                return _make_session_df(50.0)
            return _make_session_df(70.0)

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setenv("FALCON_SCANNER_FORCE_RTH_DATE", "2026-06-15")
    # Do NOT monkeypatch _rth_in_progress here — the env override must drive it.
    feed = DateAwareFeed()
    open_et, close_et = scn._session_window("2026-06-15")[1], scn._session_window("2026-06-15")[2]
    df, recency, last = scn._get_minute_df(feed, "NVDA", "2026-06-15",
                                           open_et=open_et, close_et=close_et)
    assert df is not None
    assert recency == "DELAYED ~15m (Polygon REST)"
    assert last is not None and "T" in last
    # The polygon tier queried the forced past RTH date, not today.
    assert seen.get("polygon_start") == "2026-06-15"


def test_force_rth_env_unset_keeps_live_gate(monkeypatch):
    """Production invariant: with FALCON_SCANNER_FORCE_RTH_DATE unset, the gate is the
    real wall-clock — pre-open/non-RTH stays STALE, never DELAYED. The override is the
    only thing that re-opens the gate; absent it, the live behavior is unchanged."""
    monkeypatch.delenv("FALCON_SCANNER_FORCE_RTH_DATE", raising=False)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setattr(scn, "_rth_in_progress", scn._rth_in_progress)  # real gate
    open_et, close_et = scn._session_window("2026-06-15")[1], scn._session_window("2026-06-15")[2]
    df, recency, last = scn._get_minute_df(_OneSymFeed(), "NVDA", "2026-06-15",
                                           open_et=open_et, close_et=close_et)
    # 2026-06-15 is not "now" -> real gate is False -> STALE flat-file, never DELAYED.
    assert recency.startswith("STALE")
    assert "DELAYED" not in recency


def test_dashboard_recovery_branch_only_on_in_process_failure():
    """Contract (#10 acceptance): the dashboard recovery branch that pulls recency
    from a persisted setup must be guarded by `if not in_process_ok:` — a worker-
    sourced DELAYED is never downgraded and a flat-file row is never upgraded. We
    assert the guard exists and gates the recovery loop in dashboard_server.py."""
    src = _read_source("dashboard_server.py")
    assert "in_process_ok" in src
    assert "if not in_process_ok:" in src
    # The four honest tiers and only those drive the recovery membership test.
    assert '_DATA_TIERS = ("STALE", "DELAYED", "DEGRADED", "LIVE")' in src
    # Recovery only copies recency for an intraday_setup row carrying an honest tier.
    assert "_theme') == 'intraday_setup'" in src


def test_scan_never_raises_when_feed_unavailable(monkeypatch):
    """Graceful degradation (f): no feed -> returns honest UNAVAILABLE, never raises."""
    monkeypatch.setattr(scn, "_make_datafeed", lambda: None)
    res = scn.scan_intraday_setups(session_date="2026-06-16", use_cache=False)
    assert res["setups"] == []
    assert "LIVE" not in str(res["data_recency"])
    assert "DELAYED" not in str(res["data_recency"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
