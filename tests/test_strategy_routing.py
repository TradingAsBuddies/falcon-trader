"""Strategy routing on real data.

The executor called router.route(symbol, use_yfinance=False). With live data
disabled the classifier returns *mock* data: a handful of test symbols, and
price 0.0 for anything else. route() then took its "Failed to fetch stock
data, using default strategy" branch, so every real symbol got
rsi_mean_reversion and the momentum and bollinger engines never ran. A dry run
of 29 screener candidates on 2026-10-05 routed all 29 to RSI.
"""

import pytest
import yaml
from pathlib import Path

import falcon_trader
from falcon_trader.orchestrator.execution.trade_executor import TradeExecutor
from falcon_trader.orchestrator.routers.stock_classifier import StockClassifier
from falcon_trader.orchestrator.routers.strategy_router import StrategyRouter

CONFIG = yaml.safe_load(
    (Path(falcon_trader.__file__).parent / "orchestrator" / "orchestrator_config.yaml").read_text()
)


def _bars(start, step, n=30, volume=1_000_000):
    """A price series: `step` is the per-bar return, so volatility is known."""
    prices, price = [], start
    for _ in range(n):
        prices.append(round(price, 4))
        price *= (1 + step)
    return {"price": prices[-1], "prices": prices,
            "volumes": [volume] * n, "source": "test"}


def _alternating(start, swing, n=30):
    """Up-down-up price series: large daily moves, no net drift."""
    prices, price = [], start
    for i in range(n):
        prices.append(round(price, 4))
        price *= (1 + swing) if i % 2 == 0 else (1 - swing)
    return {"price": prices[-1], "prices": prices,
            "volumes": [1_000_000] * n, "source": "test"}


def _router():
    return StrategyRouter(CONFIG)


# ── volatility from real bars ───────────────────────────────────────────

def test_a_flat_series_has_no_volatility():
    assert StockClassifier(CONFIG).annualized_volatility([10.0] * 30) == 0.0


def test_a_swinging_series_is_volatile():
    prices = _alternating(100.0, 0.05)["prices"]
    assert StockClassifier(CONFIG).annualized_volatility(prices) > 0.30


@pytest.mark.parametrize("prices", [None, [], [10.0], [10.0, 10.5], [0, 0, 0]])
def test_too_little_history_is_zero_not_an_error(prices):
    assert StockClassifier(CONFIG).annualized_volatility(prices) == 0.0


# ── classification from real bars ───────────────────────────────────────

def test_a_cheap_stock_is_a_penny_stock():
    profile = StockClassifier(CONFIG).profile_from_market_data("CDXS", _bars(1.40, 0.0))
    assert profile.classification == "penny_stock"
    assert profile.price == pytest.approx(1.40)


def test_an_etf_is_recognised_from_the_configured_list():
    profile = StockClassifier(CONFIG).profile_from_market_data("SPY", _bars(760.0, 0.0))
    assert profile.is_etf is True
    assert profile.classification == "etf"


def test_an_unknown_capitalisation_says_unknown_not_small():
    """A missing market cap used to read as small_cap, mislabelling NKE."""
    profile = StockClassifier(CONFIG).profile_from_market_data("NKE", _bars(33.87, 0.0))
    assert profile.classification == "unknown_cap"


def test_a_supplied_capitalisation_is_used():
    classifier = StockClassifier(CONFIG)
    big = classifier.profile_from_market_data("NKE", _bars(33.87, 0.0), market_cap=150e9)
    mid = classifier.profile_from_market_data("ADPT", _bars(28.05, 0.0), market_cap=20e9)
    small = classifier.profile_from_market_data("WGRX", _bars(8.52, 0.0), market_cap=4e8)
    assert (big.classification, mid.classification, small.classification) == (
        "large_cap", "mid_cap", "small_cap")


def test_the_supplied_sector_reaches_the_profile():
    profile = StockClassifier(CONFIG).profile_from_market_data(
        "NVDA", _bars(228.30, 0.0), sector="Technology")
    assert profile.sector == "Technology"


# ── routing on that profile ─────────────────────────────────────────────

def test_a_real_symbol_no_longer_falls_through_to_the_default():
    """The regression: this returned the default with 'Failed to fetch'."""
    decision = _router().route("NKE", market_data=_bars(33.87, 0.0))
    assert "Failed to fetch" not in decision.reason
    assert decision.classification != "unknown"


def test_a_penny_stock_routes_to_momentum():
    decision = _router().route("CDXS", market_data=_bars(1.40, 0.0))
    assert decision.selected_strategy == "momentum_breakout"


def test_a_volatile_stock_routes_to_momentum():
    decision = _router().route("XPEV", market_data=_alternating(9.25, 0.05))
    assert decision.profile.volatility > 0.30
    assert decision.selected_strategy == "momentum_breakout"


def test_an_etf_routes_to_rsi():
    assert _router().route("SPY", market_data=_bars(760.0, 0.0)).selected_strategy \
        == "rsi_mean_reversion"


def test_a_calm_unknown_cap_stays_on_the_configured_default():
    """Unknown capitalisation must not hand the trade to momentum on no evidence."""
    decision = _router().route("NKE", market_data=_bars(33.87, 0.0005))
    assert decision.classification == "unknown_cap"
    assert decision.selected_strategy == CONFIG["strategy_mapping"]["default"]


def test_a_calm_large_cap_routes_to_rsi():
    decision = _router().route("MDT", market_data=_bars(86.38, 0.0005), market_cap=150e9)
    assert decision.selected_strategy == "rsi_mean_reversion"


def test_without_market_data_the_old_mock_path_is_untouched():
    """Other callers still work; only the price-0 branch is avoided."""
    decision = _router().route("SPY", use_yfinance=False)
    assert decision.selected_strategy


def test_the_three_engines_are_reachable():
    """All three, from real bars: that is what was broken."""
    router = _router()
    chosen = {
        router.route("CDXS", market_data=_bars(1.40, 0.0)).selected_strategy,
        router.route("SPY", market_data=_bars(760.0, 0.0)).selected_strategy,
        router.route("XPEV", market_data=_alternating(9.25, 0.05)).selected_strategy,
    }
    assert "momentum_breakout" in chosen
    assert "rsi_mean_reversion" in chosen


# ── facts carried on the recommendation ─────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    # Suffixed values are absolute: the HTML scrape path, and model text.
    ("6.50B", 6.5e9), ("$6.50B", 6.5e9), ("950M", 950e6), ("1.2T", 1.2e12),
    ("4,500M", 4.5e9),
    # Bare numbers are Finviz's CSV cell, which is in millions.
    ("3613.63", 3.61363e9), ("4859715.75", 4.85971575e12), (11246.49, 1.124649e10),
    ("", None), ("n/a", None), (None, None), (0, None), ("0", None),
])
def test_market_cap_is_read_from_a_recommendation(raw, expected):
    rec = {"ticker": "X"} if raw is None else {"ticker": "X", "market_cap": raw}
    assert TradeExecutor._rec_market_cap(rec) == pytest.approx(expected) if expected else \
        TradeExecutor._rec_market_cap(rec) == expected


def test_the_dollar_field_wins_when_the_screener_supplies_it():
    """market_cap_usd states its unit; the raw cell does not."""
    rec = {"ticker": "X", "market_cap": "3613.63", "market_cap_usd": 3.61363e9}
    assert TradeExecutor._rec_market_cap(rec) == pytest.approx(3.61363e9)


def test_a_bare_number_no_longer_shrinks_a_company_by_a_million():
    """The regression: CDNA read as a $3.6K company, so it classified small_cap."""
    assert TradeExecutor._rec_market_cap({"market_cap": "3613.63"}) == pytest.approx(3.61363e9)


def test_a_mega_cap_from_the_csv_reaches_large_cap():
    """AAPL's cell is 4859715.75; the router's large_cap threshold is $100B."""
    assert TradeExecutor._rec_market_cap({"market_cap": "4859715.75"}) > 100e9


def test_an_unusable_dollar_field_falls_back_to_the_raw_cell():
    rec = {"market_cap": "3613.63", "market_cap_usd": "junk"}
    assert TradeExecutor._rec_market_cap(rec) == pytest.approx(3.61363e9)


def test_elanco_classifies_as_mid_cap_from_its_csv_cell():
    """ELAN 11246.49 = $11.2B: mid_cap, which nothing could reach before."""
    cap = TradeExecutor._rec_market_cap({"market_cap": "11246.49"})
    profile = StockClassifier(CONFIG).profile_from_market_data(
        "ELAN", _bars(26.0, 0.0), market_cap=cap)
    assert profile.classification == "mid_cap"


def test_apple_classifies_as_large_cap_from_its_csv_cell():
    cap = TradeExecutor._rec_market_cap({"market_cap": "4859715.75"})
    profile = StockClassifier(CONFIG).profile_from_market_data(
        "AAPL", _bars(332.99, 0.0), market_cap=cap)
    assert profile.classification == "large_cap"


@pytest.mark.parametrize("rec,expected", [
    ({"sector": "Technology"}, "Technology"),
    ({"sector": "  Energy  "}, "Energy"),
    ({"sector": ""}, None),
    ({}, None),
    (None, None),
    ("not a dict", None),
])
def test_sector_is_read_from_a_recommendation(rec, expected):
    assert TradeExecutor._rec_sector(rec) == expected


def test_a_screener_supplied_cap_changes_the_route():
    """End to end: the facts the screener now carries affect the decision."""
    router = _router()
    bars = _bars(86.38, 0.0005)
    unknown = router.route("MDT", market_data=bars)
    known = router.route("MDT", market_data=bars,
                         market_cap=TradeExecutor._rec_market_cap({"market_cap": "150B"}))
    assert unknown.classification == "unknown_cap"
    assert known.classification == "large_cap"
