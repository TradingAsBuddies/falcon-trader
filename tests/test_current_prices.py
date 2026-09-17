"""Tests for PaperTradingBot.get_current_prices.

Called on a stub rather than a constructed bot, so there is no database or
network. The method only touches `symbols`, `market_data` and `get_quote`.
"""

import threading
import time

from falcon_trader.paper_trading_bot import PaperTradingBot, QUOTE_WORKERS


class _Stub:
    def __init__(self, quotes, market_data=None, symbols=(), delay=0.0):
        self._quotes = quotes
        self.market_data = market_data or {}
        self.symbols = list(symbols)
        self._delay = delay
        self.calls = []
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def get_quote(self, symbol):
        with self._lock:
            self.calls.append(symbol)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._delay:
                time.sleep(self._delay)
            return self._quotes.get(symbol)
        finally:
            with self._lock:
                self.in_flight -= 1


def _prices(stub, symbols=None):
    return PaperTradingBot.get_current_prices(stub, symbols)


def test_returns_quoted_prices():
    stub = _Stub({"AAA": {"price": 10.0}, "BBB": {"price": 20.5}})
    assert _prices(stub, ["AAA", "BBB"]) == {"AAA": 10.0, "BBB": 20.5}


def test_symbol_without_a_quote_is_absent_not_zero():
    """Absence is what lets callers mark a position stale (falcon-trader#21)."""
    stub = _Stub({"AAA": {"price": 10.0}, "NOPE": None, "ZERO": {"price": 0}})
    assert _prices(stub, ["AAA", "NOPE", "ZERO"]) == {"AAA": 10.0}


def test_cached_symbols_are_not_requoted():
    stub = _Stub({"BBB": {"price": 20.0}}, market_data={"AAA": {"price": 9.99}})
    assert _prices(stub, ["AAA", "BBB"]) == {"AAA": 9.99, "BBB": 20.0}
    assert stub.calls == ["BBB"]


def test_defaults_to_the_watchlist():
    stub = _Stub({"SPY": {"price": 760.0}}, symbols=["SPY"])
    assert _prices(stub) == {"SPY": 760.0}


def test_empty_request_makes_no_calls():
    stub = _Stub({})
    assert _prices(stub, []) == {}
    assert stub.calls == []


def test_quotes_run_concurrently():
    """The regression: ten symbols quoted one at a time took ~5s.

    With a per-call delay, a sequential loop never has more than one request
    in flight and takes n * delay.
    """
    symbols = [f"S{i}" for i in range(10)]
    stub = _Stub({s: {"price": 1.0} for s in symbols}, delay=0.2)

    started = time.monotonic()
    result = _prices(stub, symbols)
    elapsed = time.monotonic() - started

    assert len(result) == 10
    assert stub.max_in_flight > 1
    assert elapsed < 10 * 0.2 / 2, f"took {elapsed:.2f}s; sequential would be 2.0s"


def test_concurrency_is_bounded():
    """A large book must not open one request per position at once."""
    symbols = [f"S{i}" for i in range(QUOTE_WORKERS * 3)]
    stub = _Stub({s: {"price": 1.0} for s in symbols}, delay=0.05)
    _prices(stub, symbols)
    assert stub.max_in_flight <= QUOTE_WORKERS


def test_result_order_does_not_depend_on_completion_order():
    """Faster quotes finishing first must not misassign prices to symbols."""
    class _Varied(_Stub):
        def get_quote(self, symbol):
            time.sleep({"A": 0.15, "B": 0.0, "C": 0.08}[symbol])
            return {"price": {"A": 1.0, "B": 2.0, "C": 3.0}[symbol]}

    assert _prices(_Varied({}), ["A", "B", "C"]) == {"A": 1.0, "B": 2.0, "C": 3.0}
