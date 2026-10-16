"""Finviz links beside every symbol the API reports.

The plain value must survive: the risk panel keys cooldown rows by `symbol`,
the order ticket posts it back, and the JSON is read by things that are not
browsers. So links are added as siblings, never substituted.

Symbols reach this from a screener, an LLM and a market feed, so anything that
is not a plausible ticker is left unlinked rather than interpolated into an
href.
"""

import pytest

from falcon_trader import symbol_links

flask = pytest.importorskip("flask")


# ── what counts as a ticker ─────────────────────────────────────────────

@pytest.mark.parametrize("value", ["MSFT", "SPY", "BRK.B", "RDS-A", "F", "QTEX", "msft"])
def test_plausible_tickers_are_linked(value):
    assert symbol_links.is_ticker(value)
    assert symbol_links.finviz_url(value).endswith(f"t={value.strip().upper()}")


@pytest.mark.parametrize("value", [
    "", "   ", None, 42, [], {},
    "a much longer string",
    "MSFT MSFT",
    "<script>alert(1)</script>",
    "javascript:alert(1)",
    "../../etc/passwd",
    "MSFT&t=AAPL",
    "VERYLONGTICKER",
])
def test_anything_else_is_not_linked(value):
    assert symbol_links.is_ticker(value) is False
    assert symbol_links.finviz_url(value) is None
    assert symbol_links.symbol_anchor(value) is None


def test_the_url_is_the_finviz_quote_page():
    """Verified 2026-10-16 to land on the MSFT quote page."""
    assert symbol_links.finviz_url("MSFT") == "https://elite.finviz.com/stock?t=MSFT"


# ── the anchor ──────────────────────────────────────────────────────────

def test_the_anchor_opens_in_a_new_tab_without_window_access():
    anchor = symbol_links.symbol_anchor("MSFT")
    assert 'href="https://elite.finviz.com/stock?t=MSFT"' in anchor
    assert 'target="_blank"' in anchor
    assert 'rel="noopener"' in anchor
    assert ">MSFT</a>" in anchor


def test_custom_link_text_is_escaped():
    anchor = symbol_links.symbol_anchor("MSFT", text="<b>MSFT</b>")
    assert "&lt;b&gt;" in anchor
    assert "<b>" not in anchor


# ── annotation is additive ──────────────────────────────────────────────

def test_the_plain_symbol_is_untouched():
    """The guarantee: consumers keep using `symbol` as an identifier."""
    row = {"symbol": "MSFT", "quantity": 10}
    symbol_links.annotate(row)
    assert row["symbol"] == "MSFT"
    assert row["quantity"] == 10


def test_both_a_url_and_html_are_added():
    row = symbol_links.annotate({"symbol": "MSFT"})
    assert row["symbol_url"] == "https://elite.finviz.com/stock?t=MSFT"
    assert row["symbol_html"].startswith("<a href=")


def test_ticker_keys_are_annotated_too():
    """Recommendations use `ticker`; positions use `symbol`."""
    row = symbol_links.annotate({"ticker": "APLD"})
    assert row["ticker_url"].endswith("t=APLD")
    assert "ticker_html" in row


def test_nested_lists_and_dicts_are_walked():
    payload = {
        "recommendations": [{"ticker": "APLD"}, {"ticker": "NXL"}],
        "kill_switch": {"symbols": [{"symbol": "QSR"}]},
    }
    symbol_links.annotate(payload)
    assert payload["recommendations"][0]["ticker_url"].endswith("t=APLD")
    assert payload["kill_switch"]["symbols"][0]["symbol_url"].endswith("t=QSR")


def test_a_producer_supplied_link_is_not_overwritten():
    row = symbol_links.annotate({"symbol": "MSFT", "symbol_url": "/local/MSFT"})
    assert row["symbol_url"] == "/local/MSFT"


def test_an_unlinkable_symbol_gains_nothing():
    row = symbol_links.annotate({"symbol": "n/a"})
    assert "symbol_url" not in row and "symbol_html" not in row


def test_deep_recursion_is_bounded():
    """A pathological payload must not hang a page."""
    payload = current = {}
    for _ in range(symbol_links.MAX_DEPTH + 5):
        current["next"] = {"symbol": "MSFT"}
        current = current["next"]
    symbol_links.annotate(payload)      # must return, not recurse forever


# ── through Flask ───────────────────────────────────────────────────────

def _app():
    app = flask.Flask(__name__)

    @app.route("/api/positions")
    def positions():
        return flask.jsonify([{"symbol": "QSR", "quantity": 21}])

    @app.route("/api/decorated")
    @symbol_links.with_symbol_links
    def decorated():
        return flask.jsonify({"ticker": "APLD"})

    @app.route("/api/plain-text")
    def plain_text():
        return "MSFT", 200, {"Content-Type": "text/plain"}

    @app.route("/page")
    def page():
        return "<html>MSFT</html>"

    symbol_links.install(app)
    return app


def test_every_api_response_is_annotated():
    body = _app().test_client().get("/api/positions").get_json()
    assert body[0]["symbol"] == "QSR"
    assert body[0]["symbol_url"].endswith("t=QSR")


def test_the_decorator_works_on_its_own():
    body = _app().test_client().get("/api/decorated").get_json()
    assert body["ticker_url"].endswith("t=APLD")


def test_a_non_json_api_response_passes_through():
    response = _app().test_client().get("/api/plain-text")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "MSFT"


def test_pages_outside_the_prefix_are_untouched():
    response = _app().test_client().get("/page")
    assert response.get_data(as_text=True) == "<html>MSFT</html>"
