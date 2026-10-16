"""Finviz links for every symbol the platform reports.

A ticker on a page is a dead end: you read "MSFT" and then go and look it up
somewhere else. Every symbol the API returns now carries a link to its Finviz
quote page alongside it.

**Added, never substituted.** `symbol` keeps its plain value and gains
`symbol_url` and `symbol_html` siblings. Rewriting `symbol` into markup would
break every consumer that uses it as an identifier -- the risk panel keys
cooldown rows by it, the order ticket posts it back, and the JSON is read by
things that are not browsers.

Two ways in:

* ``@with_symbol_links`` on a single view, when only that endpoint should carry
  links;
* ``install(app)``, which annotates every JSON response under /api/ -- one
  registration rather than a decorator on each of eighty routes.

Symbols are validated before they are linked. They arrive from a screener, an
LLM and a market feed, and a value that is not a plausible ticker is left
alone rather than interpolated into an href.
"""

import html
import json
import re
from functools import wraps
from typing import Any, Optional

#: Finviz's quote page. elite.finviz.com redirects to the Elite view when the
#: reader's session cookie is present and to the public page otherwise, so one
#: URL serves both. Verified 2026-10-16: it lands on
#: finviz.com/stock?t=MSFT -- "MSFT - Microsoft Corp Stock Price and Quote".
FINVIZ_QUOTE_URL = "https://elite.finviz.com/stock?t={symbol}"

#: Keys whose value is a ticker.
SYMBOL_KEYS = ("symbol", "ticker")

#: What a ticker may look like: letters, digits, dot, hyphen. Covers BRK.B and
#: RDS-A. Anything else -- a sentence, a URL, markup -- is not linked.
_TICKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")

#: Depth limit for walking a payload. Deeper than any response the dashboard
#: produces, shallow enough that a pathological structure cannot hang a page.
MAX_DEPTH = 8


def is_ticker(value: Any) -> bool:
    """True when `value` is plausibly a ticker and safe to put in a URL."""
    return isinstance(value, str) and bool(_TICKER.match(value.strip()))


def finviz_url(symbol: Any) -> Optional[str]:
    """The Finviz quote URL for `symbol`, or None if it is not a ticker."""
    if not is_ticker(symbol):
        return None
    return FINVIZ_QUOTE_URL.format(symbol=symbol.strip().upper())


def symbol_anchor(symbol: Any, text: Optional[str] = None) -> Optional[str]:
    """An <a> element for `symbol`, escaped. None if it is not a ticker.

    rel="noopener" because the link opens in a new tab, and both the symbol and
    the link text are escaped: a ticker is data from a feed, not a literal.
    """
    url = finviz_url(symbol)
    if url is None:
        return None
    label = html.escape(str(text if text is not None else symbol).strip())
    return (f'<a href="{html.escape(url, quote=True)}" target="_blank" '
            f'rel="noopener" class="ticker-link">{label}</a>')


def annotate(payload: Any, keys=SYMBOL_KEYS, _depth: int = 0) -> Any:
    """Add `<key>_url` and `<key>_html` beside every symbol key, in place.

    Returns the same object it was given. An existing `_url`/`_html` value is
    left alone, so a producer that builds its own link keeps it.
    """
    if _depth > MAX_DEPTH:
        return payload

    if isinstance(payload, dict):
        for key in list(payload.keys()):
            if key in keys:
                url = finviz_url(payload[key])
                if url is not None:
                    payload.setdefault(f"{key}_url", url)
                    payload.setdefault(f"{key}_html", symbol_anchor(payload[key]))
        for value in payload.values():
            annotate(value, keys, _depth + 1)
        return payload

    if isinstance(payload, list):
        for item in payload:
            annotate(item, keys, _depth + 1)
        return payload

    return payload


def _annotate_response(response):
    """Annotate a Flask JSON response in place. Anything else passes through."""
    if not getattr(response, "is_json", False):
        return response
    try:
        body = response.get_json(silent=True)
    except Exception:          # pragma: no cover - defensive
        return response
    if body is None:
        return response
    response.set_data(json.dumps(annotate(body)))
    return response


def with_symbol_links(view):
    """Decorator: add symbol links to one view's JSON response."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        from flask import make_response

        result = view(*args, **kwargs)
        # A view may return a Response, a (body, status) tuple, or a dict.
        if isinstance(result, tuple):
            response = make_response(*result)
            return _annotate_response(response)
        if isinstance(result, (dict, list)):
            return annotate(result)
        return _annotate_response(result)

    return wrapper


def install(app, prefix: str = "/api/"):
    """Annotate every JSON response under `prefix`.

    One registration instead of a decorator on every route, so an endpoint
    added later carries links without anyone remembering to opt in.
    """
    from flask import request

    @app.after_request
    def _add_symbol_links(response):       # pragma: no cover - exercised via app
        if request.path.startswith(prefix):
            return _annotate_response(response)
        return response

    return app
