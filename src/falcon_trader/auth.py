"""Authentication and request gating for the dashboard (falcon-trader#25).

Before this module the Flask app was reachable on the LAN with no
authentication, no CSRF protection, ``CORS(app)`` fully open, bound to
``0.0.0.0:5000``. ``POST /api/strategy/deploy`` wrote caller-supplied Python into
the installed package directory and executed it; ``POST /api/order`` reached a
real broker when ``FALCON_DAS_LIVE=1``. Both were unauthenticated.

Design notes
------------

**Everything here is a pure function.** ``flask`` is not importable in every
environment this gets tested in, and security logic that can only be exercised
through a live request object does not get exercised. The Flask wiring is a thin
adapter in :func:`install` that translates a request into arguments and a
decision back into a response.

**Two credential kinds, one decision.** Browsers get an HttpOnly ``SameSite=Strict``
session cookie; scripts and MCP clients send ``Authorization: Bearer`` or
``X-Falcon-Token``. This matters because all eleven pages under ``www/`` call
``fetch('/api/...')`` same-origin with no headers -- same-origin fetch sends
cookies by default, so the existing UI keeps working with no HTML changes.

**CSRF by Origin check, not a double-submit token.** ``SameSite=Strict`` already
prevents the cookie from riding along on a cross-site request; validating
``Origin``/``Referer`` on state-changing methods closes the remainder. A
double-submit token would mean editing every page.

**Fail closed.** No token configured means the server refuses to start rather
than starting open. That is the whole point of the issue.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time as _time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

__all__ = [
    "AuthConfig",
    "Decision",
    "AuthConfigError",
    "SESSION_COOKIE",
    "SAFE_METHODS",
    "PUBLIC_PATHS",
    "constant_time_equals",
    "issue_session",
    "verify_session",
    "session_nonce",
    "normalize_path",
    "CODE_EXECUTION_PATHS",
    "CODE_EXECUTION_PREFIXES",
    "UNSAFE_GET_PATHS",
    "extract_bearer_token",
    "is_public_path",
    "origin_allowed",
    "decide",
    "load_config",
    "install",
]

#: Name of the browser session cookie.
SESSION_COOKIE = "falcon_session"

#: Methods that do not change state. HEAD/OPTIONS included so CORS preflight
#: and health checks are not gated on a credential.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: Paths reachable with no credential at all. Deliberately tiny.
#: ``/health`` is here so container and gateway health checks keep working;
#: it exposes no account data.
PUBLIC_PATHS = ("/health", "/login", "/favicon.ico")

#: Endpoints that execute caller-supplied code, directly or eventually. Every
#: one of these requires FALCON_ALLOW_DEPLOY on top of authentication.
#:
#: This started as a single hard-coded check on /api/strategy/deploy, which was
#: wrong: /api/strategy/backtest hands `code` to the same
#: exec_module()+subprocess.run() path in strategy_manager.run_backtest() with
#: *no* validation at all, and /api/strategy/rollback restores a file into the
#: installed package directory. Gating one of three was security theatre.
CODE_EXECUTION_PATHS = frozenset({
    "/api/strategy/deploy",
    "/api/strategy/backtest",
    "/api/strategy/rollback",
    "/api/strategy/validate",
})

#: Path prefixes that execute caller- or LLM-supplied code. Same gate as
#: CODE_EXECUTION_PATHS, but matched by prefix because the route carries an id.
#: /api/strategies/youtube/<id>/activate runs LLM-generated strategy code
#: through run_backtest, which is the same exec_module path as deploy.
CODE_EXECUTION_PREFIXES = ("/api/strategies/youtube/",)


def _is_code_execution_path(canonical: str) -> bool:
    """True when the path executes code and needs FALCON_ALLOW_DEPLOY."""
    if canonical in CODE_EXECUTION_PATHS:
        return True
    return any(canonical.startswith(p) and canonical.endswith("/activate")
               for p in CODE_EXECUTION_PREFIXES)

#: GET endpoints that change state. Flask defaults to GET-only, and several
#: handlers that start or stop the trading bot were written without
#: ``methods=``. They are state-changing regardless of verb, so the CSRF check
#: must treat them as unsafe -- otherwise an <img src> on any same-origin page
#: halts the trading bot.
UNSAFE_GET_PATHS = frozenset({
    "/api/bot/start",
    "/api/bot/stop",
})

#: Minimum token length. A short shared secret on a LAN-reachable trading box
#: is not meaningfully better than none.
MIN_TOKEN_LENGTH = 24

#: Cap on remembered logout nonces. Sessions expire on their own, so this only
#: needs to cover the window between a logout and that cookie's expiry.
MAX_REVOKED_NONCES = 4096


class AuthConfigError(RuntimeError):
    """Raised when the auth configuration is missing or unusable."""


@dataclass(frozen=True)
class AuthConfig:
    """Resolved authentication settings."""

    token: str
    allowed_origins: tuple = ()
    allow_deploy: bool = False
    require_live_header: bool = True
    bind_host: str = "127.0.0.1"
    bind_port: int = 5000
    cookie_secure: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.token)


@dataclass(frozen=True)
class Decision:
    """The outcome of gating one request."""

    allowed: bool
    status: int = 200
    error: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def allow(cls) -> "Decision":
        return cls(allowed=True)

    @classmethod
    def deny(cls, status: int, error: str, reason: str) -> "Decision":
        return cls(allowed=False, status=status, error=error, reason=reason)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def issue_session(token: str, ttl_seconds: int = 12 * 3600,
                  now: Optional[int] = None) -> str:
    """Mint a signed session value for the browser cookie.

    The cookie used to be the raw ``FALCON_API_TOKEN``. Three things were wrong
    with that: the permanent global secret was handed to the browser and to
    anything co-resident on the origin (unauthenticated Grafana/Prometheus/
    Consul under the same host and port receive it in their access logs);
    ``/logout`` could not revoke it, because deleting the browser's copy leaves
    the credential valid everywhere; and it never expired.

    The value is ``v1.<expiry>.<nonce>.<hmac>`` where the HMAC is over the
    expiry and nonce keyed by the API token. It is verifiable with no server
    state, expires on its own, and is useless as an API credential -- the API
    still requires the real token.
    """
    now = int(_time.time()) if now is None else now
    expiry = now + int(ttl_seconds)
    nonce = secrets.token_urlsafe(12)
    payload = f"{expiry}.{nonce}"
    signature = hmac.new(
        str(token).encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"v1.{payload}.{signature}"


def verify_session(cookie: Optional[str], token: Optional[str],
                   now: Optional[int] = None,
                   revoked: Optional[set] = None) -> bool:
    """Validate a cookie minted by :func:`issue_session`."""
    if not cookie or not token:
        return False

    parts = str(cookie).split(".")
    if len(parts) != 4 or parts[0] != "v1":
        return False

    _, expiry_raw, nonce, signature = parts

    try:
        expiry = int(expiry_raw)
    except (TypeError, ValueError):
        return False

    now = int(_time.time()) if now is None else now
    if expiry <= now:
        return False

    if revoked is not None and nonce in revoked:
        return False

    expected = hmac.new(
        str(token).encode(), f"{expiry_raw}.{nonce}".encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def session_nonce(cookie: Optional[str]) -> Optional[str]:
    """The nonce inside a session cookie, for revocation on logout."""
    if not cookie:
        return None
    parts = str(cookie).split(".")
    return parts[2] if len(parts) == 4 and parts[0] == "v1" else None


def constant_time_equals(supplied: Optional[str], expected: Optional[str]) -> bool:
    """Compare two secrets without leaking length or content through timing.

    ``==`` on a token short-circuits at the first differing byte, which is
    measurable across a LAN.
    """
    if not supplied or not expected:
        return False
    return hmac.compare_digest(str(supplied), str(expected))


def extract_bearer_token(
    authorization: Optional[str] = None,
    x_falcon_token: Optional[str] = None,
) -> Optional[str]:
    """Pull a token out of either accepted header.

    ``Authorization: Bearer <token>`` is the standard form; ``X-Falcon-Token``
    exists because some clients cannot set ``Authorization`` without triggering
    their own auth handling.
    """
    if authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    if x_falcon_token and x_falcon_token.strip():
        return x_falcon_token.strip()
    return None


def normalize_path(path: str) -> str:
    """Reduce a request path to a single canonical form for matching.

    The endpoint guards compared ``path.rstrip("/")`` against a literal, which
    meant ``//api/strategy/deploy``, ``/api/strategy//deploy`` and
    ``/api/./strategy/deploy`` all missed the check while still passing the
    credential check -- an authenticated bypass of both the deploy guard and
    the live-order guard.

    Collapses repeated slashes, drops ``.`` segments, resolves ``..``, and
    strips the trailing slash. Comparing normalized-to-normalized means a guard
    cannot be dodged by respelling the path, whatever Werkzeug or a proxy in
    front happens to do first.
    """
    if not path:
        return "/"

    segments = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if segments:
                segments.pop()
            continue
        segments.append(part)

    return "/" + "/".join(segments) if segments else "/"


def is_public_path(path: str, public_paths: Sequence[str] = PUBLIC_PATHS) -> bool:
    """True when `path` needs no credential.

    Prefix entries end with ``/``; everything else must match exactly. Matching
    ``/health`` as a prefix would also expose ``/healthcheck-internal`` or any
    future sibling, so exact-match is the default.

    Traversal is rejected outright. Without this, ``/static/../api/strategy/deploy``
    matches the ``/static/`` prefix and is treated as public -- reaching the
    remote-code-execution endpoint with no credential. Werkzeug normally
    normalizes ``..`` out of ``request.path`` before routing, so this is very
    likely unreachable through Flask; the gate must not depend on that being
    true of every server, proxy, and future Werkzeug version in front of it.
    """
    if not path:
        return False

    # Reject traversal and encoded separators before any matching. A path that
    # needs normalizing is not a path this function should be judging.
    lowered = path.lower()
    if ".." in path or "%2e" in lowered or "%2f" in lowered or "\\" in path:
        return False

    for entry in public_paths:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry or path == entry.rstrip("/"):
            return True
    return False


def _same_origin(origin: str, host: Optional[str]) -> bool:
    if not host:
        return False
    netloc = urlsplit(origin).netloc
    return bool(netloc) and netloc == host


def origin_allowed(
    origin: Optional[str],
    referer: Optional[str],
    host: Optional[str],
    allowed_origins: Iterable[str] = (),
) -> bool:
    """CSRF check for a state-changing request.

    A request is acceptable when its ``Origin`` (or, for clients that omit it,
    the origin of its ``Referer``) either matches the host being served or is on
    the configured allow-list.

    A *missing* Origin and Referer is accepted: non-browser clients (curl, the
    canary exporter, MCP tooling) send neither, and those requests carry a
    bearer token rather than a cookie, so they are not a CSRF vector. Browsers
    always send Origin on cross-origin state-changing requests, which is the
    case this is defending against.
    """
    allowed = tuple(allowed_origins or ())

    candidate = origin
    if not candidate and referer:
        parts = urlsplit(referer)
        if parts.scheme and parts.netloc:
            candidate = f"{parts.scheme}://{parts.netloc}"

    if not candidate:
        return True

    if candidate in allowed:
        return True

    return _same_origin(candidate, host)


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def decide(
    *,
    config: AuthConfig,
    method: str,
    path: str,
    authorization: Optional[str] = None,
    x_falcon_token: Optional[str] = None,
    session_cookie: Optional[str] = None,
    origin: Optional[str] = None,
    referer: Optional[str] = None,
    host: Optional[str] = None,
    live_header: Optional[str] = None,
    execution_backend: str = "paper",
    das_live: bool = False,
) -> Decision:
    """Gate one request. Pure: no Flask, no globals, no I/O.

    Order matters. CORS preflight and public paths short-circuit first; then the
    credential; then CSRF; then the endpoint-specific guards that authentication
    alone is not sufficient for.

    All path matching happens on the *normalized* path, so a guard cannot be
    dodged by respelling the URL.
    """
    method = (method or "GET").upper()
    canonical = normalize_path(path)

    # A CORS preflight carries no credentials by design -- the browser strips
    # them -- so gating it on one makes cross-origin requests impossible rather
    # than merely restricted. The preflight reveals nothing and performs no
    # action; the actual request that follows is gated normally.
    if method == "OPTIONS":
        return Decision.allow()

    if is_public_path(canonical):
        return Decision.allow()

    if not config.enabled:
        # Should be unreachable: load_config refuses to build a disabled config.
        # Treated as a denial rather than an allow so a future refactor cannot
        # accidentally reopen the app.
        return Decision.deny(503, "Authentication is not configured", "no_token")

    # --- credential ---
    supplied = extract_bearer_token(authorization, x_falcon_token)
    has_token = constant_time_equals(supplied, config.token)
    has_session = verify_session(session_cookie, config.token)

    if not (has_token or has_session):
        return Decision.deny(401, "Authentication required", "no_credential")

    # --- CSRF, cookie-authenticated state changes only ---
    # UNSAFE_GET_PATHS is why this is not simply `method not in SAFE_METHODS`:
    # /api/bot/start and /api/bot/stop are state-changing GETs.
    state_changing = method not in SAFE_METHODS or canonical in UNSAFE_GET_PATHS
    if state_changing and has_session and not has_token:
        if not origin_allowed(origin, referer, host, config.allowed_origins):
            return Decision.deny(403, "Cross-origin request rejected", "bad_origin")

    # --- code execution surface ---
    # Authentication is necessary but not sufficient. These endpoints hand
    # caller-supplied Python to exec_module()/subprocess, or write files into
    # the installed package. They stay off unless deliberately enabled.
    if _is_code_execution_path(canonical) and method in ("POST", "PUT", "PATCH"):
        if not config.allow_deploy:
            return Decision.deny(
                403,
                f"{canonical} executes caller-supplied code and is disabled. "
                "Set FALCON_ALLOW_DEPLOY=1 to enable.",
                "deploy_disabled",
            )

    # --- real money ---
    # A live broker order needs an explicit per-request opt-in, so a client
    # holding a valid token cannot place one by accident.
    if canonical == "/api/order" and method == "POST":
        if execution_backend == "das" and das_live and config.require_live_header:
            if (live_header or "").strip() != "1":
                return Decision.deny(
                    403,
                    "Live order requires the X-Falcon-Live: 1 header",
                    "live_header_missing",
                )

    return Decision.allow()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def _split_origins(raw: Optional[str]) -> tuple:
    if not raw:
        return ()
    return tuple(o.strip() for o in raw.split(",") if o.strip())


def load_config(env=None) -> AuthConfig:
    """Build an :class:`AuthConfig` from the environment.

    Raises :class:`AuthConfigError` when ``FALCON_API_TOKEN`` is missing or too
    short. Refusing to start is the intended behavior: an unauthenticated start
    is the bug being fixed, so there is no "warn and continue" path.
    """
    env = os.environ if env is None else env

    token = (env.get("FALCON_API_TOKEN") or "").strip()
    if not token:
        raise AuthConfigError(
            "FALCON_API_TOKEN is not set. The dashboard exposes strategy "
            "deployment and order placement and will not start without "
            "authentication. Generate one with:\n"
            "    python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    if len(token) < MIN_TOKEN_LENGTH:
        raise AuthConfigError(
            f"FALCON_API_TOKEN must be at least {MIN_TOKEN_LENGTH} characters "
            f"(got {len(token)})."
        )

    def _flag(name: str, default: str = "0") -> bool:
        return (env.get(name, default) or "").strip().lower() in ("1", "true", "yes", "on")

    return AuthConfig(
        token=token,
        allowed_origins=_split_origins(env.get("FALCON_CORS_ORIGINS")),
        allow_deploy=_flag("FALCON_ALLOW_DEPLOY"),
        require_live_header=not _flag("FALCON_SKIP_LIVE_HEADER"),
        bind_host=(env.get("FALCON_BIND_HOST") or "127.0.0.1").strip(),
        bind_port=int(env.get("FALCON_BIND_PORT") or 5000),
        cookie_secure=_flag("FALCON_COOKIE_SECURE"),
    )


def generate_token() -> str:
    """A token suitable for FALCON_API_TOKEN."""
    return secrets.token_urlsafe(32)


# --------------------------------------------------------------------------
# Flask adapter
# --------------------------------------------------------------------------

def install(app, config: AuthConfig, runtime_config=None):
    """Attach the gate to a Flask app.

    Everything above is pure; this is the only part that knows about Flask. It
    reads the request, calls :func:`decide`, and turns a denial into a response.
    A ``before_request`` hook is used rather than 83 per-route decorators
    precisely because a decorator you forget to add is an open endpoint.
    """
    from flask import jsonify, make_response, redirect, request

    runtime_config = {} if runtime_config is None else runtime_config

    # Nonces revoked by /logout. In-process only: a restart clears it, which is
    # sound because every session it could revoke expires within the cookie TTL
    # anyway. Bounded with a FIFO so a logout loop cannot grow it without limit
    # -- an evicted nonce is at worst a session that stays valid until its own
    # expiry, which is the pre-revocation behaviour.
    _revoked: "OrderedDict[str, None]" = OrderedDict()

    def _revoke(nonce: str) -> None:
        _revoked[nonce] = None
        while len(_revoked) > MAX_REVOKED_NONCES:
            _revoked.popitem(last=False)

    @app.before_request
    def _gate():  # pragma: no cover - exercised through the app, not unit tests
        decision = decide(
            config=config,
            method=request.method,
            path=request.path,
            authorization=request.headers.get("Authorization"),
            x_falcon_token=request.headers.get("X-Falcon-Token"),
            session_cookie=(
                None
                if session_nonce(request.cookies.get(SESSION_COOKIE)) in _revoked
                else request.cookies.get(SESSION_COOKIE)
            ),
            origin=request.headers.get("Origin"),
            referer=request.headers.get("Referer"),
            host=request.headers.get("Host"),
            live_header=request.headers.get("X-Falcon-Live"),
            execution_backend=str(runtime_config.get("execution_backend", "paper")).lower(),
            das_live=os.getenv("FALCON_DAS_LIVE") == "1",
        )

        if decision.allowed:
            return None

        logger.warning(
            "Rejected %s %s from %s: %s",
            request.method, request.path, request.remote_addr, decision.reason,
        )

        # A browser asking for a page gets sent to the login form; anything
        # else -- and anything under /api/ -- gets JSON.
        wants_html = (
            decision.status == 401
            and request.method in SAFE_METHODS
            and not request.path.startswith("/api/")
            and "text/html" in (request.headers.get("Accept") or "")
        )
        if wants_html:
            return redirect("/login")

        response = jsonify({
            "status": "error",
            "error": decision.error,
            "reason": decision.reason,
        })
        response.status_code = decision.status
        if decision.status == 401:
            response.headers["WWW-Authenticate"] = 'Bearer realm="falcon"'
        return response

    @app.route("/login", methods=["GET", "POST"])
    def _login():  # pragma: no cover - trivial adapter
        if request.method == "GET":
            return _LOGIN_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}

        supplied = (request.form.get("token") or "").strip()
        if not constant_time_equals(supplied, config.token):
            logger.warning("Failed login from %s", request.remote_addr)
            return (
                _LOGIN_PAGE.replace(
                    "<!--ERROR-->",
                    '<p class="err">Incorrect token.</p>',
                ),
                401,
                {"Content-Type": "text/html; charset=utf-8"},
            )

        ttl = 60 * 60 * 12
        response = make_response(redirect("/"))
        response.set_cookie(
            SESSION_COOKIE,
            issue_session(config.token, ttl_seconds=ttl),
            httponly=True,
            samesite="Strict",
            secure=config.cookie_secure,
            max_age=ttl,
            path="/",
        )
        return response

    @app.route("/logout", methods=["GET", "POST"])
    def _logout():  # pragma: no cover - trivial adapter
        # Revoke server-side, not just in the browser. The old cookie was the
        # API token itself, so deleting the browser copy revoked nothing.
        nonce = session_nonce(request.cookies.get(SESSION_COOKIE))
        if nonce:
            _revoke(nonce)
        response = make_response(redirect("/login"))
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    return app


_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Falcon &middot; Sign in</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0f1115; color: #e6e6e6;
         display: flex; min-height: 100vh; align-items: center; justify-content: center;
         margin: 0; }
  form { background: #1a1d23; padding: 2rem; border-radius: 10px; width: min(360px, 90vw);
         border: 1px solid #2a2e37; }
  h1 { font-size: 1.1rem; margin: 0 0 1.25rem; font-weight: 600; }
  input { width: 100%; box-sizing: border-box; padding: .6rem .7rem; border-radius: 6px;
          border: 1px solid #2a2e37; background: #0f1115; color: #e6e6e6; font-size: 1rem; }
  button { width: 100%; margin-top: .9rem; padding: .6rem; border: 0; border-radius: 6px;
           background: #2f6feb; color: #fff; font-size: 1rem; cursor: pointer; }
  .err { color: #ff6b6b; font-size: .85rem; margin: .6rem 0 0; }
  .hint { color: #8b93a1; font-size: .8rem; margin: .9rem 0 0; }
</style>
</head>
<body>
<form method="post" action="/login">
  <h1>Falcon dashboard</h1>
  <input type="password" name="token" placeholder="API token" autofocus autocomplete="current-password">
  <button type="submit">Sign in</button>
  <!--ERROR-->
  <p class="hint">Uses the value of FALCON_API_TOKEN.</p>
</form>
</body>
</html>"""
