"""Tests for falcon_trader.auth (falcon-trader#25).

No Flask import anywhere in this file. That is deliberate and is why the module
under test keeps its decision logic in pure functions: security logic that can
only be exercised through a live request object tends not to be exercised.
"""

import pytest

from falcon_trader.auth import (
    SESSION_COOKIE,
    AuthConfig,
    AuthConfigError,
    constant_time_equals,
    decide,
    extract_bearer_token,
    generate_token,
    is_public_path,
    load_config,
    origin_allowed,
)

TOKEN = "s" * 40
OTHER = "x" * 40


def cfg(**kw):
    base = dict(token=TOKEN, allowed_origins=(), allow_deploy=False,
                require_live_header=True, bind_host="127.0.0.1", bind_port=5000)
    base.update(kw)
    return AuthConfig(**base)


def call(**kw):
    params = dict(config=cfg(), method="GET", path="/api/account")
    params.update(kw)
    return decide(**params)


# --------------------------------------------------------------------------
# the headline: the endpoints that made this a critical issue
# --------------------------------------------------------------------------

def test_unauthenticated_deploy_is_rejected():
    """POST /api/strategy/deploy wrote arbitrary Python into site-packages."""
    d = call(method="POST", path="/api/strategy/deploy")
    assert d.allowed is False
    assert d.status == 401


def test_unauthenticated_order_is_rejected():
    d = call(method="POST", path="/api/order")
    assert d.allowed is False
    assert d.status == 401


def test_authenticated_deploy_still_blocked_by_default():
    """A valid token is necessary but not sufficient for the RCE endpoint."""
    d = call(method="POST", path="/api/strategy/deploy",
             authorization=f"Bearer {TOKEN}")
    assert d.allowed is False
    assert d.status == 403
    assert d.reason == "deploy_disabled"


def test_deploy_allowed_only_with_explicit_opt_in():
    d = decide(config=cfg(allow_deploy=True), method="POST",
               path="/api/strategy/deploy", authorization=f"Bearer {TOKEN}")
    assert d.allowed is True


def test_live_order_requires_the_live_header():
    d = call(method="POST", path="/api/order", authorization=f"Bearer {TOKEN}",
             execution_backend="das", das_live=True)
    assert d.allowed is False
    assert d.reason == "live_header_missing"


def test_live_order_allowed_with_the_live_header():
    d = call(method="POST", path="/api/order", authorization=f"Bearer {TOKEN}",
             execution_backend="das", das_live=True, live_header="1")
    assert d.allowed is True


def test_paper_order_does_not_need_the_live_header():
    d = call(method="POST", path="/api/order", authorization=f"Bearer {TOKEN}",
             execution_backend="paper", das_live=False)
    assert d.allowed is True


def test_das_dry_run_does_not_need_the_live_header():
    """Backend is DAS but FALCON_DAS_LIVE is not set -- no real money."""
    d = call(method="POST", path="/api/order", authorization=f"Bearer {TOKEN}",
             execution_backend="das", das_live=False)
    assert d.allowed is True


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def test_bearer_token_accepted():
    assert call(authorization=f"Bearer {TOKEN}").allowed is True


def test_bearer_is_case_insensitive_on_the_scheme():
    assert call(authorization=f"bearer {TOKEN}").allowed is True


def test_x_falcon_token_header_accepted():
    assert call(x_falcon_token=TOKEN).allowed is True


def test_session_cookie_accepted():
    assert call(session_cookie=TOKEN).allowed is True


def test_wrong_token_rejected():
    d = call(authorization=f"Bearer {OTHER}")
    assert d.allowed is False
    assert d.status == 401


def test_empty_bearer_rejected():
    assert call(authorization="Bearer ").allowed is False


def test_malformed_authorization_rejected():
    assert call(authorization=TOKEN).allowed is False          # no scheme
    assert call(authorization=f"Basic {TOKEN}").allowed is False


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_state_changing_methods_need_a_credential(method):
    d = call(method=method, path="/api/screener/profiles/1")
    assert d.allowed is False
    assert d.status == 401


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_state_changing_methods_pass_with_a_token(method):
    d = call(method=method, path="/api/screener/profiles/1",
             authorization=f"Bearer {TOKEN}")
    assert d.allowed is True


# --------------------------------------------------------------------------
# public paths
# --------------------------------------------------------------------------

def test_health_is_public():
    """Container and gateway health checks must not need a credential."""
    assert call(path="/health").allowed is True


def test_login_is_public():
    assert call(path="/login").allowed is True


def test_health_prefix_does_not_leak_to_siblings():
    """Exact match, so a future /healthcheck-internal is not public."""
    assert is_public_path("/health") is True
    assert is_public_path("/healthcheck-internal") is False
    assert is_public_path("/health/detail") is False


def test_static_is_a_prefix_match():
    assert is_public_path("/static/app.css") is True


@pytest.mark.parametrize("path", [
    "/static/../api/strategy/deploy",
    "/static/..%2fapi/strategy/deploy",
    "/static/%2e%2e/api/strategy/deploy",
    "/static/%2E%2E/api/strategy/deploy",
    "/static/..\\api/strategy/deploy",
    "/health/../api/order",
])
def test_traversal_out_of_a_public_prefix_is_not_public(path):
    """Found by probing: /static/ is a prefix rule, so /static/../api/... matched.

    Werkzeug normalizes `..` out of request.path before routing, so this was
    very likely unreachable through Flask -- but the gate must not depend on
    that being true of every proxy and Werkzeug version in front of it.
    """
    assert is_public_path(path) is False


@pytest.mark.parametrize("path", [
    "/static/../api/strategy/deploy",
    "/static/%2e%2e/api/strategy/deploy",
])
def test_traversal_cannot_reach_deploy_unauthenticated(path):
    d = call(method="POST", path=path)
    assert d.allowed is False
    assert d.status == 401


@pytest.mark.parametrize("path", [
    "//api/strategy/deploy",
    "/API/STRATEGY/DEPLOY",
    "/api/strategy/deploy%00",
    "/api/strategy/deploy/",
])
def test_path_mangling_does_not_bypass_the_gate(path):
    assert call(method="POST", path=path).allowed is False


def test_dashboard_pages_are_not_public():
    for path in ("/", "/dashboard", "/trading", "/advisor", "/orchestrator"):
        assert call(path=path).allowed is False, f"{path} must not be public"


def test_api_routes_are_not_public():
    for path in ("/api/account", "/api/positions", "/api/trades",
                 "/api/advisor/proposals", "/api/das/account"):
        assert call(path=path).allowed is False, f"{path} must not be public"


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------

def test_cookie_auth_cross_origin_state_change_is_rejected():
    """The CORS(app) + cookie combination this replaces was the CSRF hole."""
    d = call(method="POST", path="/api/order", session_cookie=TOKEN,
             origin="http://evil.example", host="192.168.1.17:5000")
    assert d.allowed is False
    assert d.reason == "bad_origin"


def test_cookie_auth_same_origin_state_change_is_allowed():
    d = call(method="POST", path="/api/order", session_cookie=TOKEN,
             origin="http://192.168.1.17:5000", host="192.168.1.17:5000")
    assert d.allowed is True


def test_cookie_auth_allowlisted_origin_is_allowed():
    d = decide(config=cfg(allowed_origins=("http://falcon.localhost",)),
               method="POST", path="/api/order", session_cookie=TOKEN,
               origin="http://falcon.localhost", host="192.168.1.17:5000")
    assert d.allowed is True


def test_bearer_token_is_not_subject_to_the_origin_check():
    """A token cannot be replayed by a browser the way a cookie can."""
    d = call(method="POST", path="/api/order", authorization=f"Bearer {TOKEN}",
             origin="http://evil.example", host="192.168.1.17:5000")
    assert d.allowed is True


def test_cookie_auth_safe_method_ignores_origin():
    d = call(method="GET", path="/api/account", session_cookie=TOKEN,
             origin="http://evil.example", host="192.168.1.17:5000")
    assert d.allowed is True


def test_referer_used_when_origin_absent():
    d = call(method="POST", path="/api/order", session_cookie=TOKEN,
             referer="http://evil.example/page", host="192.168.1.17:5000")
    assert d.allowed is False


def test_missing_origin_and_referer_is_allowed():
    """curl and the canary exporter send neither and carry a token, not a cookie."""
    assert origin_allowed(None, None, "192.168.1.17:5000") is True


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def test_constant_time_equals_rejects_empty_and_none():
    assert constant_time_equals(None, TOKEN) is False
    assert constant_time_equals("", TOKEN) is False
    assert constant_time_equals(TOKEN, None) is False
    assert constant_time_equals(None, None) is False


def test_constant_time_equals_matches():
    assert constant_time_equals(TOKEN, TOKEN) is True
    assert constant_time_equals(TOKEN, TOKEN + "x") is False


def test_extract_bearer_token_forms():
    assert extract_bearer_token(f"Bearer {TOKEN}", None) == TOKEN
    assert extract_bearer_token(None, TOKEN) == TOKEN
    assert extract_bearer_token(f"  Bearer   {TOKEN}  ", None) == TOKEN
    assert extract_bearer_token(None, None) is None
    assert extract_bearer_token("Bearer", None) is None


def test_authorization_wins_over_x_falcon_token():
    assert extract_bearer_token(f"Bearer {TOKEN}", OTHER) == TOKEN


# --------------------------------------------------------------------------
# configuration -- fail closed
# --------------------------------------------------------------------------

def test_missing_token_refuses_to_build_a_config():
    with pytest.raises(AuthConfigError) as excinfo:
        load_config({})
    assert "FALCON_API_TOKEN" in str(excinfo.value)


def test_blank_token_refuses():
    with pytest.raises(AuthConfigError):
        load_config({"FALCON_API_TOKEN": "   "})


def test_short_token_refuses():
    with pytest.raises(AuthConfigError) as excinfo:
        load_config({"FALCON_API_TOKEN": "short"})
    assert "at least" in str(excinfo.value)


def test_bind_host_defaults_to_loopback():
    """0.0.0.0 on a LAN-reachable trading box was half the issue."""
    assert load_config({"FALCON_API_TOKEN": TOKEN}).bind_host == "127.0.0.1"


def test_bind_host_is_overridable():
    c = load_config({"FALCON_API_TOKEN": TOKEN, "FALCON_BIND_HOST": "0.0.0.0"})
    assert c.bind_host == "0.0.0.0"


def test_cors_origins_default_to_empty():
    assert load_config({"FALCON_API_TOKEN": TOKEN}).allowed_origins == ()


def test_cors_origins_parsed_from_csv():
    c = load_config({
        "FALCON_API_TOKEN": TOKEN,
        "FALCON_CORS_ORIGINS": "http://a.local, http://b.local ,",
    })
    assert c.allowed_origins == ("http://a.local", "http://b.local")


def test_deploy_disabled_by_default():
    assert load_config({"FALCON_API_TOKEN": TOKEN}).allow_deploy is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_deploy_flag_truthy_forms(raw):
    c = load_config({"FALCON_API_TOKEN": TOKEN, "FALCON_ALLOW_DEPLOY": raw})
    assert c.allow_deploy is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "", "off", "maybe"])
def test_deploy_flag_falsy_forms(raw):
    c = load_config({"FALCON_API_TOKEN": TOKEN, "FALCON_ALLOW_DEPLOY": raw})
    assert c.allow_deploy is False


def test_generated_token_passes_the_length_floor():
    token = generate_token()
    assert load_config({"FALCON_API_TOKEN": token}).token == token


def test_disabled_config_denies_rather_than_allows():
    """Belt and braces: a config with no token must never fall open."""
    d = decide(config=cfg(token=""), method="GET", path="/api/account")
    assert d.allowed is False
    assert d.status == 503


def test_session_cookie_name_is_stable():
    """The Flask adapter and any future gateway config depend on this name."""
    assert SESSION_COOKIE == "falcon_session"
