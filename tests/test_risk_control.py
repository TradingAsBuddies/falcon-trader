"""Halt and resume from the dashboard, through the real Flask routes.

/api/risk/status once shipped returning 500 while every test passed, because
nothing exercised the route. These tests go through a Flask test client, and
the last group installs the real auth gate in front of the blueprint.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

flask = pytest.importorskip("flask")

from falcon_trader import auth, risk_control
from falcon_trader.orchestrator.utils.timezone import ET
from falcon_trader.risk_limits import KillSwitch
from falcon_trader.trading_guards import CooldownPolicy

LOCAL_MOUNTINFO = "1 0 0:1 / / rw - overlay o rw\n"


def _shared_mountinfo(halt_file):
    """mountinfo with the halt file's directory mounted, as the control volume is."""
    return LOCAL_MOUNTINFO + f"2 1 0:2 / {halt_file.parent} rw - btrfs x rw\n"


class _FakeDB:
    def execute(self, query, params=None, fetch=None):
        if "FROM positions" in query:
            return [{"symbol": "AAPL"}]
        return []


def _state(db, symbol):
    return SimpleNamespace(symbol=symbol, last_exit_at=None, last_exit_pnl=None,
                           loss_streak=0, round_trips_today=0)


def _app(halt_file, env=None, mountinfo=None, bot=True):
    fake_bot = SimpleNamespace(
        kill_switch=KillSwitch(halt_file=halt_file, env=env if env is not None else {}),
        cooldown_policy=CooldownPolicy(),
    ) if bot else None
    if mountinfo is None:
        mountinfo = _shared_mountinfo(halt_file)
    app = flask.Flask(__name__)
    app.register_blueprint(risk_control.create_blueprint(
        get_bot=lambda: fake_bot, get_db=_FakeDB, load_state=_state,
        mountinfo=lambda: mountinfo,
    ))
    return app


@pytest.fixture
def halt_file(tmp_path):
    return tmp_path / "control" / "TRADING_HALTED"


# ── clean_reason ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "", "   ", 42])
def test_reason_is_required(raw):
    with pytest.raises(ValueError):
        risk_control.clean_reason(raw)


def test_reason_newlines_are_collapsed():
    """The halt file is line-oriented; a newline would forge the source line."""
    assert risk_control.clean_reason("bad\nfills\r\nsource: x") == "bad fills source: x"


def test_reason_length_is_bounded():
    risk_control.clean_reason("x" * risk_control.MAX_REASON_LENGTH)
    with pytest.raises(ValueError):
        risk_control.clean_reason("x" * (risk_control.MAX_REASON_LENGTH + 1))


# ── file operations ─────────────────────────────────────────────────────

def test_write_halt_is_honored_by_a_separate_kill_switch(halt_file):
    """The whole point: another process's KillSwitch sees it."""
    assert risk_control.write_halt(halt_file, "manual") is True
    other_process = KillSwitch(halt_file=halt_file, env={})
    assert other_process.is_trading_enabled() is False


def test_second_halt_keeps_the_first_reason(halt_file):
    risk_control.write_halt(halt_file, "first")
    assert risk_control.write_halt(halt_file, "second") is False
    assert risk_control.read_halt_note(halt_file)["reason"] == "first"


def test_halt_note_round_trip(halt_file):
    risk_control.write_halt(halt_file, "daily loss hit")
    note = risk_control.read_halt_note(halt_file)
    assert note["reason"] == "daily loss hit"
    assert note["source"] == "source: dashboard"
    assert note["halted_at"]


def test_halt_time_is_eastern_not_container_utc(halt_file):
    """The regression: the container clock is UTC and the note said 12:37 for 08:37 ET."""
    utc = dt.datetime(2026, 9, 17, 12, 37, 44, tzinfo=dt.timezone.utc)
    risk_control.write_halt(halt_file, "x", now=utc.astimezone(ET))
    note = risk_control.read_halt_note(halt_file)
    assert note["halted_at"].endswith("-04:00")
    assert note["halted_at_display"] == "2026-09-17 08:37:44 ET"


def test_default_halt_time_carries_an_offset(halt_file):
    risk_control.write_halt(halt_file, "x")
    assert dt.datetime.fromisoformat(risk_control.read_halt_note(halt_file)["halted_at"]).tzinfo is not None


@pytest.mark.parametrize("raw,expected", [
    ("2026-09-17T12:37:44+00:00", "2026-09-17 08:37:44 ET"),
    ("2026-12-01T15:00:00+00:00", "2026-12-01 10:00:00 ET"),   # EST, not EDT
    ("2026-09-17T12:37:44.966480", "2026-09-17 12:37:44 (timezone not recorded)"),
    ("not a time", "not a time"),
    (None, None),
])
def test_format_halted_at(raw, expected):
    """A naive timestamp (KillSwitch.halt, or by hand) is labelled, never assumed ET."""
    assert risk_control.format_halted_at(raw) == expected


def test_note_records_no_client_address(halt_file):
    """Every request arrives from the podman gateway; an address here would mislead."""
    app = _app(halt_file)
    app.test_client().post("/api/risk/halt", json={"reason": "x"},
                           environ_base={"REMOTE_ADDR": "10.89.0.133"})
    assert "10.89.0.133" not in halt_file.read_text()


def test_halt_note_of_hand_touched_file(halt_file):
    """`touch TRADING_HALTED` from a shell is still a valid halt."""
    halt_file.parent.mkdir(parents=True)
    halt_file.touch()
    assert risk_control.read_halt_note(halt_file) == {
        "halted_at": None, "halted_at_display": None, "reason": None, "source": None}


def test_unwritable_directory_raises_instead_of_reporting_success(tmp_path):
    blocker = tmp_path / "control"
    blocker.write_text("a file where the directory should be")
    with pytest.raises(risk_control.HaltWriteError):
        risk_control.write_halt(blocker / "TRADING_HALTED", "x")


def test_remove_halt_reports_presence(halt_file):
    assert risk_control.remove_halt(halt_file) is False
    risk_control.write_halt(halt_file, "x")
    assert risk_control.remove_halt(halt_file) is True
    assert not halt_file.exists()


# ── routes ──────────────────────────────────────────────────────────────

def test_status_route_returns_200_with_halt_note(halt_file):
    risk_control.write_halt(halt_file, "why")
    resp = _app(halt_file).test_client().get("/api/risk/status")
    assert resp.status_code == 200
    ks = resp.get_json()["kill_switch"]
    assert ks["trading_enabled"] is False
    assert ks["halt_note"]["reason"] == "why"
    assert ks["halt_file_on_shared_mount"] is True
    assert [s["symbol"] for s in resp.get_json()["symbols"]] == ["AAPL"]


def test_status_route_before_bot_init_is_503(halt_file):
    assert _app(halt_file, bot=False).test_client().get("/api/risk/status").status_code == 503


def test_halt_route_halts(halt_file):
    resp = _app(halt_file).test_client().post("/api/risk/halt", json={"reason": "stop now"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "halted"
    assert body["kill_switch"]["trading_enabled"] is False
    assert halt_file.exists()


def test_halt_route_twice_reports_already_halted(halt_file):
    client = _app(halt_file).test_client()
    client.post("/api/risk/halt", json={"reason": "one"})
    body = client.post("/api/risk/halt", json={"reason": "two"}).get_json()
    assert body["status"] == "already_halted"
    assert body["kill_switch"]["halt_note"]["reason"] == "one"


def test_halt_route_requires_reason(halt_file):
    resp = _app(halt_file).test_client().post("/api/risk/halt", json={})
    assert resp.status_code == 400
    assert not halt_file.exists()


def test_halt_route_carries_reach_warning_when_not_shared(halt_file):
    """A 200 must not read as 'the trader stopped' when it cannot see the file."""
    body = _app(halt_file, mountinfo=LOCAL_MOUNTINFO).test_client().post(
        "/api/risk/halt", json={"reason": "x"}).get_json()
    assert body["kill_switch"]["halt_file_on_shared_mount"] is False
    assert "not visible to the trader" in body["kill_switch"]["reach_warning"]


def test_halt_route_failure_is_500_not_200(tmp_path):
    blocker = tmp_path / "control"
    blocker.write_text("not a directory")
    resp = _app(blocker / "TRADING_HALTED").test_client().post("/api/risk/halt", json={"reason": "x"})
    assert resp.status_code == 500
    assert "NOT written" in resp.get_json()["error"]


def test_resume_requires_confirmation(halt_file):
    risk_control.write_halt(halt_file, "x")
    client = _app(halt_file).test_client()
    for payload in (None, {}, {"confirm": True}, {"confirm": "yes"}):
        resp = client.post("/api/risk/resume", json=payload)
        assert resp.status_code == 400
    assert halt_file.exists()


def test_resume_route_resumes(halt_file):
    risk_control.write_halt(halt_file, "x")
    body = _app(halt_file).test_client().post("/api/risk/resume", json={"confirm": "resume"}).get_json()
    assert body["status"] == "resumed"
    assert body["kill_switch"]["trading_enabled"] is True
    assert not halt_file.exists()


def test_resume_when_not_halted(halt_file):
    body = _app(halt_file).test_client().post("/api/risk/resume", json={"confirm": "resume"}).get_json()
    assert body["status"] == "not_halted"


def test_resume_cannot_override_the_env_flag(halt_file):
    """FALCON_TRADING_ENABLED=0 is a deploy-time decision the dashboard cannot undo."""
    risk_control.write_halt(halt_file, "x")
    app = _app(halt_file, env={"FALCON_TRADING_ENABLED": "0"})
    body = app.test_client().post("/api/risk/resume", json={"confirm": "resume"}).get_json()
    assert body["status"] == "still_halted"
    assert body["kill_switch"]["trading_enabled"] is False
    assert "FALCON_TRADING_ENABLED" in body["error"]


# ── behind the real auth gate ───────────────────────────────────────────

TOKEN = "t" * 40


def _gated_app(halt_file):
    app = _app(halt_file)
    auth.install(app, auth.AuthConfig(token=TOKEN))
    return app


@pytest.mark.parametrize("path,payload", [
    ("/api/risk/halt", {"reason": "x"}),
    ("/api/risk/resume", {"confirm": "resume"}),
])
def test_control_routes_require_a_credential(halt_file, path, payload):
    resp = _gated_app(halt_file).test_client().post(path, json=payload)
    assert resp.status_code == 401
    assert not halt_file.exists()


def test_cross_origin_cookie_halt_is_refused(halt_file):
    """A browser session is the CSRF vector; a bearer token is not."""
    client = _gated_app(halt_file).test_client()
    client.set_cookie(auth.SESSION_COOKIE, auth.issue_session(TOKEN))
    resp = client.post("/api/risk/halt", json={"reason": "x"},
                       headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403
    assert not halt_file.exists()


def test_same_origin_cookie_halt_succeeds(halt_file):
    client = _gated_app(halt_file).test_client()
    client.set_cookie(auth.SESSION_COOKIE, auth.issue_session(TOKEN))
    resp = client.post("/api/risk/halt", json={"reason": "x"},
                       headers={"Origin": "http://localhost"})
    assert resp.status_code == 200
    assert halt_file.exists()


@pytest.mark.parametrize("payload", [[], "halt", 7])
def test_non_object_json_is_400_not_500(halt_file, payload):
    client = _app(halt_file).test_client()
    assert client.post("/api/risk/halt", json=payload).status_code == 400
    assert client.post("/api/risk/resume", json=payload).status_code == 400


def test_status_requires_a_credential(halt_file):
    assert _gated_app(halt_file).test_client().get("/api/risk/status").status_code == 401


def test_authenticated_halt_succeeds(halt_file):
    resp = _gated_app(halt_file).test_client().post(
        "/api/risk/halt", json={"reason": "x"}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert halt_file.exists()
