"""Halt and resume trading from the dashboard, and the risk status route.

The halt file is the one kill-switch source that crosses a process boundary:
the environment flag is fixed at container start and an in-process halt lives
in one interpreter. So "halt from the dashboard" means writing that file, and
it only reaches the trading engine when both containers mount the directory it
lives in. The status route reports whether they do; halt and resume report it
again in their responses, so nobody reads a 200 as "the trader stopped" when it
did not.

Written as a blueprint over injected providers so the routes themselves are
tested. /api/risk/status first shipped inline in dashboard_server, untested,
and returned 500 in production while the suite was green.

Two deliberate asymmetries:

* Halting never overwrites an existing halt. The first reason stands; a second
  click must not erase why trading was stopped.
* Resume removes only the file. It cannot clear FALCON_TRADING_ENABLED=0, and
  the response says so when that flag is still holding trading off.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from falcon_trader import risk_status
from falcon_trader.orchestrator.utils.timezone import ET, now_et

logger = logging.getLogger(__name__)

#: Longest accepted halt reason, after whitespace is collapsed.
MAX_REASON_LENGTH = 200

#: The literal a resume request must carry. A stray POST, or a retry loop in
#: some client, should not be able to re-enable trading.
RESUME_CONFIRMATION = "resume"


class HaltWriteError(RuntimeError):
    """The halt file could not be written or removed as asked."""


def clean_reason(raw: Any) -> str:
    """A single-line, bounded halt reason. Raises ValueError when unusable.

    The halt file is line-oriented (timestamp, reason, source), so newlines are
    collapsed rather than stored.
    """
    if not isinstance(raw, str):
        raise ValueError("reason is required")
    reason = re.sub(r"\s+", " ", raw).strip()
    if not reason:
        raise ValueError("reason is required")
    if len(reason) > MAX_REASON_LENGTH:
        raise ValueError(f"reason must be at most {MAX_REASON_LENGTH} characters")
    return reason


def write_halt(path: Path, reason: str, now: Optional[datetime] = None,
               source: str = "dashboard") -> bool:
    """Create the halt file. True if this call halted, False if already halted.

    O_EXCL makes "already halted" atomic: two operators clicking at once cannot
    both believe they wrote the reason. The file is checked afterwards because a
    halt that silently failed to land is the worst outcome this can have.

    The timestamp is Eastern with its offset. The containers run on UTC, and a
    naive datetime.now() wrote 12:37 for a halt made at 08:37 ET.

    No client address is recorded: behind rootless podman's port forward every
    request comes from the network gateway, and with one shared API token there
    is no per-person identity to record. A made-up "who" is worse than none.
    """
    path = Path(path)
    now = now or now_et()
    body = f"{now.isoformat()}\n{reason}\nsource: {source}\n"
    # mkdir is outside the O_EXCL block on purpose: it raises FileExistsError
    # too (when a file sits where the directory should be), and reading that as
    # "already halted" would report a halt that was never written.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HaltWriteError(f"could not create {path.parent}: {exc}") from exc
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if not path.is_file():
            raise HaltWriteError(f"{path} exists but is not a file")
        return False
    except OSError as exc:
        raise HaltWriteError(f"could not create {path}: {exc}") from exc
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
    except OSError as exc:
        # The file exists, so trading is halted even though the note is short.
        logger.error("Halt file %s created but its note was not written: %s", path, exc)
    if not path.exists():
        raise HaltWriteError(f"{path} does not exist after writing it")
    return True


def remove_halt(path: Path) -> bool:
    """Delete the halt file. True if one was present."""
    path = Path(path)
    try:
        path.unlink()
        present = True
    except FileNotFoundError:
        present = False
    except OSError as exc:
        raise HaltWriteError(f"could not remove {path}: {exc}") from exc
    if path.exists():
        raise HaltWriteError(f"{path} still exists after removing it")
    return present


def read_halt_note(path: Path) -> Optional[Dict[str, Any]]:
    """Who halted and why, from the halt file. None when there is no halt file.

    Reads at most a few KB: the file may have been written by hand.
    """
    path = Path(path)
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read(4096)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"unreadable": str(exc)}
    lines = [ln.strip() for ln in text.splitlines()]
    halted_at = lines[0] if len(lines) > 0 else None
    return {
        "halted_at": halted_at,
        "halted_at_display": format_halted_at(halted_at),
        "reason": lines[1][:MAX_REASON_LENGTH] if len(lines) > 1 else None,
        "source": lines[2][:MAX_REASON_LENGTH] if len(lines) > 2 else None,
    }


def format_halted_at(raw: Optional[str]) -> Optional[str]:
    """The halt time in Eastern, for display. Never presents an unknown zone as ET.

    Halt files written by KillSwitch.halt() or by hand may carry a naive
    timestamp; those are shown as-is and labelled, not guessed at.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw[:40]
    if parsed.tzinfo is None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S") + " (timezone not recorded)"
    return parsed.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def read_mountinfo() -> str:
    try:
        with open("/proc/self/mountinfo") as fh:
            return fh.read()
    except OSError:
        return ""


def create_blueprint(get_bot: Callable[[], Any], get_db: Callable[[], Any],
                     load_state: Optional[Callable[[Any, str], Any]] = None,
                     mountinfo: Callable[[], str] = read_mountinfo):
    """The /api/risk/* routes.

    ``get_bot`` returns the dashboard's PaperTradingBot (or None before it is
    initialized); its kill_switch names the halt file. Authentication and the
    Origin check are not here: auth.install gates every route on the app,
    blueprint routes included.
    """
    from flask import Blueprint, jsonify, request

    if load_state is None:
        from falcon_trader.symbol_state import load_symbol_state as load_state

    bp = Blueprint("risk", __name__)

    def _kill_switch_view(bot) -> Dict[str, Any]:
        status = bot.kill_switch.status()
        status["halt_note"] = read_halt_note(bot.kill_switch.halt_file)
        return risk_status.kill_switch_view(status, mountinfo())

    @bp.route("/api/risk/status")
    def risk_status_route():
        bot = get_bot()
        if not bot:
            return jsonify({"error": "Bot not initialized"}), 503
        db = get_db()
        held, symbols = risk_status.held_and_recent_symbols(db)
        states = [load_state(db, sym) for sym in symbols]
        body = risk_status.build_status(
            kill_switch_status=bot.kill_switch.status(),
            mountinfo=mountinfo(),
            policy=bot.cooldown_policy,
            states=states,
            held=held,
        )
        body["kill_switch"] = _kill_switch_view(bot)
        return jsonify(body)

    @bp.route("/api/risk/halt", methods=["POST"])
    def halt_route():
        bot = get_bot()
        if not bot:
            return jsonify({"error": "Bot not initialized"}), 503
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        try:
            reason = clean_reason(data.get("reason"))
        except ValueError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400

        path = bot.kill_switch.halt_file
        try:
            halted_now = write_halt(path, reason)
        except HaltWriteError as exc:
            logger.critical("HALT REQUEST FAILED from %s: %s", request.remote_addr, exc)
            return jsonify({
                "status": "error",
                "error": f"Halt was NOT written: {exc}",
                "kill_switch": _kill_switch_view(bot),
            }), 500

        if halted_now:
            logger.critical("TRADING HALTED from dashboard by %s: %s",
                            request.remote_addr, reason)
        return jsonify({
            "status": "halted" if halted_now else "already_halted",
            "kill_switch": _kill_switch_view(bot),
        })

    @bp.route("/api/risk/resume", methods=["POST"])
    def resume_route():
        bot = get_bot()
        if not bot:
            return jsonify({"error": "Bot not initialized"}), 503
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        if data.get("confirm") != RESUME_CONFIRMATION:
            return jsonify({
                "status": "error",
                "error": f'resume requires {{"confirm": "{RESUME_CONFIRMATION}"}}',
            }), 400

        path = bot.kill_switch.halt_file
        note = read_halt_note(path)
        try:
            was_present = remove_halt(path)
        except HaltWriteError as exc:
            logger.critical("RESUME REQUEST FAILED from %s: %s", request.remote_addr, exc)
            return jsonify({
                "status": "error",
                "error": f"Halt file was NOT removed: {exc}",
                "kill_switch": _kill_switch_view(bot),
            }), 500

        if was_present:
            logger.warning("Halt file removed from dashboard by %s (was: %s)",
                           request.remote_addr, note)

        view = _kill_switch_view(bot)
        body = {
            "status": "resumed" if was_present else "not_halted",
            "kill_switch": view,
        }
        if not view["trading_enabled"]:
            body["status"] = "still_halted"
            body["error"] = f"Halt file cleared, but trading is still off: {view['reason']}"
        return jsonify(body)

    return bp
