"""What risk controls are actually in force, for the dashboard's risk panel.

Pure functions over plain values, so they are testable without Flask, a
database or a container. dashboard_server's /api/risk/status is the adapter.

The panel exists to answer "what will stop a bad trade right now?" honestly.
That means reporting three things a naive status page would get wrong:

* **Scope.** The kill switch is checked by some order paths and not others.
* **Reach.** A halt file only halts processes that can see it. The dashboard and
  trader run in separate containers, and a halt written by one is invisible to
  the other unless the directory is a shared mount.
* **What is defined but not enforced.** risk_limits.RiskLimits declares a daily
  loss limit, position caps and more, and nothing calls check_pre_trade. Showing
  those values without saying so would imply protection that does not exist.
"""

import datetime as _dt
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

from falcon_trader.risk_limits import RiskLimits
from falcon_trader.trading_guards import CooldownPolicy, check_cooldown

#: Order paths that consult the kill switch before opening a position.
ENFORCED_ON = (
    {"path": "dashboard order ticket", "code": "PaperTradingBot.place_order"},
    {"path": "orchestrator entries", "code": "BaseEngine.execute_buy"},
)

#: Deliberately never gated by the kill switch.
NEVER_BLOCKED = (
    {"path": "exits and sells",
     "why": "a halt that blocked exits would trap the book in open positions"},
)

#: Out of reach from this deployment.
NOT_REACHABLE = (
    {"path": "DAS execution",
     "why": "DAS runs on Windows with a host-only interface; no proxy is deployed"},
)


def is_mount_point_or_below(path: str, mountinfo: str) -> bool:
    """True when `path` sits on a mount other than the container root.

    `mountinfo` is the text of /proc/self/mountinfo, whose fifth field is the
    mount point. A halt file on the container's root filesystem is private to
    that container and gone when it is recreated; one under a mounted volume can
    be shared.
    """
    target = PurePosixPath(path)
    mount_points = []
    for line in (mountinfo or "").splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[4] != "/":
            mount_points.append(PurePosixPath(fields[4]))
    return any(target == mp or mp in target.parents for mp in mount_points)


def kill_switch_view(status: Dict[str, Any], mountinfo: str) -> Dict[str, Any]:
    """The kill switch state plus whether a halt from here reaches anything else."""
    shared = is_mount_point_or_below(status.get("halt_file", ""), mountinfo)
    view = dict(status)
    view["halt_file_on_shared_mount"] = shared
    if not shared:
        view["reach_warning"] = (
            "The halt file is on this container's own filesystem. A halt written "
            "here is not visible to the trader container and is lost when this "
            "container is recreated."
        )
    return view


def _seconds(delta: _dt.timedelta) -> int:
    return int(delta.total_seconds())


def cooldown_policy_view(policy: CooldownPolicy) -> Dict[str, Any]:
    return {
        "cooldown_seconds": _seconds(policy.cooldown),
        "max_loss_streak": policy.max_loss_streak,
        "lockout_seconds": _seconds(policy.lockout),
        "max_round_trips_per_day": policy.max_round_trips_per_day,
        "blacklist": sorted(policy.blacklist),
        "enforced": True,
    }


def defined_not_enforced(limits: Optional[RiskLimits] = None) -> Dict[str, Any]:
    """RiskLimits values, explicitly marked as not in force.

    check_pre_trade is not called by any order path. These are the defaults the
    module declares, reported so the gap is visible rather than implied away.
    """
    limits = limits or RiskLimits()
    return {
        "enforced": False,
        "note": "Defined in risk_limits.RiskLimits; check_pre_trade is not called "
                "by any order path, so none of these block a trade.",
        "limits": {
            "max_position_pct": limits.max_position_pct,
            "max_gross_exposure_pct": limits.max_gross_exposure_pct,
            "max_daily_loss_pct": limits.max_daily_loss_pct,
            "max_open_positions": limits.max_open_positions,
            "min_price": limits.min_price,
            "min_dollar_volume": limits.min_dollar_volume,
            "max_day_trades": limits.max_day_trades,
            "blacklist": sorted(limits.blacklist),
        },
    }


def symbol_rows(states: Iterable[Any], policy: CooldownPolicy,
                held: Iterable[str], now: Optional[_dt.datetime] = None) -> List[Dict[str, Any]]:
    """Per-symbol churn state, and whether a new entry would be allowed now."""
    held = set(held)
    rows = []
    for st in states:
        decision = check_cooldown(
            st.symbol, policy,
            last_exit_at=st.last_exit_at,
            loss_streak=st.loss_streak,
            round_trips_today=st.round_trips_today,
            moment=now,
        )
        rows.append({
            "symbol": st.symbol,
            "held": st.symbol in held,
            "entry_allowed": decision.allowed,
            "block_reason": decision.reason,
            "block_message": decision.message,
            "last_exit_at": st.last_exit_at.isoformat() if st.last_exit_at else None,
            "last_exit_pnl": st.last_exit_pnl,
            "loss_streak": st.loss_streak,
            "round_trips_today": st.round_trips_today,
        })
    rows.sort(key=lambda r: (r["entry_allowed"], not r["held"], r["symbol"]))
    return rows


def build_status(kill_switch_status: Dict[str, Any], mountinfo: str,
                 policy: CooldownPolicy, states: Iterable[Any],
                 held: Iterable[str], now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    return {
        "kill_switch": kill_switch_view(kill_switch_status, mountinfo),
        "enforced_on": list(ENFORCED_ON),
        "never_blocked": list(NEVER_BLOCKED),
        "not_reachable": list(NOT_REACHABLE),
        "cooldown_policy": cooldown_policy_view(policy),
        "symbols": symbol_rows(states, policy, held, now),
        "pre_trade_limits": defined_not_enforced(),
    }
