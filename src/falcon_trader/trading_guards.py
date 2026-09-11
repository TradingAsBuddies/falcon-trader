"""Entry gates for automated trading (falcon-trader#23, #24).

Two families of guard, both pure so they can be tested without a clock, a
database, or a market data key:

*Session guards* (#23) -- is the market actually open, and is this fill priced
off a bar from the current session? The live book contains a BUY and five SELLs
timestamped 16:50, fifty minutes after the close, all "filled" at the prior
day's close. No trading loop in the repo had a market-hours check.

*Churn guards* (#24) -- has this symbol just been exited, and how many times has
it lost recently? The live book shows CDXS sold at a $277 loss and re-bought
twenty hours later a cent higher, and PCVX sold at a $90 loss and re-bought two
hours later at a higher price.

``is_rth`` comes from :mod:`falcon_core.market_calendar` when available. It is
imported defensively: falcon-trader and falcon-core deploy as separate packages
and can skew, and a trading loop that dies on ImportError is worse than one
running on a slightly coarser calendar. The fallback is weekday + clock only,
which is strictly more conservative than nothing but does not know holidays --
so it says so, loudly, once.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

__all__ = [
    "EASTERN",
    "GuardResult",
    "CooldownPolicy",
    "is_rth",
    "calendar_is_complete",
    "check_market_open",
    "check_fill_price_freshness",
    "should_flatten",
    "check_cooldown",
    "check_entry",
]

EASTERN = ZoneInfo("America/New_York")

_MARKET_OPEN = _dt.time(9, 30)
_MARKET_CLOSE = _dt.time(16, 0)

#: Default flatten time. 15:55 leaves five minutes to actually get filled.
DEFAULT_FLATTEN_AT = _dt.time(15, 55)

#: A fill priced off a bar older than this is refused.
DEFAULT_MAX_BAR_AGE = _dt.timedelta(minutes=30)


# --------------------------------------------------------------------------
# calendar, with a safe degradation path
# --------------------------------------------------------------------------

try:
    from falcon_core.market_calendar import is_rth as _core_is_rth
    from falcon_core.market_calendar import is_session as _core_is_session

    _CALENDAR_COMPLETE = True
except Exception as _exc:  # pragma: no cover - depends on deployed core version
    _core_is_rth = None
    _core_is_session = None
    _CALENDAR_COMPLETE = False
    logger.error(
        "falcon_core.market_calendar unavailable (%s). Falling back to a "
        "weekday+clock check, which does NOT know market holidays. Upgrade "
        "falcon-core to get holiday-aware gating (falcon-core#20).", _exc,
    )


def calendar_is_complete() -> bool:
    """True when the holiday-aware calendar is in use."""
    return _CALENDAR_COMPLETE


def _as_eastern(moment: Optional[_dt.datetime]) -> _dt.datetime:
    if moment is None:
        return _dt.datetime.now(EASTERN)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=EASTERN)
    return moment.astimezone(EASTERN)


def is_rth(moment: Optional[_dt.datetime] = None) -> bool:
    """True when `moment` is inside regular trading hours."""
    if _core_is_rth is not None:
        return bool(_core_is_rth(moment))
    now = _as_eastern(moment)
    if now.weekday() >= 5:
        return False
    return _MARKET_OPEN <= now.time() < _MARKET_CLOSE


def _is_session(day) -> bool:
    if _core_is_session is not None:
        return bool(_core_is_session(day))
    return day.weekday() < 5


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardResult:
    """Whether an action may proceed, and why not."""

    allowed: bool
    reason: Optional[str] = None
    message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.allowed

    @classmethod
    def ok(cls) -> "GuardResult":
        return cls(allowed=True)

    @classmethod
    def block(cls, reason: str, message: str) -> "GuardResult":
        return cls(allowed=False, reason=reason, message=message)


@dataclass(frozen=True)
class CooldownPolicy:
    """Per-symbol churn limits (falcon-trader#24).

    Defaults are deliberately conservative; every field is meant to be driven
    from the orchestrator config rather than edited here.
    """

    #: Minimum wall-clock gap between exiting a symbol and re-entering it.
    cooldown: _dt.timedelta = _dt.timedelta(days=1)
    #: Consecutive losing round-trips before the symbol is locked out.
    max_loss_streak: int = 2
    #: How long the lockout lasts once the streak is hit.
    lockout: _dt.timedelta = _dt.timedelta(days=5)
    #: Round-trips allowed per symbol per session.
    max_round_trips_per_day: int = 1
    #: Symbols never to trade, whatever the signal says.
    blacklist: frozenset = frozenset()

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "CooldownPolicy":
        """Build from a plain config dict (orchestrator YAML)."""
        config = config or {}

        def _td(key, default):
            hours = config.get(key)
            return _dt.timedelta(hours=float(hours)) if hours is not None else default

        return cls(
            cooldown=_td("cooldown_hours", cls.cooldown),
            max_loss_streak=int(config.get("max_loss_streak", cls.max_loss_streak)),
            lockout=_td("lockout_hours", cls.lockout),
            max_round_trips_per_day=int(
                config.get("max_round_trips_per_day", cls.max_round_trips_per_day)
            ),
            blacklist=frozenset(
                s.upper() for s in (config.get("blacklist") or ())
            ),
        )


# --------------------------------------------------------------------------
# session guards (#23)
# --------------------------------------------------------------------------

def check_market_open(moment: Optional[_dt.datetime] = None) -> GuardResult:
    """Refuse anything outside regular trading hours."""
    now = _as_eastern(moment)

    if not _is_session(now.date()):
        return GuardResult.block(
            "not_a_session",
            f"{now.date()} is not a trading session (weekend or market holiday)",
        )

    if not is_rth(now):
        return GuardResult.block(
            "outside_rth",
            f"{now:%Y-%m-%d %H:%M:%S %Z} is outside regular trading hours",
        )

    return GuardResult.ok()


def check_fill_price_freshness(
    bar_timestamp: Optional[_dt.datetime],
    moment: Optional[_dt.datetime] = None,
    max_age: _dt.timedelta = DEFAULT_MAX_BAR_AGE,
) -> GuardResult:
    """Refuse a fill priced off a bar that is not from the current session.

    Fills were priced from ``/v2/aggs/ticker/{sym}/prev`` -- the *previous
    session's* close -- so an order at any wall-clock time "filled" at a stale
    price. That is what let the 16:50 orders succeed at all.
    """
    if bar_timestamp is None:
        return GuardResult.block(
            "no_bar",
            "No price bar available for this fill",
        )

    now = _as_eastern(moment)
    bar = _as_eastern(bar_timestamp)

    if bar.date() != now.date():
        return GuardResult.block(
            "stale_price",
            f"Price bar is from {bar.date()}, not the current session {now.date()}",
        )

    age = now - bar
    if age > max_age:
        return GuardResult.block(
            "stale_price",
            f"Price bar is {int(age.total_seconds() // 60)} minutes old "
            f"(limit {int(max_age.total_seconds() // 60)})",
        )

    if age < -_dt.timedelta(minutes=1):
        return GuardResult.block(
            "future_bar",
            f"Price bar is timestamped {bar} which is ahead of {now}",
        )

    return GuardResult.ok()


def should_flatten(
    moment: Optional[_dt.datetime] = None,
    flatten_at: _dt.time = DEFAULT_FLATTEN_AT,
) -> bool:
    """True inside the end-of-day flatten window.

    There was no EOD flatten anywhere -- positions simply carried, and the
    16:50 SELL burst was the 5-minute monitor cycle happening to land after the
    close rather than a deliberate exit.
    """
    now = _as_eastern(moment)
    if not _is_session(now.date()):
        return False
    return flatten_at <= now.time() < _MARKET_CLOSE


# --------------------------------------------------------------------------
# churn guards (#24)
# --------------------------------------------------------------------------

def check_cooldown(
    symbol: str,
    policy: CooldownPolicy,
    last_exit_at: Optional[_dt.datetime] = None,
    loss_streak: int = 0,
    round_trips_today: int = 0,
    moment: Optional[_dt.datetime] = None,
) -> GuardResult:
    """Refuse a re-entry that is churn rather than signal."""
    symbol = (symbol or "").upper()
    now = _as_eastern(moment)

    if symbol in policy.blacklist:
        return GuardResult.block(
            "blacklisted", f"{symbol} is blacklisted",
        )

    if round_trips_today >= policy.max_round_trips_per_day:
        return GuardResult.block(
            "daily_round_trip_cap",
            f"{symbol} already had {round_trips_today} round-trip(s) today "
            f"(cap {policy.max_round_trips_per_day})",
        )

    # A loss streak locks the symbol out for longer than the ordinary cooldown.
    if loss_streak >= policy.max_loss_streak and last_exit_at is not None:
        unlock = _as_eastern(last_exit_at) + policy.lockout
        if now < unlock:
            return GuardResult.block(
                "loss_lockout",
                f"{symbol} has {loss_streak} consecutive losing round-trips; "
                f"locked out until {unlock:%Y-%m-%d %H:%M %Z}",
            )

    if last_exit_at is not None:
        ready = _as_eastern(last_exit_at) + policy.cooldown
        if now < ready:
            return GuardResult.block(
                "cooldown",
                f"{symbol} was exited at {_as_eastern(last_exit_at):%Y-%m-%d %H:%M %Z}; "
                f"re-entry allowed from {ready:%Y-%m-%d %H:%M %Z}",
            )

    return GuardResult.ok()


# --------------------------------------------------------------------------
# the combined gate
# --------------------------------------------------------------------------

def check_entry(
    symbol: str,
    policy: Optional[CooldownPolicy] = None,
    *,
    moment: Optional[_dt.datetime] = None,
    bar_timestamp: Optional[_dt.datetime] = None,
    last_exit_at: Optional[_dt.datetime] = None,
    loss_streak: int = 0,
    round_trips_today: int = 0,
    require_fresh_price: bool = True,
) -> GuardResult:
    """Every entry gate, in order, first failure wins.

    Session before churn: if the market is shut, why the entry was refused is
    not interesting.
    """
    session = check_market_open(moment)
    if not session.allowed:
        return session

    if should_flatten(moment):
        return GuardResult.block(
            "eod_window",
            "Inside the end-of-day flatten window; no new entries",
        )

    if require_fresh_price:
        fresh = check_fill_price_freshness(bar_timestamp, moment)
        if not fresh.allowed:
            return fresh

    return check_cooldown(
        symbol,
        policy or CooldownPolicy(),
        last_exit_at=last_exit_at,
        loss_streak=loss_streak,
        round_trips_today=round_trips_today,
        moment=moment,
    )
