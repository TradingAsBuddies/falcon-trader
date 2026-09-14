"""Tests for falcon_trader.trading_guards (falcon-trader#23, #24).

The two tests the issue named explicitly are ``test_the_1650_order_is_rejected``
and ``test_an_order_on_labor_day_is_rejected``. The rest pin the surrounding
behavior so the guards cannot be loosened by accident.
"""

import datetime as dt

import pytest

from falcon_trader.trading_guards import (
    DEFAULT_FLATTEN_AT,
    CooldownPolicy,
    calendar_is_complete,
    check_cooldown,
    check_entry,
    check_fill_price_freshness,
    check_market_open,
    is_rth,
    should_flatten,
)

# A normal Tuesday session.
SESSION = dt.date(2026, 9, 8)


def at(hour, minute=0, day=SESSION):
    return dt.datetime.combine(day, dt.time(hour, minute))


# --------------------------------------------------------------------------
# the two the issue asked for
# --------------------------------------------------------------------------

def test_the_1650_order_is_rejected():
    """Live book: BUY INDV and five SELLs at 16:50 on 2026-09-08."""
    result = check_market_open(dt.datetime(2026, 9, 8, 16, 50, 7))
    assert result.allowed is False
    assert result.reason == "outside_rth"


def test_an_order_on_labor_day_is_rejected():
    """2026-09-07 10:05 -- market open time, but a holiday."""
    result = check_market_open(dt.datetime(2026, 9, 7, 10, 5))
    assert result.allowed is False
    # Needs the real calendar to say "not_a_session"; the fallback still blocks.
    if calendar_is_complete():
        assert result.reason == "not_a_session"
    else:  # pragma: no cover - only without falcon-core
        assert result.reason in ("not_a_session", "outside_rth")


# --------------------------------------------------------------------------
# session boundaries
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "moment, allowed",
    [
        (at(9, 29), False),
        (at(9, 30), True),
        (at(12, 0), True),
        (at(15, 59), True),
        (at(16, 0), False),
        (at(16, 50), False),
        (at(4, 0), False),
        (at(20, 0), False),
    ],
)
def test_market_open_boundaries(moment, allowed):
    assert check_market_open(moment).allowed is allowed


def test_weekend_is_rejected():
    saturday = dt.datetime(2026, 9, 12, 12, 0)
    result = check_market_open(saturday)
    assert result.allowed is False


def test_is_rth_agrees_with_check_market_open():
    for hour in range(0, 24):
        moment = at(hour)
        assert is_rth(moment) is check_market_open(moment).allowed


def test_the_calendar_is_the_real_one_here():
    """falcon-core is on the path in this repo's test env, so no fallback."""
    assert calendar_is_complete() is True


# --------------------------------------------------------------------------
# fill price freshness
# --------------------------------------------------------------------------

def test_previous_session_close_is_refused_as_a_fill_price():
    """paper_trading_bot priced fills from /v2/aggs/.../prev."""
    prev_close = dt.datetime(2026, 9, 4, 16, 0)
    result = check_fill_price_freshness(prev_close, moment=at(10, 0))
    assert result.allowed is False
    assert result.reason == "stale_price"


def test_a_current_session_bar_is_accepted():
    result = check_fill_price_freshness(at(9, 55), moment=at(10, 0))
    assert result.allowed is True


def test_an_old_bar_from_today_is_refused():
    result = check_fill_price_freshness(at(9, 31), moment=at(15, 0))
    assert result.allowed is False
    assert result.reason == "stale_price"


def test_a_missing_bar_is_refused():
    result = check_fill_price_freshness(None, moment=at(10, 0))
    assert result.allowed is False
    assert result.reason == "no_bar"


def test_a_future_bar_is_refused():
    result = check_fill_price_freshness(at(11, 0), moment=at(10, 0))
    assert result.allowed is False
    assert result.reason == "future_bar"


def test_small_clock_skew_is_tolerated():
    result = check_fill_price_freshness(at(10, 0, ), moment=at(10, 0))
    assert result.allowed is True


# --------------------------------------------------------------------------
# EOD flatten
# --------------------------------------------------------------------------

def test_flatten_window_opens_at_1555():
    assert should_flatten(at(15, 54)) is False
    assert should_flatten(at(15, 55)) is True
    assert should_flatten(at(15, 59)) is True


def test_flatten_window_closes_at_the_bell():
    assert should_flatten(at(16, 0)) is False
    assert should_flatten(at(16, 50)) is False


def test_no_flatten_on_a_holiday():
    assert should_flatten(dt.datetime(2026, 9, 7, 15, 56)) is False


def test_flatten_time_is_configurable():
    assert should_flatten(at(15, 30), flatten_at=dt.time(15, 30)) is True
    assert DEFAULT_FLATTEN_AT == dt.time(15, 55)


def test_no_new_entries_inside_the_flatten_window():
    result = check_entry("AAPL", moment=at(15, 56), bar_timestamp=at(15, 55))
    assert result.allowed is False
    assert result.reason == "eod_window"


# --------------------------------------------------------------------------
# cooldown and loss lockout (#24)
# --------------------------------------------------------------------------

def test_cdxs_reentry_twenty_hours_later_is_blocked():
    """Live: sold 2026-09-08 16:50 at a loss, re-bought 2026-09-09 12:40."""
    result = check_cooldown(
        "CDXS", CooldownPolicy(),
        last_exit_at=dt.datetime(2026, 9, 8, 16, 50),
        loss_streak=1,
        round_trips_today=0,
        moment=dt.datetime(2026, 9, 9, 12, 40),
    )
    assert result.allowed is False
    assert result.reason == "cooldown"


def test_pcvx_reentry_two_hours_later_is_blocked():
    """Live: sold 2026-09-09 10:37 at a loss, re-bought 12:56 the same day."""
    result = check_cooldown(
        "PCVX", CooldownPolicy(),
        last_exit_at=dt.datetime(2026, 9, 9, 10, 37),
        loss_streak=1,
        round_trips_today=1,
        moment=dt.datetime(2026, 9, 9, 12, 56),
    )
    assert result.allowed is False
    # The daily cap catches it first, which is the stricter and correct answer.
    assert result.reason in ("daily_round_trip_cap", "cooldown")


def test_reentry_after_the_cooldown_is_allowed():
    result = check_cooldown(
        "CDXS", CooldownPolicy(cooldown=dt.timedelta(hours=4)),
        last_exit_at=dt.datetime(2026, 9, 8, 10, 0),
        moment=dt.datetime(2026, 9, 8, 15, 0),
    )
    assert result.allowed is True


def test_a_symbol_never_traded_is_allowed():
    assert check_cooldown("NEW", CooldownPolicy(), last_exit_at=None).allowed is True


def test_loss_streak_triggers_a_longer_lockout():
    policy = CooldownPolicy(
        cooldown=dt.timedelta(hours=1),
        max_loss_streak=2,
        lockout=dt.timedelta(days=5),
    )
    result = check_cooldown(
        "WGRX", policy,
        last_exit_at=dt.datetime(2026, 9, 8, 10, 0),
        loss_streak=2,
        moment=dt.datetime(2026, 9, 9, 10, 0),   # past cooldown, inside lockout
    )
    assert result.allowed is False
    assert result.reason == "loss_lockout"


def test_lockout_expires():
    policy = CooldownPolicy(max_loss_streak=2, lockout=dt.timedelta(days=5),
                            cooldown=dt.timedelta(hours=1))
    result = check_cooldown(
        "WGRX", policy,
        last_exit_at=dt.datetime(2026, 9, 1, 10, 0),
        loss_streak=2,
        moment=dt.datetime(2026, 9, 9, 10, 0),
    )
    assert result.allowed is True


def test_daily_round_trip_cap():
    result = check_cooldown("AAA", CooldownPolicy(max_round_trips_per_day=1),
                            round_trips_today=1)
    assert result.allowed is False
    assert result.reason == "daily_round_trip_cap"


def test_blacklist_blocks_regardless():
    policy = CooldownPolicy(blacklist=frozenset({"SNYR"}))
    assert check_cooldown("snyr", policy).allowed is False
    assert check_cooldown("SNYR", policy).reason == "blacklisted"


# --------------------------------------------------------------------------
# policy configuration -- thresholds must not be hardcoded
# --------------------------------------------------------------------------

def test_policy_from_config():
    policy = CooldownPolicy.from_config({
        "cooldown_hours": 2,
        "max_loss_streak": 3,
        "lockout_hours": 48,
        "max_round_trips_per_day": 2,
        "blacklist": ["snyr", "aeon"],
    })
    assert policy.cooldown == dt.timedelta(hours=2)
    assert policy.max_loss_streak == 3
    assert policy.lockout == dt.timedelta(hours=48)
    assert policy.max_round_trips_per_day == 2
    assert policy.blacklist == frozenset({"SNYR", "AEON"})


def test_policy_from_empty_config_uses_defaults():
    assert CooldownPolicy.from_config({}) == CooldownPolicy()
    assert CooldownPolicy.from_config(None) == CooldownPolicy()


# --------------------------------------------------------------------------
# the combined gate
# --------------------------------------------------------------------------

def test_check_entry_allows_a_clean_entry():
    result = check_entry(
        "AAPL", moment=at(10, 0), bar_timestamp=at(9, 59), last_exit_at=None,
    )
    assert result.allowed is True


def test_check_entry_reports_session_before_churn():
    """At 16:50 the answer is 'market shut', not 'cooldown'."""
    result = check_entry(
        "CDXS", moment=at(16, 50), bar_timestamp=at(16, 49),
        last_exit_at=at(9, 0), loss_streak=5,
    )
    assert result.reason == "outside_rth"


def test_check_entry_can_skip_the_price_check():
    result = check_entry(
        "AAPL", moment=at(10, 0), bar_timestamp=None, require_fresh_price=False,
    )
    assert result.allowed is True


def test_guard_result_is_falsy_when_blocked():
    assert not check_market_open(at(16, 50))
    assert check_market_open(at(10, 0))
