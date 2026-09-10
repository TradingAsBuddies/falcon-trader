"""Tests for falcon_trader.portfolio (falcon-trader#21, #22).

The account invariant test the issue asked for is :func:`test_invariant_holds...`
below. Everything else exists to pin the specific ways the live book drifted.
"""

import pytest

from falcon_trader.portfolio import (
    check_invariant,
    mark_position,
    reduce_position,
    replay,
    validate_sell,
    value_account,
    weighted_average_entry,
)


# --------------------------------------------------------------------------
# sells that used to mint cash
# --------------------------------------------------------------------------

def test_sell_with_no_position_is_rejected():
    """_update_position was a silent no-op; place_order credited cash anyway."""
    check = validate_sell(None, 100)
    assert check.ok is False
    assert check.reason == "no_position"


def test_sell_with_zero_position_is_rejected():
    assert validate_sell(0, 10).ok is False


def test_sell_exceeding_held_quantity_is_rejected():
    """The old path deleted the row and credited the full notional."""
    check = validate_sell(50, 100)
    assert check.ok is False
    assert check.reason == "insufficient_shares"
    assert "only 50 held" in check.message


def test_sell_of_exactly_the_held_quantity_is_allowed():
    assert validate_sell(50, 50).ok is True


def test_partial_sell_is_allowed():
    assert validate_sell(50, 20).ok is True


@pytest.mark.parametrize("qty", [0, -1, None])
def test_non_positive_sell_quantity_is_rejected(qty):
    check = validate_sell(100, qty)
    assert check.ok is False
    assert check.reason == "invalid_quantity"


def test_reduce_position_never_goes_negative():
    assert reduce_position(50, 50) == 0
    assert reduce_position(50, 80) == 0
    assert reduce_position(50, 20) == 30


# --------------------------------------------------------------------------
# scale-ins
# --------------------------------------------------------------------------

def test_weighted_average_on_scale_in():
    """base_engine incremented quantity without recomputing entry_price."""
    # 100 @ $10 then 100 @ $20 -> 200 @ $15
    assert weighted_average_entry(100, 10.0, 100, 20.0) == pytest.approx(15.0)


def test_weighted_average_respects_size():
    # 300 @ $10 then 100 @ $20 -> 400 @ $12.50
    assert weighted_average_entry(300, 10.0, 100, 20.0) == pytest.approx(12.5)


def test_weighted_average_from_empty_is_the_new_price():
    assert weighted_average_entry(0, 0.0, 100, 17.5) == pytest.approx(17.5)


def test_weighted_average_of_nothing_is_zero():
    assert weighted_average_entry(0, 0.0, 0, 0.0) == 0.0


# --------------------------------------------------------------------------
# marking -- the currentPrice == avgPrice bug
# --------------------------------------------------------------------------

def test_missing_quote_marks_stale_and_does_not_fall_back_to_cost():
    """All 10 live positions showed currentPrice == avgPrice."""
    mark = mark_position("INDV", 48, 34.32, {})
    assert mark.stale is True
    assert mark.current_price is None
    assert mark.to_dict()["currentPrice"] is None


def test_real_quote_produces_a_real_mark():
    mark = mark_position("INDV", 48, 34.32, {"INDV": 35.00})
    assert mark.stale is False
    assert mark.current_price == 35.00
    assert mark.current_price != mark.avg_price
    assert mark.unrealized_pnl == pytest.approx((35.00 - 34.32) * 48)


def test_mark_updates_after_a_price_refresh():
    """The test the issue explicitly asked for."""
    before = mark_position("CDXS", 1588, 1.45, {})
    assert before.current_price is None

    after = mark_position("CDXS", 1588, 1.45, {"CDXS": 1.61})
    assert after.current_price is not None
    assert after.current_price != after.avg_price
    assert after.unrealized_pnl > 0


@pytest.mark.parametrize("bad", [None, 0, -1, "", "n/a"])
def test_unusable_prices_are_treated_as_stale(bad):
    mark = mark_position("BMBL", 100, 2.79, {"BMBL": bad})
    assert mark.stale is True
    assert mark.current_price is None


def test_stale_position_reports_zero_unrealized_not_a_fake_number():
    mark = mark_position("WGRX", 100, 2.80, {})
    assert mark.unrealized_pnl == 0.0
    assert mark.unrealized_pnl_pct == 0.0
    # still carried at cost so the account total stays finite
    assert mark.market_value == pytest.approx(280.0)


def test_loss_is_reported_as_negative():
    mark = mark_position("PCVX", 30, 59.63, {"PCVX": 55.00})
    assert mark.unrealized_pnl < 0


# --------------------------------------------------------------------------
# one price map, one answer
# --------------------------------------------------------------------------

def test_positions_value_and_total_value_agree():
    """The live account had cash + positions != totalValue by $361.74."""
    positions = [
        {"symbol": "AAA", "quantity": 100, "avgPrice": 10.0},
        {"symbol": "BBB", "quantity": 50, "avgPrice": 20.0},
    ]
    val = value_account(
        cash=5000.0, initial_balance=10000.0, positions=positions,
        price_map={"AAA": 11.0, "BBB": 19.0},
    )
    assert val.positions_value == pytest.approx(100 * 11.0 + 50 * 19.0)
    assert val.total_value == pytest.approx(val.cash + val.positions_value)


def test_stale_symbols_are_reported():
    positions = [
        {"symbol": "AAA", "quantity": 100, "avgPrice": 10.0},
        {"symbol": "BBB", "quantity": 50, "avgPrice": 20.0},
    ]
    val = value_account(5000.0, 10000.0, positions, {"AAA": 11.0})
    assert val.stale_symbols == ("BBB",)
    assert val.has_stale_marks is True


def test_entry_price_key_is_accepted_as_well_as_avgprice():
    """DB rows use entry_price; API payloads use avgPrice."""
    val = value_account(
        0.0, 0.0,
        [{"symbol": "AAA", "quantity": 10, "entry_price": 5.0}],
        {"AAA": 6.0},
    )
    assert val.positions_value == pytest.approx(60.0)


# --------------------------------------------------------------------------
# the invariant
# --------------------------------------------------------------------------

def test_invariant_holds_after_an_arbitrary_fill_sequence():
    """cash + positions == initial + realized + unrealized, after any fills."""
    fills = [
        {"symbol": "AAA", "side": "buy",  "quantity": 100, "price": 10.0},
        {"symbol": "BBB", "side": "buy",  "quantity": 50,  "price": 20.0},
        {"symbol": "AAA", "side": "buy",  "quantity": 100, "price": 12.0},
        {"symbol": "AAA", "side": "sell", "quantity": 150, "price": 15.0},
        {"symbol": "BBB", "side": "sell", "quantity": 50,  "price": 18.0},
        {"symbol": "CCC", "side": "buy",  "quantity": 10,  "price": 30.0},
    ]
    book = replay(10000.0, fills)
    prices = {"AAA": 16.0, "CCC": 33.0}

    val = value_account(
        cash=book["cash"],
        initial_balance=10000.0,
        positions=book["positions"],
        price_map=prices,
        realized_pnl=book["realized_pnl"],
    )
    report = check_invariant(val)
    assert report.ok, (
        f"invariant violated by {report.difference:.4f}: {report.details}"
    )


def test_invariant_holds_when_every_position_is_closed():
    fills = [
        {"symbol": "AAA", "side": "buy",  "quantity": 100, "price": 10.0},
        {"symbol": "AAA", "side": "sell", "quantity": 100, "price": 12.0},
    ]
    book = replay(10000.0, fills)
    assert book["positions"] == []
    assert book["cash"] == pytest.approx(10200.0)
    assert book["realized_pnl"] == pytest.approx(200.0)

    val = value_account(book["cash"], 10000.0, [], {},
                        realized_pnl=book["realized_pnl"])
    assert check_invariant(val).ok


def test_rejected_sell_leaves_cash_untouched():
    """The regression that matters: a bad sell must not move the books."""
    fills = [
        {"symbol": "AAA", "side": "buy",  "quantity": 100, "price": 10.0},
        {"symbol": "ZZZ", "side": "sell", "quantity": 500, "price": 4.0},
    ]
    book = replay(10000.0, fills)
    assert book["cash"] == pytest.approx(9000.0)   # not 9000 + 2000
    assert len(book["rejected"]) == 1
    assert book["rejected"][0]["reason"] == "no_position"

    val = value_account(book["cash"], 10000.0, book["positions"],
                        {"AAA": 10.0}, realized_pnl=book["realized_pnl"])
    assert check_invariant(val).ok


def test_oversized_sell_leaves_cash_untouched():
    fills = [
        {"symbol": "AAA", "side": "buy",  "quantity": 100, "price": 10.0},
        {"symbol": "AAA", "side": "sell", "quantity": 400, "price": 12.0},
    ]
    book = replay(10000.0, fills)
    assert book["cash"] == pytest.approx(9000.0)
    assert book["positions"][0]["quantity"] == 100
    assert book["rejected"][0]["reason"] == "insufficient_shares"


def test_buy_beyond_cash_is_rejected():
    book = replay(1000.0, [
        {"symbol": "AAA", "side": "buy", "quantity": 1000, "price": 10.0},
    ])
    assert book["cash"] == pytest.approx(1000.0)
    assert book["rejected"][0]["reason"] == "insufficient_funds"


def test_the_live_account_numbers_are_detected_as_broken():
    """Feed the actual 2026-09-10 /api/account payload through the check."""
    val = value_account(
        cash=7358.48,
        initial_balance=10000.0,
        positions=[{"symbol": "X", "quantity": 1, "avgPrice": 20648.00}],
        price_map={"X": 20648.00},
        realized_pnl=478.0,
    )
    report = check_invariant(val)
    assert report.ok is False
    # cash + positions = 28006.48 ; initial + realized = 10478.00
    assert report.actual == pytest.approx(28006.48)
    assert abs(report.difference) > 1.0


def test_invariant_tolerance_absorbs_float_noise():
    val = value_account(
        cash=10000.0 + 1e-9, initial_balance=10000.0,
        positions=[], price_map={},
    )
    assert check_invariant(val).ok is True
