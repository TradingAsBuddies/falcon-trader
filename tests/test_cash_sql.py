"""Integration tests for the cash SQL, against a real SQLite database.

Written because the review of PR #28 was right that everything else in this
suite is a pure function, and the account arithmetic is exactly where that is
not good enough: the conditional-debit statement depends on driver behaviour
(`RETURNING`, and what `DatabaseManager.execute` hands back), which no
pure-function test can reach.

Needs only stdlib sqlite3 and falcon_core.db_manager -- no pandas, no Flask,
no network.
"""

import datetime as dt

import pytest

from falcon_core.db_manager import DatabaseManager


DEBIT_SQL = """UPDATE account
      SET cash = cash - %s, last_updated = %s
    WHERE id = (SELECT id FROM account ORDER BY id LIMIT 1)
      AND cash >= %s
RETURNING cash"""

CREDIT_SQL = """UPDATE account
      SET cash = cash + %s, last_updated = %s
    WHERE id = (SELECT id FROM account ORDER BY id LIMIT 1)"""


@pytest.fixture
def db(tmp_path):
    mgr = DatabaseManager({'db_type': 'sqlite', 'db_path': str(tmp_path / "t.db")})
    mgr.execute(
        "CREATE TABLE account (id INTEGER PRIMARY KEY, cash REAL NOT NULL, "
        "last_updated TEXT NOT NULL)"
    )
    mgr.execute(
        "INSERT INTO account (id, cash, last_updated) VALUES (1, 10000.0, %s)",
        (dt.datetime.now().isoformat(),),
    )
    return mgr


def cash(db):
    return db.execute("SELECT cash FROM account WHERE id = 1", fetch='one')['cash']


def debit(db, amount):
    return db.execute(
        DEBIT_SQL, (amount, dt.datetime.now().isoformat(), amount), fetch='one',
    )


# --------------------------------------------------------------------------
# the behaviour a row count could not give us
# --------------------------------------------------------------------------

def test_sufficient_funds_debit_succeeds(db):
    assert debit(db, 2500.0) is not None
    assert cash(db) == pytest.approx(7500.0)


def test_overdraft_is_refused_and_changes_nothing(db):
    assert debit(db, 10_000.01) is None
    assert cash(db) == pytest.approx(10000.0)


def test_exact_balance_is_allowed(db):
    assert debit(db, 10000.0) is not None
    assert cash(db) == pytest.approx(0.0)


def test_refusal_is_detectable_on_sqlite(db):
    """The regression this test exists for.

    `DatabaseManager.execute` returns `cursor.lastrowid` on SQLite and
    `cursor.rowcount` on Postgres (db_manager.py). Measured here: on SQLite
    lastrowid is **0 for both a successful and a refused UPDATE** -- the two
    are indistinguishable. So the previous `if rows in (0,)` check did not
    merely fail to catch overdrafts, it reported *every* debit as refused,
    which would have rejected every buy as insufficient funds.

    RETURNING distinguishes them on both backends.
    """
    before = cash(db)

    good = db.execute(
        "UPDATE account SET cash = cash - %s WHERE id = 1 AND cash >= %s",
        (100.0, 100.0),
    )
    bad = db.execute(
        "UPDATE account SET cash = cash - %s WHERE id = 1 AND cash >= %s",
        (99_999.0, 99_999.0),
    )
    assert good == bad, (
        "lastrowid is expected to be indistinguishable for applied and "
        "non-applied UPDATEs on SQLite; if this fails the row-count approach "
        "may have become viable, but RETURNING is still correct"
    )
    assert cash(db) == pytest.approx(before - 100.0), (
        "the successful UPDATE must have applied even though its return value "
        "says nothing"
    )

    # RETURNING, by contrast, is definitive in both directions.
    assert debit(db, 50.0) is not None
    assert debit(db, 99_999.0) is None


def test_credit_increases_cash(db):
    db.execute(CREDIT_SQL, (250.0, dt.datetime.now().isoformat()))
    assert cash(db) == pytest.approx(10250.0)


def test_sequential_debits_cannot_overdraw(db):
    assert debit(db, 6000.0) is not None
    assert debit(db, 6000.0) is None
    assert cash(db) == pytest.approx(4000.0)


def test_where_clause_confines_the_update_to_one_row(db):
    """Without a WHERE, the statement rewrote every account row."""
    db.execute(
        "INSERT INTO account (id, cash, last_updated) VALUES (2, 500.0, %s)",
        (dt.datetime.now().isoformat(),),
    )
    debit(db, 1000.0)
    assert cash(db) == pytest.approx(9000.0)
    other = db.execute("SELECT cash FROM account WHERE id = 2", fetch='one')
    assert other['cash'] == pytest.approx(500.0)


# --------------------------------------------------------------------------
# scale-in entry price, as actually executed by SQLite
# --------------------------------------------------------------------------

UPSERT_SQL = """INSERT INTO positions (symbol, quantity, entry_price, last_updated)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT(symbol) DO UPDATE SET
        entry_price = (
            (positions.entry_price * positions.quantity)
            + (excluded.entry_price * excluded.quantity)
        ) / (positions.quantity + excluded.quantity),
        quantity = positions.quantity + excluded.quantity,
        last_updated = %s"""


@pytest.fixture
def posdb(tmp_path):
    mgr = DatabaseManager({'db_type': 'sqlite', 'db_path': str(tmp_path / "p.db")})
    mgr.execute(
        "CREATE TABLE positions (symbol TEXT PRIMARY KEY, quantity INTEGER, "
        "entry_price REAL, last_updated TEXT)"
    )
    return mgr


def add(db, symbol, qty, price):
    now = dt.datetime.now().isoformat()
    db.execute(UPSERT_SQL, (symbol, qty, price, now, now))


def test_scale_in_recomputes_weighted_entry_price(posdb):
    """quantity was incremented while entry_price kept the first fill."""
    add(posdb, "AAA", 100, 10.0)
    add(posdb, "AAA", 100, 20.0)
    row = posdb.execute(
        "SELECT quantity, entry_price FROM positions WHERE symbol = 'AAA'",
        fetch='one',
    )
    assert row['quantity'] == 200
    assert row['entry_price'] == pytest.approx(15.0)


def test_three_way_scale_in(posdb):
    add(posdb, "BBB", 300, 10.0)
    add(posdb, "BBB", 100, 20.0)
    add(posdb, "BBB", 100, 30.0)
    row = posdb.execute(
        "SELECT quantity, entry_price FROM positions WHERE symbol = 'BBB'",
        fetch='one',
    )
    assert row['quantity'] == 500
    assert row['entry_price'] == pytest.approx((300*10 + 100*20 + 100*30) / 500)


def test_first_fill_sets_the_price(posdb):
    add(posdb, "CCC", 50, 7.25)
    row = posdb.execute(
        "SELECT quantity, entry_price FROM positions WHERE symbol = 'CCC'",
        fetch='one',
    )
    assert (row['quantity'], row['entry_price']) == (50, pytest.approx(7.25))


def test_scale_in_does_not_round_to_the_cent(posdb):
    """A weighted average is not required to land on a cent, and truncating it
    biases entry price down -- which overstates P&L on longs."""
    add(posdb, "DDD", 3, 10.00)
    add(posdb, "DDD", 1, 10.03)
    row = posdb.execute(
        "SELECT entry_price FROM positions WHERE symbol = 'DDD'", fetch='one',
    )
    assert row['entry_price'] == pytest.approx(10.0075)
