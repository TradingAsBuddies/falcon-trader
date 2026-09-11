"""Pre-trade risk limits and the kill switch (falcon-trader#26).

The epic asks for an order router, pre-trade risk limits, a kill switch, and
broker reconciliation. Three of those four are broker-agnostic and are
implemented here. The fourth -- the concrete broker adapter -- is not, because
it depends on a choice between the DAS CMD API and IBKR that has not been made;
:class:`BrokerAdapter` states the interface an adapter must satisfy.

What exists today, and why this is needed:

* The only position sizing is "25% of cash" in ``base_engine``. There is no cap
  on gross exposure, no maximum daily loss, no limit on open positions, no PDT
  awareness, no minimum price or dollar-volume floor, and no halted-symbol check.
* ``/api/bot/stop`` stops the market-data thread only. The strategy executor and
  the orchestrator keep running. There is no single flag that stops trading.

Everything here is pure. The kill switch reads a file and an environment
variable rather than Redis or Consul KV so it works before either is deployed
and cannot fail closed on a network partition -- and file plus env means an
operator can halt trading with ``touch``, from any shell, with no service up.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "RiskLimits",
    "RiskDecision",
    "KillSwitch",
    "BrokerAdapter",
    "ReconciliationReport",
    "check_pre_trade",
    "reconcile",
]

#: Default halt file. `touch` it to stop trading; remove it to resume.
DEFAULT_HALT_FILE = Path(
    os.getenv("FALCON_HALT_FILE", "/var/lib/falcon/TRADING_HALTED")
)


@dataclass(frozen=True)
class RiskDecision:
    """Whether an order may be routed."""

    allowed: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    adjusted_quantity: Optional[int] = None

    def __bool__(self) -> bool:
        return self.allowed

    @classmethod
    def ok(cls, quantity: Optional[int] = None) -> "RiskDecision":
        return cls(allowed=True, adjusted_quantity=quantity)

    @classmethod
    def block(cls, reason: str, message: str) -> "RiskDecision":
        return cls(allowed=False, reason=reason, message=message)


@dataclass(frozen=True)
class RiskLimits:
    """Pre-trade limits. Every value is meant to come from config."""

    #: Largest single position as a fraction of total account value.
    max_position_pct: float = 0.25
    #: Largest total exposure as a fraction of total account value.
    #: Below 1.0 means no leverage. The live account showed $20.6k of positions
    #: on a $10k account -- 2x gross with no limit anywhere (#22).
    max_gross_exposure_pct: float = 1.0
    #: Realized + unrealized loss for the day that halts new entries.
    max_daily_loss_pct: float = 0.03
    #: Maximum concurrent open positions.
    max_open_positions: int = 10
    #: Refuse anything under this price. Penny stocks were being recommended
    #: (falcon-screener#4) and traded.
    min_price: float = 5.0
    #: Refuse anything thinner than this in average daily dollar volume.
    min_dollar_volume: float = 5_000_000.0
    #: Day-trade count allowed in a rolling 5 session window on a margin
    #: account under $25k. Set to None to disable the check.
    max_day_trades: Optional[int] = 3
    #: Symbols never to route.
    blacklist: frozenset = frozenset()

    @classmethod
    def from_config(cls, config: Optional[Mapping]) -> "RiskLimits":
        config = config or {}
        return cls(
            max_position_pct=float(config.get("max_position_pct", cls.max_position_pct)),
            max_gross_exposure_pct=float(
                config.get("max_gross_exposure_pct", cls.max_gross_exposure_pct)
            ),
            max_daily_loss_pct=float(
                config.get("max_daily_loss_pct", cls.max_daily_loss_pct)
            ),
            max_open_positions=int(
                config.get("max_open_positions", cls.max_open_positions)
            ),
            min_price=float(config.get("min_price", cls.min_price)),
            min_dollar_volume=float(
                config.get("min_dollar_volume", cls.min_dollar_volume)
            ),
            max_day_trades=config.get("max_day_trades", cls.max_day_trades),
            blacklist=frozenset(
                s.upper() for s in (config.get("blacklist") or ())
            ),
        )


# --------------------------------------------------------------------------
# kill switch
# --------------------------------------------------------------------------

class KillSwitch:
    """One flag every trading loop honors.

    Three independent ways to halt, because the point of a kill switch is that
    it works when things are broken:

    * ``FALCON_TRADING_ENABLED=0`` in the environment (deploy-time)
    * a halt file on disk (``touch`` it from any shell, no service needed)
    * :meth:`halt` in-process (a loop halting itself on a risk breach)

    Any one of them halts. All three must be clear to trade.
    """

    def __init__(self, halt_file: Optional[Path] = None, env=None):
        self.halt_file = Path(halt_file) if halt_file else DEFAULT_HALT_FILE
        self._env = os.environ if env is None else env
        self._in_process_halt: Optional[str] = None

    # -- state --------------------------------------------------------
    def _env_enabled(self) -> bool:
        raw = (self._env.get("FALCON_TRADING_ENABLED", "1") or "").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def _file_halted(self) -> bool:
        try:
            return self.halt_file.exists()
        except OSError:
            # If we cannot even stat the halt file, assume halted. A kill switch
            # that fails open is not a kill switch.
            logger.error("Cannot read halt file %s; treating as halted", self.halt_file)
            return True

    def is_trading_enabled(self) -> bool:
        return (
            self._in_process_halt is None
            and self._env_enabled()
            and not self._file_halted()
        )

    def reason(self) -> Optional[str]:
        if self._in_process_halt is not None:
            return self._in_process_halt
        if not self._env_enabled():
            return "FALCON_TRADING_ENABLED is not set to a truthy value"
        if self._file_halted():
            return f"Halt file present: {self.halt_file}"
        return None

    # -- control ------------------------------------------------------
    def halt(self, reason: str, persist: bool = True) -> None:
        """Stop trading. `persist` also writes the halt file."""
        self._in_process_halt = reason
        logger.critical("TRADING HALTED: %s", reason)
        if not persist:
            return
        try:
            self.halt_file.parent.mkdir(parents=True, exist_ok=True)
            self.halt_file.write_text(
                f"{_dt.datetime.now().isoformat()}\n{reason}\n"
            )
        except OSError as exc:
            logger.error("Could not write halt file %s: %s", self.halt_file, exc)

    def resume(self) -> None:
        """Clear the in-process and file halts. Never clears the env flag."""
        self._in_process_halt = None
        try:
            self.halt_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("Could not remove halt file %s: %s", self.halt_file, exc)
        logger.warning("Trading resumed")


# --------------------------------------------------------------------------
# pre-trade checks
# --------------------------------------------------------------------------

def check_pre_trade(
    *,
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    limits: RiskLimits,
    total_value: float,
    cash: float,
    open_positions: Sequence[Mapping] = (),
    positions_value: float = 0.0,
    daily_pnl: float = 0.0,
    avg_dollar_volume: Optional[float] = None,
    is_halted_symbol: bool = False,
    day_trades_used: int = 0,
    kill_switch: Optional[KillSwitch] = None,
) -> RiskDecision:
    """Every pre-trade limit, in order. First failure wins.

    Sells are only subject to the kill switch and the halted-symbol check:
    refusing to *reduce* risk because an exposure limit is breached would be
    backwards.
    """
    symbol = (symbol or "").upper()
    side = (side or "").lower()

    if kill_switch is not None and not kill_switch.is_trading_enabled():
        return RiskDecision.block("trading_halted", kill_switch.reason() or "halted")

    if is_halted_symbol:
        return RiskDecision.block(
            "symbol_halted", f"{symbol} is halted by the exchange",
        )

    if side == "sell":
        return RiskDecision.ok(quantity)

    if symbol in limits.blacklist:
        return RiskDecision.block("blacklisted", f"{symbol} is blacklisted")

    if price < limits.min_price:
        return RiskDecision.block(
            "below_min_price",
            f"{symbol} at ${price:,.2f} is below the ${limits.min_price:,.2f} floor",
        )

    if avg_dollar_volume is not None and avg_dollar_volume < limits.min_dollar_volume:
        return RiskDecision.block(
            "insufficient_liquidity",
            f"{symbol} average dollar volume ${avg_dollar_volume:,.0f} is below "
            f"${limits.min_dollar_volume:,.0f}",
        )

    if total_value <= 0:
        return RiskDecision.block(
            "no_account_value", "Account total value is zero or unknown",
        )

    # Daily loss halt.
    daily_loss_pct = -daily_pnl / total_value if daily_pnl < 0 else 0.0
    if daily_loss_pct >= limits.max_daily_loss_pct:
        return RiskDecision.block(
            "max_daily_loss",
            f"Daily loss {daily_loss_pct:.2%} has reached the "
            f"{limits.max_daily_loss_pct:.2%} limit",
        )

    held_symbols = {str(p.get("symbol", "")).upper() for p in open_positions}
    if symbol not in held_symbols and len(held_symbols) >= limits.max_open_positions:
        return RiskDecision.block(
            "max_open_positions",
            f"Already holding {len(held_symbols)} positions "
            f"(limit {limits.max_open_positions})",
        )

    order_value = price * quantity

    if order_value > cash:
        return RiskDecision.block(
            "insufficient_cash",
            f"Order value ${order_value:,.2f} exceeds cash ${cash:,.2f}",
        )

    # Position concentration, measured against the position that would result.
    existing_value = sum(
        float(p.get("quantity", 0)) * float(p.get("avgPrice", 0) or 0)
        for p in open_positions
        if str(p.get("symbol", "")).upper() == symbol
    )
    resulting_pct = (existing_value + order_value) / total_value
    if resulting_pct > limits.max_position_pct:
        return RiskDecision.block(
            "max_position_size",
            f"{symbol} would be {resulting_pct:.1%} of the account "
            f"(limit {limits.max_position_pct:.1%})",
        )

    resulting_gross = (positions_value + order_value) / total_value
    if resulting_gross > limits.max_gross_exposure_pct:
        return RiskDecision.block(
            "max_gross_exposure",
            f"Gross exposure would be {resulting_gross:.1%} "
            f"(limit {limits.max_gross_exposure_pct:.1%})",
        )

    if limits.max_day_trades is not None and day_trades_used >= limits.max_day_trades:
        return RiskDecision.block(
            "pdt_limit",
            f"{day_trades_used} day trades used in the rolling window "
            f"(limit {limits.max_day_trades})",
        )

    return RiskDecision.ok(quantity)


# --------------------------------------------------------------------------
# broker reconciliation
# --------------------------------------------------------------------------

class BrokerAdapter:
    """Interface a concrete broker adapter must satisfy.

    Not implemented here on purpose. The epic requires choosing between the DAS
    CMD API (already stubbed in ``das_execution.py``) and IBKR, and that choice
    belongs to the operator, not to this module. Recording it in SPEC.md is a
    listed task on the issue.
    """

    def positions(self) -> List[Mapping]:
        """``[{'symbol', 'quantity', 'avgPrice'}, ...]`` as the broker sees it."""
        raise NotImplementedError

    def cash(self) -> float:
        """Settled cash as the broker sees it."""
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, quantity: int,
                    price: Optional[float], client_order_id: str) -> Mapping:
        """Route an order. `client_order_id` must make retries idempotent."""
        raise NotImplementedError


@dataclass
class ReconciliationReport:
    """Difference between our books and the broker's."""

    matched: bool
    cash_difference: float = 0.0
    position_differences: List[dict] = field(default_factory=list)
    missing_locally: List[str] = field(default_factory=list)
    missing_at_broker: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.matched:
            return "Books match the broker"
        parts = []
        if abs(self.cash_difference) > 0.01:
            parts.append(f"cash off by ${self.cash_difference:,.2f}")
        if self.position_differences:
            parts.append(f"{len(self.position_differences)} position mismatch(es)")
        if self.missing_locally:
            parts.append(f"unknown locally: {', '.join(self.missing_locally)}")
        if self.missing_at_broker:
            parts.append(f"absent at broker: {', '.join(self.missing_at_broker)}")
        return "; ".join(parts)


def reconcile(
    local_positions: Sequence[Mapping],
    local_cash: float,
    broker_positions: Sequence[Mapping],
    broker_cash: float,
    tolerance: float = 0.01,
) -> ReconciliationReport:
    """Compare our books against the broker's. Drift should halt trading.

    Pure, so it can be tested without a broker connection -- which matters,
    because the case that needs testing is the one where they disagree.
    """
    local = {
        str(p["symbol"]).upper(): int(p.get("quantity", 0))
        for p in local_positions
    }
    remote = {
        str(p["symbol"]).upper(): int(p.get("quantity", 0))
        for p in broker_positions
    }

    differences = []
    for symbol in sorted(set(local) | set(remote)):
        ours, theirs = local.get(symbol, 0), remote.get(symbol, 0)
        if ours != theirs:
            differences.append({
                "symbol": symbol, "local": ours, "broker": theirs,
                "difference": ours - theirs,
            })

    cash_difference = local_cash - broker_cash

    return ReconciliationReport(
        matched=not differences and abs(cash_difference) <= tolerance,
        cash_difference=cash_difference,
        position_differences=differences,
        missing_locally=sorted(set(remote) - set(local)),
        missing_at_broker=sorted(set(local) - set(remote)),
    )
