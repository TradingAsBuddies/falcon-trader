"""
Strategy Analytics

Thin analytics facade the dashboard uses for strategy leaderboards and
per-strategy performance summaries. It delegates to falcon-core's
``BacktestResultsStore`` (the canonical store of backtest results, keyed by
``strategy_name``) and bridges it to the integer ``strategy_id`` the dashboard's
``active_strategies`` table uses.

Backtest results live in the shared falcon database (PostgreSQL via the shared
DatabaseManager); a SQLite ``db_path`` is accepted only as a legacy fallback.
"""

import logging
from typing import Dict, List, Optional

from falcon_core.backtesting.results_api import BacktestResultsStore

logger = logging.getLogger(__name__)


class StrategyAnalytics:
    """
    Analytics over strategy backtest results.

    Args:
        db_path: Legacy standalone SQLite path. When the shared falcon
            DatabaseManager is available it is preferred and ``db_path`` is
            ignored; ``db_path`` is only used as a fallback.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self._db_manager = None

        # Prefer the shared falcon DatabaseManager so analytics read from the
        # same database the dashboard writes to. Fall back to standalone SQLite.
        try:
            from falcon_core import get_db_manager
            self._db_manager = get_db_manager()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Shared DatabaseManager unavailable, using db_path: %s", e)

        if self._db_manager is not None:
            self._store = BacktestResultsStore(db_manager=self._db_manager)
        else:
            self._store = BacktestResultsStore(db_path=db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _name_to_id_map(self) -> Dict[str, int]:
        """Map strategy_name -> active_strategies.id, when that table exists."""
        if self._db_manager is None:
            return {}
        try:
            rows = self._db_manager.execute(
                "SELECT id, strategy_name FROM active_strategies",
                fetch='all',
            ) or []
        except Exception:
            # Table may not exist yet (created lazily on first activation).
            return {}
        return {r['strategy_name']: r['id'] for r in rows}

    @staticmethod
    def _to_performance(row: Dict) -> Dict:
        """Map a BacktestResultsStore summary row to the dashboard 'performance' shape."""
        row = row or {}
        return {
            "avg_return": float(row.get('avg_return') or 0),
            "avg_win_rate": float(row.get('avg_win_rate') or 0),
            "avg_sharpe": float(row.get('avg_sharpe') or 0),
            "total_trades": int(row.get('total_trades') or 0),
            "total_runs": int(row.get('total_runs') or 0),
            "symbols_tested": int(row.get('symbols_tested') or 0),
        }

    # ------------------------------------------------------------------
    # Public API consumed by the dashboard
    # ------------------------------------------------------------------
    def get_all_strategies_leaderboard(self) -> List[Dict]:
        """
        Rank strategies by backtest performance.

        Returns a list of ``{strategy_id, strategy_name, performance:{...}}``.
        ``strategy_id`` is the ``active_strategies.id`` when the strategy is
        registered there, otherwise ``None``.
        """
        summaries = self._store.get_all_strategies_summary() or []
        name_to_id = self._name_to_id_map()

        leaderboard = []
        for row in summaries:
            name = row.get('strategy_name')
            leaderboard.append({
                "strategy_id": name_to_id.get(name),
                "strategy_name": name,
                "performance": self._to_performance(row),
            })
        return leaderboard

    def get_strategy_summary(self, strategy_id: int) -> Optional[Dict]:
        """
        Detailed performance for a single strategy by integer id.

        Resolves ``strategy_id`` -> ``strategy_name`` via active_strategies, then
        delegates to the backtest store. Returns ``None`` if the strategy is not
        found or has no backtest data.
        """
        strategy_name = None
        if self._db_manager is not None:
            try:
                row = self._db_manager.execute(
                    "SELECT strategy_name FROM active_strategies WHERE id = %s",
                    (strategy_id,),
                    fetch='one',
                )
                if row:
                    strategy_name = row['strategy_name']
            except Exception:
                strategy_name = None

        if not strategy_name:
            return None

        summary = self._store.get_strategy_summary(strategy_name) or {}
        if not summary or not summary.get('total_runs'):
            return None

        return {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "performance": self._to_performance(summary),
        }

    def get_aggregate_statistics(self) -> Dict:
        """Aggregate backtest statistics across all strategies."""
        summaries = self._store.get_all_strategies_summary() or []

        total_strategies = len(summaries)
        total_trades = sum(int(r.get('total_trades') or 0) for r in summaries)
        total_runs = sum(int(r.get('total_runs') or 0) for r in summaries)

        if total_runs > 0:
            avg_return = sum(
                float(r.get('avg_return') or 0) * int(r.get('total_runs') or 0)
                for r in summaries
            ) / total_runs
            avg_win_rate = sum(
                float(r.get('avg_win_rate') or 0) * int(r.get('total_runs') or 0)
                for r in summaries
            ) / total_runs
        else:
            avg_return = 0.0
            avg_win_rate = 0.0

        return {
            "total_strategies": total_strategies,
            "total_runs": total_runs,
            "total_trades": total_trades,
            "avg_return": avg_return,
            "avg_win_rate": avg_win_rate,
        }
