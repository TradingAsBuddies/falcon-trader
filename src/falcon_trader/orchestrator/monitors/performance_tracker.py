"""
Performance Tracker

Tracks routing decisions, trade outcomes, and strategy performance
for continuous learning and optimization.

Features:
- Log routing decisions with confidence scores
- Track trade entries and exits
- Calculate performance metrics by strategy + stock type
- Aggregate win rates, average returns, drawdowns
- Adaptive routing confidence adjustments
- Performance reports and insights
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from db_manager import DatabaseManager


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy: str
    stock_type: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_profit_winners: float
    avg_loss_losers: float
    total_return: float
    max_drawdown: float
    avg_hold_days: float
    sharpe_ratio: float
    confidence_accuracy: float


class PerformanceTracker:
    """
    Track and analyze strategy performance

    Database Schema:
    - routing_decisions: Log all routing decisions
    - trade_tracking: Track all trades from entry to exit
    - strategy_metrics: Aggregated performance by strategy + stock type
    """

    def __init__(self, config: dict, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize performance tracker

        Args:
            config: Configuration dictionary
            db_manager: Optional DatabaseManager instance
        """
        self.config = config

        # Initialize database manager
        if db_manager:
            self.db = db_manager
        else:
            db_config = {'db_type': 'sqlite', 'db_path': 'paper_trading.db'}
            self.db = DatabaseManager(db_config)

        # Create tables if they don't exist
        self._create_tables()

        print("[TRACKER] Performance Tracker initialized")

    def _create_tables(self):
        """Create tracking tables if they don't exist"""
        try:
            # Routing decisions table
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    selected_strategy TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            # Trade tracking table
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS trade_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_date TEXT,
                    exit_price REAL,
                    quantity INTEGER NOT NULL,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    hold_days INTEGER,
                    exit_reason TEXT,
                    routing_confidence REAL,
                    was_profitable INTEGER
                )
            """)

            # Strategy metrics table (aggregated)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    stock_type TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_profit REAL DEFAULT 0,
                    avg_profit_winners REAL DEFAULT 0,
                    avg_loss_losers REAL DEFAULT 0,
                    total_return REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    avg_hold_days REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(strategy, stock_type, period_start, period_end)
                )
            """)

            print("[TRACKER] Database tables ready")

        except Exception as e:
            print(f"[ERROR] Error creating tables: {e}")

    def log_routing_decision(self, decision_id: str, symbol: str, strategy: str,
                            classification: str, confidence: float, reason: str = ""):
        """
        Log a routing decision

        Args:
            decision_id: Unique decision ID
            symbol: Stock symbol
            strategy: Selected strategy
            classification: Stock classification
            confidence: Routing confidence score
            reason: Routing reason
        """
        try:
            self.db.execute("""
                INSERT OR REPLACE INTO routing_decisions (
                    decision_id, symbol, selected_strategy, classification,
                    confidence, reason, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                decision_id, symbol, strategy, classification,
                confidence, reason, datetime.now().isoformat()
            ))

            print(f"[TRACKER] Logged routing: {symbol} -> {strategy} ({confidence:.1%})")

        except Exception as e:
            print(f"[ERROR] Error logging routing decision: {e}")

    def log_trade_entry(self, trade_id: str, symbol: str, strategy: str,
                       classification: str, entry_price: float, quantity: int,
                       routing_confidence: float):
        """
        Log a trade entry

        Args:
            trade_id: Unique trade ID
            symbol: Stock symbol
            strategy: Strategy used
            classification: Stock classification
            entry_price: Entry price
            quantity: Number of shares
            routing_confidence: Confidence from routing decision
        """
        try:
            self.db.execute("""
                INSERT OR REPLACE INTO trade_tracking (
                    trade_id, symbol, strategy, classification,
                    entry_date, entry_price, quantity, routing_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                trade_id, symbol, strategy, classification,
                datetime.now().isoformat(), entry_price, quantity,
                routing_confidence
            ))

            print(f"[TRACKER] Logged entry: {symbol} @ ${entry_price:.2f}")

        except Exception as e:
            print(f"[ERROR] Error logging trade entry: {e}")

    def log_trade_exit(self, trade_id: str, exit_price: float, exit_reason: str = ""):
        """
        Log a trade exit and calculate metrics

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        try:
            # Get trade entry data
            trade = self.db.execute(
                "SELECT * FROM trade_tracking WHERE trade_id = %s",
                (trade_id,),
                fetch='one'
            )

            if not trade:
                print(f"[WARNING] Trade {trade_id} not found")
                return

            # Calculate metrics
            entry_price = float(trade['entry_price'])
            quantity = int(trade['quantity'])
            entry_date = datetime.fromisoformat(trade['entry_date'])
            exit_date = datetime.now()

            profit_loss = (exit_price - entry_price) * quantity
            profit_loss_pct = (exit_price - entry_price) / entry_price
            hold_days = (exit_date - entry_date).days
            was_profitable = 1 if profit_loss > 0 else 0

            # Update trade record
            self.db.execute("""
                UPDATE trade_tracking SET
                    exit_date = %s,
                    exit_price = %s,
                    profit_loss = %s,
                    profit_loss_pct = %s,
                    hold_days = %s,
                    exit_reason = %s,
                    was_profitable = %s
                WHERE trade_id = %s
            """, (
                exit_date.isoformat(), exit_price, profit_loss,
                profit_loss_pct, hold_days, exit_reason, was_profitable,
                trade_id
            ))

            print(f"[TRACKER] Logged exit: {trade['symbol']} @ ${exit_price:.2f} ({profit_loss_pct:+.1%})")

            # Update aggregated metrics
            self._update_strategy_metrics(
                trade['strategy'],
                trade['classification']
            )

        except Exception as e:
            print(f"[ERROR] Error logging trade exit: {e}")

    def _update_strategy_metrics(self, strategy: str, stock_type: str):
        """
        Update aggregated metrics for a strategy + stock type

        Args:
            strategy: Strategy name
            stock_type: Stock classification
        """
        try:
            # Calculate metrics for last 30 days
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
            end_date = datetime.now().isoformat()

            # Get completed trades
            trades = self.db.execute("""
                SELECT * FROM trade_tracking
                WHERE strategy = %s
                  AND classification = %s
                  AND exit_date IS NOT NULL
                  AND entry_date >= %s
            """, (strategy, stock_type, start_date), fetch='all')

            if not trades:
                return

            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['was_profitable'] == 1)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            profits = [float(t['profit_loss_pct']) for t in trades]
            avg_profit = sum(profits) / len(profits) if profits else 0

            winners = [float(t['profit_loss_pct']) for t in trades if t['was_profitable'] == 1]
            avg_profit_winners = sum(winners) / len(winners) if winners else 0

            losers = [float(t['profit_loss_pct']) for t in trades if t['was_profitable'] == 0]
            avg_loss_losers = sum(losers) / len(losers) if losers else 0

            total_return = sum(profits)

            # Calculate max drawdown
            cumulative = 0
            peak = 0
            max_dd = 0
            for profit in profits:
                cumulative += profit
                peak = max(peak, cumulative)
                drawdown = peak - cumulative
                max_dd = max(max_dd, drawdown)

            # Average hold days
            hold_days = [int(t['hold_days']) for t in trades if t['hold_days'] is not None]
            avg_hold_days = sum(hold_days) / len(hold_days) if hold_days else 0

            # Sharpe ratio (simplified)
            if len(profits) > 1:
                import numpy as np
                std = np.std(profits)
                sharpe = (avg_profit / std) if std > 0 else 0
            else:
                sharpe = 0

            # Update or insert metrics
            self.db.execute("""
                INSERT OR REPLACE INTO strategy_metrics (
                    strategy, stock_type, period_start, period_end,
                    total_trades, winning_trades, losing_trades, win_rate,
                    avg_profit, avg_profit_winners, avg_loss_losers,
                    total_return, max_drawdown, avg_hold_days, sharpe_ratio,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                strategy, stock_type, start_date, end_date,
                total_trades, winning_trades, losing_trades, win_rate,
                avg_profit, avg_profit_winners, avg_loss_losers,
                total_return, max_dd, avg_hold_days, sharpe,
                datetime.now().isoformat()
            ))

        except Exception as e:
            print(f"[ERROR] Error updating metrics: {e}")

    def get_strategy_performance(self, strategy: str, stock_type: Optional[str] = None,
                                days: int = 30) -> List[StrategyPerformance]:
        """
        Get performance metrics for a strategy

        Args:
            strategy: Strategy name
            stock_type: Optional stock classification filter
            days: Number of days to analyze

        Returns:
            List of StrategyPerformance objects
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            if stock_type:
                metrics = self.db.execute("""
                    SELECT * FROM strategy_metrics
                    WHERE strategy = %s
                      AND stock_type = %s
                      AND period_start >= %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (strategy, stock_type, start_date), fetch='all')
            else:
                metrics = self.db.execute("""
                    SELECT * FROM strategy_metrics
                    WHERE strategy = %s
                      AND period_start >= %s
                    ORDER BY stock_type, updated_at DESC
                """, (strategy, start_date), fetch='all')

            results = []
            for m in metrics or []:
                # Calculate confidence accuracy
                confidence_accuracy = self._calculate_confidence_accuracy(
                    m['strategy'],
                    m['stock_type'],
                    days
                )

                results.append(StrategyPerformance(
                    strategy=m['strategy'],
                    stock_type=m['stock_type'],
                    total_trades=m['total_trades'],
                    winning_trades=m['winning_trades'],
                    losing_trades=m['losing_trades'],
                    win_rate=m['win_rate'],
                    avg_profit=m['avg_profit'],
                    avg_profit_winners=m['avg_profit_winners'],
                    avg_loss_losers=m['avg_loss_losers'],
                    total_return=m['total_return'],
                    max_drawdown=m['max_drawdown'],
                    avg_hold_days=m['avg_hold_days'],
                    sharpe_ratio=m['sharpe_ratio'],
                    confidence_accuracy=confidence_accuracy
                ))

            return results

        except Exception as e:
            print(f"[ERROR] Error getting performance: {e}")
            return []

    def _calculate_confidence_accuracy(self, strategy: str, stock_type: str,
                                      days: int = 30) -> float:
        """
        Calculate how accurate the routing confidence was

        Args:
            strategy: Strategy name
            stock_type: Stock classification
            days: Number of days

        Returns:
            Confidence accuracy score (0.0-1.0)
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            # Get trades with routing confidence
            trades = self.db.execute("""
                SELECT routing_confidence, was_profitable
                FROM trade_tracking
                WHERE strategy = %s
                  AND classification = %s
                  AND exit_date IS NOT NULL
                  AND entry_date >= %s
                  AND routing_confidence IS NOT NULL
            """, (strategy, stock_type, start_date), fetch='all')

            if not trades:
                return 0.5  # Neutral

            # Calculate correlation between confidence and success
            correct = 0
            total = 0

            for trade in trades:
                confidence = float(trade['routing_confidence'])
                profitable = trade['was_profitable'] == 1

                # High confidence trades should be profitable
                if confidence > 0.75 and profitable:
                    correct += 1
                # Low confidence trades being unprofitable is also correct
                elif confidence < 0.5 and not profitable:
                    correct += 1

                total += 1

            return correct / total if total > 0 else 0.5

        except Exception as e:
            print(f"[ERROR] Error calculating confidence accuracy: {e}")
            return 0.5

    def get_performance_report(self, days: int = 30) -> Dict:
        """
        Get comprehensive performance report

        Args:
            days: Number of days to analyze

        Returns:
            Dict with performance data
        """
        try:
            report = {
                'period_days': days,
                'generated_at': datetime.now().isoformat(),
                'strategies': {},
                'overall': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0
                }
            }

            # Get all strategies
            strategies = ['rsi_mean_reversion', 'momentum_breakout', 'bollinger_mean_reversion']

            for strategy in strategies:
                perf = self.get_strategy_performance(strategy, days=days)

                if perf:
                    report['strategies'][strategy] = {}

                    for p in perf:
                        report['strategies'][strategy][p.stock_type] = {
                            'total_trades': p.total_trades,
                            'winning_trades': p.winning_trades,
                            'losing_trades': p.losing_trades,
                            'win_rate': p.win_rate,
                            'avg_profit': p.avg_profit,
                            'avg_profit_winners': p.avg_profit_winners,
                            'avg_loss_losers': p.avg_loss_losers,
                            'total_return': p.total_return,
                            'max_drawdown': p.max_drawdown,
                            'avg_hold_days': p.avg_hold_days,
                            'sharpe_ratio': p.sharpe_ratio,
                            'confidence_accuracy': p.confidence_accuracy
                        }

                        # Aggregate overall
                        report['overall']['total_trades'] += p.total_trades
                        report['overall']['winning_trades'] += p.winning_trades
                        report['overall']['total_return'] += p.total_return

            # Calculate overall win rate
            if report['overall']['total_trades'] > 0:
                report['overall']['win_rate'] = (
                    report['overall']['winning_trades'] / report['overall']['total_trades']
                )

            return report

        except Exception as e:
            print(f"[ERROR] Error generating report: {e}")
            return {}

    def get_top_performing_strategies(self, metric: str = 'win_rate',
                                     days: int = 30, limit: int = 5) -> List[Tuple]:
        """
        Get top performing strategies

        Args:
            metric: Metric to rank by (win_rate, avg_profit, total_return, sharpe_ratio)
            days: Number of days
            limit: Max results

        Returns:
            List of (strategy, stock_type, metric_value) tuples
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            # Map metric names to columns
            metric_col = {
                'win_rate': 'win_rate',
                'avg_profit': 'avg_profit',
                'total_return': 'total_return',
                'sharpe_ratio': 'sharpe_ratio'
            }.get(metric, 'win_rate')

            results = self.db.execute(f"""
                SELECT strategy, stock_type, {metric_col}, total_trades
                FROM strategy_metrics
                WHERE period_start >= %s
                  AND total_trades >= 3
                ORDER BY {metric_col} DESC
                LIMIT %s
            """, (start_date, limit), fetch='all')

            return [
                (r['strategy'], r['stock_type'], float(r[metric_col]))
                for r in (results or [])
            ]

        except Exception as e:
            print(f"[ERROR] Error getting top performers: {e}")
            return []

    def get_routing_accuracy(self, days: int = 30) -> Dict:
        """
        Analyze routing decision accuracy

        Args:
            days: Number of days

        Returns:
            Dict with routing accuracy metrics
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            # Get all routing decisions with outcomes
            results = self.db.execute("""
                SELECT
                    rd.symbol,
                    rd.selected_strategy,
                    rd.classification,
                    rd.confidence,
                    tt.was_profitable,
                    tt.profit_loss_pct
                FROM routing_decisions rd
                LEFT JOIN trade_tracking tt ON rd.symbol = tt.symbol
                    AND rd.selected_strategy = tt.strategy
                    AND tt.entry_date >= rd.timestamp
                WHERE rd.timestamp >= %s
                  AND tt.exit_date IS NOT NULL
            """, (start_date,), fetch='all')

            if not results:
                return {'total_decisions': 0, 'accuracy': 0.0}

            # Analyze accuracy
            high_conf_correct = 0
            high_conf_total = 0
            medium_conf_correct = 0
            medium_conf_total = 0
            low_conf_correct = 0
            low_conf_total = 0

            for r in results:
                confidence = float(r['confidence'])
                profitable = r['was_profitable'] == 1

                if confidence >= 0.80:
                    high_conf_total += 1
                    if profitable:
                        high_conf_correct += 1
                elif confidence >= 0.60:
                    medium_conf_total += 1
                    if profitable:
                        medium_conf_correct += 1
                else:
                    low_conf_total += 1
                    if profitable:
                        low_conf_correct += 1

            return {
                'total_decisions': len(results),
                'high_confidence': {
                    'total': high_conf_total,
                    'correct': high_conf_correct,
                    'accuracy': high_conf_correct / high_conf_total if high_conf_total > 0 else 0
                },
                'medium_confidence': {
                    'total': medium_conf_total,
                    'correct': medium_conf_correct,
                    'accuracy': medium_conf_correct / medium_conf_total if medium_conf_total > 0 else 0
                },
                'low_confidence': {
                    'total': low_conf_total,
                    'correct': low_conf_correct,
                    'accuracy': low_conf_correct / low_conf_total if low_conf_total > 0 else 0
                }
            }

        except Exception as e:
            print(f"[ERROR] Error calculating routing accuracy: {e}")
            return {}

    def print_performance_summary(self, days: int = 30):
        """
        Print formatted performance summary

        Args:
            days: Number of days to analyze
        """
        print(f"\n{'=' * 80}")
        print(f"PERFORMANCE SUMMARY - Last {days} Days")
        print(f"{'=' * 80}\n")

        report = self.get_performance_report(days)

        if not report.get('strategies'):
            print("No performance data available yet")
            return

        # Overall metrics
        print(f"[OVERALL]")
        print(f"  Total Trades: {report['overall']['total_trades']}")
        print(f"  Winning Trades: {report['overall']['winning_trades']}")
        print(f"  Win Rate: {report['overall']['win_rate']:.1%}")
        print(f"  Total Return: {report['overall']['total_return']:+.2%}")

        # Strategy breakdown
        for strategy, stock_types in report['strategies'].items():
            print(f"\n[{strategy.upper().replace('_', ' ')}]")

            for stock_type, metrics in stock_types.items():
                print(f"  {stock_type}:")
                print(f"    Trades: {metrics['total_trades']} ({metrics['winning_trades']}W / {metrics['losing_trades']}L)")
                print(f"    Win Rate: {metrics['win_rate']:.1%}")
                print(f"    Avg Profit: {metrics['avg_profit']:+.2%}")
                print(f"    Total Return: {metrics['total_return']:+.2%}")
                print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"    Confidence Accuracy: {metrics['confidence_accuracy']:.1%}")

        # Top performers
        print(f"\n[TOP PERFORMERS BY WIN RATE]")
        top = self.get_top_performing_strategies('win_rate', days, limit=3)
        for i, (strategy, stock_type, value) in enumerate(top, 1):
            print(f"  {i}. {strategy} ({stock_type}): {value:.1%}")

        print(f"\n{'=' * 80}")
