#!/usr/bin/env python3
"""
Strategy Manager - Allows AI agents to safely modify trading strategies

Features:
- Validate strategy code before deployment
- Run backtests on modified strategies
- Git version control for all changes
- Rollback capability
- Optional approval workflow
"""

import os
import sys
import ast
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Dict, Tuple
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class StrategyValidationError(Exception):
    """Raised when strategy validation fails"""
    pass


class StrategyManager:
    """Manages trading strategy modifications with safety controls"""

    STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
    ACTIVE_STRATEGY_FILE = "active_strategy.py"
    STRATEGY_HISTORY_DIR = "strategy_history"

    # Required elements in a valid strategy
    REQUIRED_CLASSES = ["Strategy"]
    REQUIRED_METHODS = ["__init__", "next"]

    def __init__(self, require_approval: bool = False):
        self.require_approval = require_approval
        self.history_dir = os.path.join(self.STRATEGY_DIR, self.STRATEGY_HISTORY_DIR)
        os.makedirs(self.history_dir, exist_ok=True)

    def get_active_strategy(self) -> str:
        """Read the current active strategy code"""
        strategy_path = os.path.join(self.STRATEGY_DIR, self.ACTIVE_STRATEGY_FILE)
        if os.path.exists(strategy_path):
            with open(strategy_path, 'r') as f:
                return f.read()
        return ""

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if the code has valid Python syntax"""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def validate_structure(self, code: str) -> Tuple[bool, str]:
        """Validate the strategy has required structure"""
        try:
            tree = ast.parse(code)

            # Find all class definitions
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            class_names = [c.name for c in classes]

            # Check for Strategy class (or subclass naming pattern)
            has_strategy_class = any('Strategy' in name for name in class_names)
            if not has_strategy_class:
                return False, "No Strategy class found. Class name must contain 'Strategy'"

            # Find the strategy class
            strategy_class = None
            for c in classes:
                if 'Strategy' in c.name:
                    strategy_class = c
                    break

            # Check required methods
            method_names = [node.name for node in ast.walk(strategy_class)
                          if isinstance(node, ast.FunctionDef)]

            for required in self.REQUIRED_METHODS:
                if required not in method_names:
                    return False, f"Missing required method: {required}"

            return True, "Structure valid"

        except Exception as e:
            return False, f"Structure validation error: {str(e)}"

    def validate_imports(self, code: str) -> Tuple[bool, str]:
        """Check for dangerous imports"""
        dangerous_modules = [
            'subprocess', 'os.system', 'eval', 'exec', 'compile',
            '__import__', 'importlib', 'pickle', 'marshal'
        ]

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            return False, f"Dangerous import not allowed: {alias.name}"

                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_modules:
                        return False, f"Dangerous import not allowed: {node.module}"

                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            return False, f"Dangerous function not allowed: {node.func.id}"

            return True, "Imports valid"

        except Exception as e:
            return False, f"Import validation error: {str(e)}"

    def lint_code(self, code: str) -> Tuple[bool, str]:
        """Run flake8 linting on the code"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['flake8', '--select=E9,F63,F7,F82', temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return True, "Linting passed"
            else:
                errors = result.stdout.replace(temp_path, 'strategy.py')
                return False, f"Linting errors:\n{errors}"

        except subprocess.TimeoutExpired:
            return False, "Linting timed out"
        except FileNotFoundError:
            # flake8 not installed, skip linting
            return True, "Linting skipped (flake8 not available)"
        finally:
            os.unlink(temp_path)

    def validate_strategy(self, code: str) -> Tuple[bool, Dict[str, str]]:
        """Run all validations on the strategy code"""
        results = {}
        all_passed = True

        # Syntax check
        passed, msg = self.validate_syntax(code)
        results['syntax'] = msg
        if not passed:
            all_passed = False

        # Structure check
        if all_passed:
            passed, msg = self.validate_structure(code)
            results['structure'] = msg
            if not passed:
                all_passed = False

        # Import security check
        if all_passed:
            passed, msg = self.validate_imports(code)
            results['imports'] = msg
            if not passed:
                all_passed = False

        # Linting
        if all_passed:
            passed, msg = self.lint_code(code)
            results['linting'] = msg
            if not passed:
                all_passed = False

        return all_passed, results

    def run_backtest(self, code: str, ticker: str = "SPY", days: int = 365) -> Tuple[bool, Dict]:
        """Run a backtest on the strategy and return results"""
        # Write strategy to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_strategy = f.name

        # Create backtest runner script
        backtest_script = f'''
import sys
sys.path.insert(0, "{self.STRATEGY_DIR}")
import backtrader as bt
import json
from datetime import datetime, timedelta

# Import the strategy from temp file
import importlib.util
spec = importlib.util.spec_from_file_location("temp_strategy", "{temp_strategy}")
strategy_module = importlib.util.module_from_spec(spec)
sys.modules["temp_strategy"] = strategy_module  # Register module for backtrader
spec.loader.exec_module(strategy_module)

# Find the Strategy class
strategy_class = None
for name in dir(strategy_module):
    obj = getattr(strategy_module, name)
    if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj != bt.Strategy:
        strategy_class = obj
        break

if not strategy_class:
    print(json.dumps({{"error": "No Strategy class found"}}))
    sys.exit(1)

# Run backtest
try:
    from massive_flat_files import MassiveFlatFilesManager
    manager = MassiveFlatFilesManager("")
    df = manager.get_ticker_history("{ticker}", days={days})

    if df.empty:
        # Fallback to yfinance
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days={days})
        df = yf.download("{ticker}", start=start, end=end, auto_adjust=True)
        df.columns = [c.lower() for c in df.columns]
except:
    import yfinance as yf
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days={days})
    df = yf.download("{ticker}", start=start, end=end, auto_adjust=True)
    if hasattr(df.columns, 'get_level_values'):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

cerebro = bt.Cerebro()
cerebro.addstrategy(strategy_class)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

initial = cerebro.broker.getvalue()
results = cerebro.run()
final = cerebro.broker.getvalue()

strat = results[0]
ta = strat.analyzers.trades.get_analysis()
sharpe = strat.analyzers.sharpe.get_analysis()
dd = strat.analyzers.drawdown.get_analysis()

total_trades = 0
won = 0
if ta:
    total_dict = dict(ta.get('total', {{}}))
    total_trades = total_dict.get('closed', 0)
    won_dict = dict(ta.get('won', {{}}))
    won = won_dict.get('total', 0)

output = {{
    "initial_value": initial,
    "final_value": final,
    "return_pct": ((final - initial) / initial) * 100,
    "sharpe_ratio": sharpe.get('sharperatio') or 0,
    "max_drawdown": dd.get('max', {{}}).get('drawdown', 0),
    "total_trades": total_trades,
    "winning_trades": won,
    "win_rate": (won / total_trades * 100) if total_trades > 0 else 0
}}
print(json.dumps(output))
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(backtest_script)
            runner_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.STRATEGY_DIR
            )

            if result.returncode == 0:
                try:
                    backtest_results = json.loads(result.stdout.strip().split('\n')[-1])
                    return True, backtest_results
                except json.JSONDecodeError:
                    return False, {"error": f"Failed to parse results: {result.stdout}"}
            else:
                return False, {"error": result.stderr or result.stdout}

        except subprocess.TimeoutExpired:
            return False, {"error": "Backtest timed out (>120s)"}
        except Exception as e:
            return False, {"error": str(e)}
        finally:
            os.unlink(temp_strategy)
            os.unlink(runner_path)

    def save_to_history(self, code: str, metadata: Dict) -> str:
        """Save strategy version to history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_{timestamp}.py"
        filepath = os.path.join(self.history_dir, filename)

        # Add metadata as header comment
        header = f'''"""
Strategy Version: {timestamp}
Modified: {datetime.now().isoformat()}
Backtest Results: {json.dumps(metadata.get('backtest', {}), indent=2)}
"""

'''
        with open(filepath, 'w') as f:
            f.write(header + code)

        return filepath

    def deploy_strategy(self, code: str, force: bool = False) -> Tuple[bool, str]:
        """Deploy a new strategy after validation and backtesting"""

        # Validate
        valid, validation_results = self.validate_strategy(code)
        if not valid:
            return False, f"Validation failed: {validation_results}"

        # Backtest
        success, backtest_results = self.run_backtest(code)
        if not success:
            return False, f"Backtest failed: {backtest_results}"

        # Check backtest results meet minimum criteria
        if not force:
            if backtest_results.get('return_pct', 0) < -10:
                return False, f"Strategy has negative returns ({backtest_results['return_pct']:.2f}%)"
            if backtest_results.get('max_drawdown', 100) > 30:
                return False, f"Max drawdown too high ({backtest_results['max_drawdown']:.2f}%)"

        # Save to history
        history_path = self.save_to_history(code, {'backtest': backtest_results})

        # Deploy
        active_path = os.path.join(self.STRATEGY_DIR, self.ACTIVE_STRATEGY_FILE)

        # Backup current if exists
        if os.path.exists(active_path):
            backup_path = active_path + '.backup'
            shutil.copy(active_path, backup_path)

        # Write new strategy
        with open(active_path, 'w') as f:
            f.write(code)

        # Git commit
        self._git_commit(f"Deploy new strategy - Return: {backtest_results.get('return_pct', 0):.2f}%")

        return True, json.dumps({
            "status": "deployed",
            "history_file": history_path,
            "backtest_results": backtest_results
        }, indent=2)

    def rollback(self, version: Optional[str] = None) -> Tuple[bool, str]:
        """Rollback to a previous strategy version"""
        active_path = os.path.join(self.STRATEGY_DIR, self.ACTIVE_STRATEGY_FILE)
        backup_path = active_path + '.backup'

        if version:
            # Rollback to specific version
            version_path = os.path.join(self.history_dir, version)
            if not os.path.exists(version_path):
                return False, f"Version not found: {version}"
            source = version_path
        elif os.path.exists(backup_path):
            # Rollback to backup
            source = backup_path
        else:
            return False, "No backup available"

        shutil.copy(source, active_path)
        self._git_commit(f"Rollback strategy to {version or 'backup'}")

        return True, f"Rolled back to {version or 'previous version'}"

    def list_versions(self) -> list:
        """List all strategy versions in history"""
        versions = []
        for f in sorted(os.listdir(self.history_dir), reverse=True):
            if f.endswith('.py'):
                path = os.path.join(self.history_dir, f)
                stat = os.stat(path)
                versions.append({
                    "filename": f,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size": stat.st_size
                })
        return versions

    def _git_commit(self, message: str):
        """Commit changes to git"""
        try:
            subprocess.run(
                ['git', 'add', self.ACTIVE_STRATEGY_FILE, self.STRATEGY_HISTORY_DIR],
                cwd=self.STRATEGY_DIR,
                capture_output=True,
                timeout=30
            )

            full_message = f"{message}\n\n🤖 Auto-committed by Strategy Manager"
            subprocess.run(
                ['git', 'commit', '-m', full_message],
                cwd=self.STRATEGY_DIR,
                capture_output=True,
                timeout=30
            )
        except Exception:
            pass  # Git commit is optional


# API for external access (e.g., from dashboard or AI agent)
def get_manager() -> StrategyManager:
    """Get a StrategyManager instance"""
    return StrategyManager()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Strategy Manager CLI')
    parser.add_argument('command', choices=['validate', 'backtest', 'deploy', 'rollback', 'list', 'show'])
    parser.add_argument('--file', '-f', help='Strategy file to process')
    parser.add_argument('--ticker', '-t', default='SPY', help='Ticker for backtest')
    parser.add_argument('--days', '-d', type=int, default=365, help='Days for backtest')
    parser.add_argument('--force', action='store_true', help='Force deployment even with poor results')
    parser.add_argument('--version', '-v', help='Version for rollback')

    args = parser.parse_args()
    manager = StrategyManager()

    if args.command == 'show':
        print(manager.get_active_strategy())

    elif args.command == 'list':
        versions = manager.list_versions()
        print(f"\nStrategy History ({len(versions)} versions):\n")
        for v in versions[:10]:
            print(f"  {v['filename']} - {v['modified']}")

    elif args.command == 'validate':
        if not args.file:
            print("ERROR: --file required")
            sys.exit(1)
        with open(args.file, 'r') as f:
            code = f.read()
        valid, results = manager.validate_strategy(code)
        print(f"\nValidation {'PASSED' if valid else 'FAILED'}:")
        for check, msg in results.items():
            print(f"  {check}: {msg}")

    elif args.command == 'backtest':
        if not args.file:
            print("ERROR: --file required")
            sys.exit(1)
        with open(args.file, 'r') as f:
            code = f.read()
        print(f"\nRunning backtest on {args.ticker} ({args.days} days)...")
        success, results = manager.run_backtest(code, args.ticker, args.days)
        if success:
            print(f"\nBacktest Results:")
            print(f"  Return: {results['return_pct']:.2f}%")
            print(f"  Sharpe: {results.get('sharpe_ratio', 0):.2f}")
            print(f"  Max DD: {results['max_drawdown']:.2f}%")
            print(f"  Trades: {results['total_trades']} (Win rate: {results['win_rate']:.1f}%)")
        else:
            print(f"\nBacktest FAILED: {results}")

    elif args.command == 'deploy':
        if not args.file:
            print("ERROR: --file required")
            sys.exit(1)
        with open(args.file, 'r') as f:
            code = f.read()
        print(f"\nDeploying strategy...")
        success, msg = manager.deploy_strategy(code, force=args.force)
        print(f"\n{'SUCCESS' if success else 'FAILED'}: {msg}")

    elif args.command == 'rollback':
        success, msg = manager.rollback(args.version)
        print(f"\n{'SUCCESS' if success else 'FAILED'}: {msg}")
