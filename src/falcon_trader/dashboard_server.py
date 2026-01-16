import os
from flask import Flask, jsonify, send_file, request, redirect
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime
from falcon_core import get_db_manager, FalconConfig

# Optional YouTube strategy support
try:
    from falcon_trader.youtube_strategies import YouTubeStrategyDB, YouTubeStrategyExtractor
    YOUTUBE_AVAILABLE = True
except ImportError:
    YouTubeStrategyDB = None
    YouTubeStrategyExtractor = None
    YOUTUBE_AVAILABLE = False

try:
    falcon_config = FalconConfig()
except Exception:
    falcon_config = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import backtest results API for analytics
try:
    from falcon_core.backtesting.results_api import BacktestResultsStore, create_api_routes
    BACKTEST_RESULTS_AVAILABLE = True
except ImportError:
    BacktestResultsStore = None
    create_api_routes = None
    BACKTEST_RESULTS_AVAILABLE = False

# Import your paper trading bot
# Assuming the previous code is in a file called paper_trading_bot.py
# from paper_trading_bot import PaperTradingBot, MassiveRealTimeFeed

app = Flask(__name__)
CORS(app)  # Enable CORS for web dashboard

# Global bot instance
bot = None

# Initialize database manager (uses environment variables for config)
db = get_db_manager()

# Initialize backtest results store and register API routes
backtest_results_store = None
if BACKTEST_RESULTS_AVAILABLE:
    try:
        db_path = os.getenv('DB_PATH')  # Use environment variable if set
        backtest_results_store = BacktestResultsStore(db_path=db_path)
        create_api_routes(app, backtest_results_store)
        print(f"Backtest analytics API routes registered (db: {db_path or 'default'})")
    except Exception as e:
        print(f"Warning: Could not initialize backtest results store: {e}")

# Get database path from config for legacy compatibility
if falcon_config:
    DB_PATH = falcon_config.get('db_path', '/var/lib/falcon/paper_trading.db')
else:
    DB_PATH = "paper_trading.db"

# API Routes

@app.route('/health')
def health():
    """Health check endpoint for container orchestration"""
    return jsonify({"status": "healthy", "service": "falcon-trading"}), 200

@app.route('/api/account')
def get_account():
    """Get current account information"""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503

    account = bot.get_account()

    # Calculate positions value
    positions = bot.get_positions()
    current_prices = bot.get_current_prices()
    positions_value = 0.0
    for pos in positions:
        current_price = float(current_prices.get(pos['symbol'], pos.get('average_price', 0)))
        positions_value += current_price * float(pos['quantity'])

    return jsonify({
        "totalValue": account['total_value'],
        "cash": account['cash'],
        "positionsValue": positions_value,
        "initialBalance": 10000.0  # TODO: Store this in database
    })

@app.route('/api/positions')
def get_positions():
    """Get current positions with stop-loss data"""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503

    positions = bot.get_positions()
    current_prices = bot.get_current_prices()

    # Get stop-loss data from database
    positions_list = []
    for pos in positions:
        symbol = pos['symbol']
        current_price = float(current_prices.get(symbol, pos.get('average_price', 0)))

        # Get stop-loss for this position
        row = db.execute('SELECT stop_loss FROM positions WHERE symbol = %s', (symbol,), fetch='one')
        stop_loss = float(row['stop_loss']) if row and row.get('stop_loss') else None

        positions_list.append({
            "symbol": symbol,
            "quantity": int(pos['quantity']),
            "avgPrice": float(pos.get('average_price', 0)),
            "currentPrice": current_price,
            "stopLoss": stop_loss
        })

    return jsonify(positions_list)

@app.route('/api/order', methods=['POST'])
def place_order():
    """Place a buy or sell order"""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503

    data = request.json
    symbol = data.get('symbol', '').upper()
    side = data.get('side', '').lower()
    quantity = data.get('quantity', 0)
    order_type = data.get('order_type', 'market')
    price = data.get('price')

    # Validate inputs
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    if side not in ['buy', 'sell']:
        return jsonify({"error": "Side must be 'buy' or 'sell'"}), 400
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        return jsonify({"error": "Quantity must be a positive number"}), 400

    # Place the order
    result = bot.place_order(symbol, side, int(quantity), order_type, price)

    if result.get('status') == 'success':
        return jsonify(result), 200
    else:
        return jsonify(result), 400

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    rows = db.execute('''
        SELECT symbol, side, quantity, price, timestamp, pnl
        FROM orders
        ORDER BY timestamp DESC
        LIMIT 50
    ''', fetch='all')

    trades = []
    for row in rows or []:
        trades.append({
            "symbol": row['symbol'],
            "side": row['side'],
            "quantity": float(row['quantity']) if row['quantity'] else 0,
            "price": float(row['price']) if row['price'] else 0,
            "timestamp": str(row['timestamp']) if row['timestamp'] else None,
            "pnl": float(row['pnl']) if row['pnl'] else 0
        })

    return jsonify(trades)

@app.route('/api/trades/summary')
def get_trades_summary():
    """Get trading statistics"""
    # Calculate PnL for closed positions
    trades = db.execute('''
        SELECT
            symbol,
            side,
            quantity,
            price,
            timestamp
        FROM orders
        ORDER BY timestamp
    ''', fetch='all') or []

    positions_tracker = {}
    closed_trades = []

    for trade in trades:
        symbol, side, quantity, price, timestamp = trade
        side = side.upper()  # Normalize to uppercase

        if side == 'BUY':
            if symbol not in positions_tracker:
                positions_tracker[symbol] = []
            positions_tracker[symbol].append({
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp
            })
        elif side == 'SELL' and symbol in positions_tracker:
            remaining_sell = quantity
            sell_proceeds = 0
            buy_cost = 0

            while remaining_sell > 0 and positions_tracker[symbol]:
                buy_trade = positions_tracker[symbol][0]

                if buy_trade['quantity'] <= remaining_sell:
                    # Close this buy position completely
                    qty = buy_trade['quantity']
                    buy_cost += qty * buy_trade['price']
                    sell_proceeds += qty * price
                    remaining_sell -= qty
                    positions_tracker[symbol].pop(0)
                else:
                    # Partial close
                    buy_cost += remaining_sell * buy_trade['price']
                    sell_proceeds += remaining_sell * price
                    buy_trade['quantity'] -= remaining_sell
                    remaining_sell = 0

            if buy_cost > 0:
                pnl = sell_proceeds - buy_cost
                closed_trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'timestamp': timestamp
                })

    # Calculate statistics
    total_trades = len(closed_trades)
    winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    pnls = [t['pnl'] for t in closed_trades]
    best_trade = max(pnls) if pnls else 0
    worst_trade = min(pnls) if pnls else 0
    total_pnl = sum(pnls)

    return jsonify({
        "totalTrades": total_trades,
        "winningTrades": winning_trades,
        "losingTrades": losing_trades,
        "winRate": win_rate,
        "bestTrade": best_trade,
        "worstTrade": worst_trade,
        "totalPnL": total_pnl
    })

@app.route('/api/performance')
def get_performance():
    """Get performance history"""
    rows = db.execute('''
        SELECT timestamp, total_value, cash, positions_value
        FROM performance
        ORDER BY timestamp DESC
        LIMIT 200
    ''', fetch='all') or []

    performance = []
    for row in rows:
        performance.append({
            "timestamp": str(row['timestamp']) if row['timestamp'] else None,
            "totalValue": float(row['total_value']) if row['total_value'] else 0,
            "cash": float(row['cash']) if row['cash'] else 0,
            "positionsValue": float(row['positions_value']) if row['positions_value'] else 0
        })

    return jsonify(list(reversed(performance)))

@app.route('/api/signals')
def get_signals():
    """Get current trading signals"""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503
    
    signals = []
    for symbol in bot.symbols:
        df = bot.data_feed.get_aggregates(symbol, multiplier=5, timespan="minute", limit=100)
        if not df.empty:
            analysis = bot.strategies[symbol].analyze(df)
            signals.append({
                "symbol": symbol,
                "signal": analysis['signal'],
                "reason": analysis['reason'],
                "confidence": analysis['confidence'],
                "indicators": analysis['indicators']
            })
    
    return jsonify(signals)

@app.route('/api/bot/status')
def get_bot_status():
    """Get bot running status"""
    if not bot:
        return jsonify({"running": False, "message": "Bot not initialized"})
    
    return jsonify({
        "running": bot.running,
        "symbols": bot.symbols,
        "updateInterval": bot.update_interval,
        "startTime": getattr(bot, 'start_time', None)
    })

@app.route('/api/bot/start')
def start_bot():
    """Start the trading bot"""
    if not bot:
        return jsonify({"error": "Bot not configured"}), 400
    
    if bot.running:
        return jsonify({"message": "Bot already running"})
    
    bot.start()
    return jsonify({"message": "Bot started successfully"})

@app.route('/api/bot/stop')
def stop_bot():
    """Stop the trading bot"""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503
    
    if not bot.running:
        return jsonify({"message": "Bot not running"})
    
    bot.stop()
    return jsonify({"message": "Bot stopped successfully"})

@app.route('/api/analysis')
def get_ai_analysis():
    """Get AI analysis of trading performance"""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503

    analysis = bot.get_ai_analysis()
    return jsonify({"analysis": analysis})


@app.route('/api/recommendations')
def get_recommendations():
    """Get AI stock recommendations from the screener"""
    try:
        screened_file = os.path.join(os.path.dirname(__file__), 'screened_stocks.json')
        if not os.path.exists(screened_file):
            return jsonify({
                "status": "no_data",
                "message": "No screening results available yet",
                "recommendations": []
            })

        with open(screened_file, 'r') as f:
            results = json.load(f)

        if not results:
            return jsonify({
                "status": "no_data",
                "message": "Screening results file is empty",
                "recommendations": []
            })

        # Return the most recent screening result
        latest = results[-1]
        return jsonify({
            "status": "success",
            "timestamp": latest.get('timestamp'),
            "screen_type": latest.get('screen_type'),
            "total_stocks_screened": latest.get('total_stocks', 0),
            "recommendations": latest.get('recommendations', [])
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in screened_stocks.json"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommendations/history')
def get_recommendations_history():
    """Get historical screening results"""
    try:
        screened_file = os.path.join(os.path.dirname(__file__), 'screened_stocks.json')
        if not os.path.exists(screened_file):
            return jsonify({"history": []})

        with open(screened_file, 'r') as f:
            results = json.load(f)

        # Return summary of each screening run
        history = []
        for result in results:
            history.append({
                "timestamp": result.get('timestamp'),
                "screen_type": result.get('screen_type'),
                "total_stocks": result.get('total_stocks', 0),
                "recommendation_count": len(result.get('recommendations', []))
            })

        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def serve_index():
    """Serve the landing page"""
    return send_file('www/index.html')


@app.route('/dashboard')
@app.route('/orchestrator')
@app.route('/orchestrator.html')
def serve_dashboard():
    """Serve the main trading dashboard with account, positions, and trades"""
    return send_file('www/orchestrator.html')


@app.route('/strategies.html')
def serve_strategies_page():
    """Serve the strategies list page"""
    return send_file('www/strategies.html')


@app.route('/strategy-view.html')
def serve_strategy_view_page():
    """Serve the strategy detail view page"""
    return send_file('www/strategy-view.html')


@app.route('/strategies/<int:strategy_id>.html')
def serve_strategy_view(strategy_id):
    """Serve the strategy detail view page with proper ID routing"""
    # Redirect to the page with query parameter
    return redirect(f'/strategy-view.html?id={strategy_id}')

# WebSocket support for real-time updates (optional, using Server-Sent Events)
@app.route('/api/stream')
def stream():
    """Server-Sent Events endpoint for real-time updates"""
    def event_stream():
        while True:
            if bot and bot.running:
                # Get current data
                current_prices = bot.get_current_prices()
                account_value = bot.account.get_account_value(current_prices)
                
                data = {
                    "type": "update",
                    "timestamp": datetime.now().isoformat(),
                    "account": {
                        "totalValue": account_value['total_value'],
                        "cash": account_value['cash'],
                        "positionsValue": account_value['positions_value']
                    }
                }
                
                yield f"data: {json.dumps(data)}\n\n"
            
            time.sleep(5)  # Update every 5 seconds
    
    return app.response_class(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

# Strategy Management API
# These endpoints allow AI agents to modify trading strategies

@app.route('/api/strategy')
def get_strategy():
    """Get the current active strategy code"""
    try:
        from falcon_trader.strategy_manager import StrategyManager
        manager = StrategyManager()
        code = manager.get_active_strategy()
        return jsonify({
            "status": "success",
            "strategy": code
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategy/validate', methods=['POST'])
def validate_strategy():
    """Validate a strategy without deploying it"""
    try:
        from flask import request
        from falcon_trader.strategy_manager import StrategyManager

        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({"error": "Missing 'code' in request body"}), 400

        manager = StrategyManager()
        valid, results = manager.validate_strategy(data['code'])

        return jsonify({
            "valid": valid,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategy/backtest', methods=['POST'])
def backtest_strategy():
    """Run a backtest on a strategy"""
    try:
        from flask import request
        from falcon_trader.strategy_manager import StrategyManager

        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({"error": "Missing 'code' in request body"}), 400

        ticker = data.get('ticker', 'SPY')
        days = data.get('days', 365)

        manager = StrategyManager()
        success, results = manager.run_backtest(data['code'], ticker, days)

        return jsonify({
            "success": success,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategy/deploy', methods=['POST'])
def deploy_strategy():
    """Deploy a new strategy (validates and backtests first)"""
    try:
        from flask import request
        from falcon_trader.strategy_manager import StrategyManager

        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({"error": "Missing 'code' in request body"}), 400

        force = data.get('force', False)

        manager = StrategyManager()
        success, message = manager.deploy_strategy(data['code'], force=force)

        if success:
            return jsonify({
                "status": "deployed",
                "details": json.loads(message)
            })
        else:
            return jsonify({
                "status": "failed",
                "error": message
            }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategy/rollback', methods=['POST'])
def rollback_strategy():
    """Rollback to a previous strategy version"""
    try:
        from flask import request
        from falcon_trader.strategy_manager import StrategyManager

        data = request.get_json() or {}
        version = data.get('version')

        manager = StrategyManager()
        success, message = manager.rollback(version)

        return jsonify({
            "success": success,
            "message": message
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategy/versions')
def list_strategy_versions():
    """List all strategy versions in history"""
    try:
        from falcon_trader.strategy_manager import StrategyManager
        manager = StrategyManager()
        versions = manager.list_versions()
        return jsonify({
            "versions": versions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# YouTube Strategies Endpoints (optional)
strategy_db = YouTubeStrategyDB() if YOUTUBE_AVAILABLE else None
strategy_extractor = None  # Will be initialized with Claude API key

@app.route('/api/youtube-strategies', methods=['GET'])
def get_youtube_strategies():
    """Get all YouTube trading strategies"""
    if not YOUTUBE_AVAILABLE:
        return jsonify({"error": "YouTube features not available. Install with: pip install falcon-trader[youtube]"}), 503
    try:
        strategies = strategy_db.get_all_strategies()
        return jsonify({"status": "success", "strategies": strategies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/youtube-strategies/<int:strategy_id>', methods=['GET'])
def get_youtube_strategy(strategy_id):
    """Get a specific strategy by ID"""
    if not YOUTUBE_AVAILABLE:
        return jsonify({"error": "YouTube features not available. Install with: pip install falcon-trader[youtube]"}), 503
    try:
        strategy = strategy_db.get_strategy_by_id(strategy_id)
        if strategy:
            return jsonify({"status": "success", "strategy": strategy})
        else:
            return jsonify({"error": "Strategy not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/youtube-strategies/submit', methods=['POST'])
def submit_youtube_url():
    """Submit a YouTube URL for strategy extraction"""
    if not YOUTUBE_AVAILABLE:
        return jsonify({"error": "YouTube features not available. Install with: pip install falcon-trader[youtube]"}), 503
    try:
        data = request.json
        youtube_url = data.get('youtube_url')

        if not youtube_url:
            return jsonify({"error": "youtube_url is required"}), 400

        if not strategy_extractor:
            return jsonify({"error": "Strategy extractor not initialized (Claude API key required)"}), 503

        # Process the YouTube URL
        strategy_data = strategy_extractor.process_youtube_url(youtube_url)

        if "error" in strategy_data:
            return jsonify(strategy_data), 400

        # Save to database
        strategy_id = strategy_db.add_strategy(strategy_data)

        return jsonify({
            "status": "success",
            "message": "Strategy extracted and saved",
            "strategy_id": strategy_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Active Strategy Management Endpoints
# These endpoints manage the automated strategy execution system

@app.route('/api/strategies/youtube/<int:youtube_strategy_id>/activate', methods=['POST'])
def activate_youtube_strategy(youtube_strategy_id):
    """
    Activate a YouTube strategy for live trading

    Body: {
        "symbols": ["SPY", "QQQ"],
        "allocation_pct": 20.0
    }
    """
    if not YOUTUBE_AVAILABLE:
        return jsonify({"error": "YouTube features not available. Install with: pip install falcon-trader[youtube]"}), 503
    try:
        from strategy_parser import StrategyCodeGenerator
        from falcon_trader.strategy_manager import StrategyManager

        data = request.json or {}
        symbols = data.get('symbols', ['SPY', 'QQQ'])
        allocation_pct = data.get('allocation_pct', 20.0)

        # Get YouTube strategy
        youtube_strategy = strategy_db.get_strategy_by_id(youtube_strategy_id)
        if not youtube_strategy:
            return jsonify({"error": "YouTube strategy not found"}), 404

        # Get Claude API key
        claude_key = os.getenv('CLAUDE_API_KEY')
        if not claude_key:
            return jsonify({"error": "CLAUDE_API_KEY not set"}), 503

        # Generate code from YouTube strategy
        generator = StrategyCodeGenerator(claude_key)
        success, code, error = generator.generate_from_youtube_strategy(youtube_strategy)

        if not success:
            return jsonify({
                "error": "Failed to generate code",
                "details": error
            }), 400

        # Validate
        manager = StrategyManager()
        valid, validation_results = manager.validate_strategy(code)
        if not valid:
            return jsonify({
                "error": "Generated code failed validation",
                "details": validation_results.get('error', 'Unknown error')
            }), 400

        # Backtest (reject if return < -5%)
        _, backtest_results = manager.run_backtest(code, ticker=symbols[0], days=365)
        if backtest_results.get('return_pct', 0) < -5:
            return jsonify({
                "error": "Strategy failed backtest (return < -5%)",
                "backtest_results": backtest_results
            }), 400

        # Save to active_strategies
        strategy_id = db.execute('''
            INSERT INTO active_strategies
            (youtube_strategy_id, strategy_name, strategy_code, parameters,
             symbols, status, allocation_pct, performance_weight, created_at, activated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            youtube_strategy_id,
            youtube_strategy.get('title', 'Unknown Strategy'),
            code,
            json.dumps({}),  # Parameters extracted from code
            json.dumps(symbols),
            'active',
            allocation_pct,
            1.0,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

        return jsonify({
            "status": "success",
            "message": "Strategy activated for live trading",
            "strategy_id": strategy_id,
            "backtest_results": backtest_results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/active', methods=['GET'])
def get_active_strategies():
    """List all active strategies with current performance"""
    try:
        from strategy_analytics import StrategyAnalytics

        analytics = StrategyAnalytics(DB_PATH)
        leaderboard = analytics.get_all_strategies_leaderboard()

        # Also get all strategies (not just those with performance data)
        rows = db.execute('''
            SELECT id, strategy_name, status, allocation_pct,
                   performance_weight, created_at, activated_at
            FROM active_strategies
            WHERE status IN ('active', 'paused')
            ORDER BY activated_at DESC
        ''', fetch='all') or []

        strategies = []
        for row in rows:
            strategy_id = row['id']

            # Find performance data from leaderboard
            perf_data = next((s for s in leaderboard if s['strategy_id'] == strategy_id), None)

            strategies.append({
                "strategy_id": strategy_id,
                "strategy_name": row['strategy_name'],
                "status": row['status'],
                "allocation_pct": float(row['allocation_pct']) if row['allocation_pct'] else 0,
                "performance_weight": float(row['performance_weight']) if row['performance_weight'] else 0,
                "created_at": str(row['created_at']) if row['created_at'] else None,
                "activated_at": str(row['activated_at']) if row['activated_at'] else None,
                "performance": perf_data['performance'] if perf_data else {}
            })

        return jsonify({
            "status": "success",
            "strategies": strategies
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/<int:strategy_id>/performance', methods=['GET'])
def get_strategy_performance(strategy_id):
    """Get detailed metrics for a strategy"""
    try:
        from strategy_analytics import StrategyAnalytics

        analytics = StrategyAnalytics(DB_PATH)
        summary = analytics.get_strategy_summary(strategy_id)

        if not summary:
            return jsonify({"error": "Strategy not found"}), 404

        return jsonify({
            "status": "success",
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/<int:strategy_id>/pause', methods=['POST'])
def pause_strategy(strategy_id):
    """Pause a strategy (stop generating signals)"""
    try:
        rowcount = db.execute('''
            UPDATE active_strategies
            SET status = 'paused', deactivated_at = %s
            WHERE id = %s AND status = 'active'
        ''', (datetime.now().isoformat(), strategy_id))

        if rowcount == 0:
            return jsonify({"error": "Strategy not found or already paused"}), 404

        return jsonify({
            "status": "success",
            "message": f"Strategy {strategy_id} paused"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/<int:strategy_id>/resume', methods=['POST'])
def resume_strategy(strategy_id):
    """Resume a paused strategy"""
    try:
        rowcount = db.execute('''
            UPDATE active_strategies
            SET status = 'active', activated_at = %s
            WHERE id = %s AND status = 'paused'
        ''', (datetime.now().isoformat(), strategy_id))

        if rowcount == 0:
            return jsonify({"error": "Strategy not found or not paused"}), 404

        return jsonify({
            "status": "success",
            "message": f"Strategy {strategy_id} resumed"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/<int:strategy_id>/signals', methods=['GET'])
def get_strategy_signals(strategy_id):
    """Get recent signals (last 50) for debugging"""
    try:
        rows = db.execute('''
            SELECT symbol, signal_type, signal_reason, confidence,
                   market_price, action_taken, timestamp
            FROM strategy_signals
            WHERE strategy_id = %s
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (strategy_id,), fetch='all') or []

        signals = []
        for row in rows:
            signals.append({
                "symbol": row['symbol'],
                "signal_type": row['signal_type'],
                "signal_reason": row['signal_reason'],
                "confidence": float(row['confidence']) if row['confidence'] else None,
                "market_price": float(row['market_price']) if row['market_price'] else None,
                "action_taken": row['action_taken'],
                "timestamp": str(row['timestamp']) if row['timestamp'] else None
            })

        return jsonify({
            "status": "success",
            "signals": signals
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/leaderboard', methods=['GET'])
def get_strategy_leaderboard():
    """Rank strategies by win rate and ROI"""
    try:
        from strategy_analytics import StrategyAnalytics

        analytics = StrategyAnalytics(DB_PATH)
        leaderboard = analytics.get_all_strategies_leaderboard()

        return jsonify({
            "status": "success",
            "leaderboard": leaderboard
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/positions/set-stop-loss', methods=['POST'])
def set_stop_loss():
    """Set or update stop-loss for a position"""
    try:
        data = request.json
        symbol = data.get('symbol', '').upper()
        stop_loss = data.get('stop_loss')

        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400

        if stop_loss is not None:
            stop_loss = float(stop_loss)
            if stop_loss < 0:
                return jsonify({"error": "Stop-loss must be a positive number or null"}), 400

        # Update stop-loss in database
        # Check if position exists
        row = db.execute('SELECT symbol FROM positions WHERE symbol = %s', (symbol,), fetch='one')
        if not row:
            return jsonify({"error": f"No position found for {symbol}"}), 404

        # Update stop-loss
        db.execute('''
            UPDATE positions
            SET stop_loss = %s
            WHERE symbol = %s
        ''', (stop_loss, symbol))

        return jsonify({
            "status": "success",
            "message": f"Stop-loss {'set' if stop_loss else 'removed'} for {symbol}",
            "symbol": symbol,
            "stop_loss": stop_loss
        })

    except ValueError:
        return jsonify({"error": "Invalid stop-loss value"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies/aggregate', methods=['GET'])
def get_aggregate_statistics():
    """Get aggregate statistics across all active strategies"""
    try:
        from strategy_analytics import StrategyAnalytics

        analytics = StrategyAnalytics(DB_PATH)
        stats = analytics.get_aggregate_statistics()

        return jsonify({
            "status": "success",
            "statistics": stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# CHART DATA ENDPOINTS
# ============================================

@app.route('/api/charts/equity', methods=['GET'])
def get_equity_chart():
    """Get equity curve data for charting"""
    try:
        # Get optional parameters
        limit = request.args.get('limit', 500, type=int)
        interval = request.args.get('interval', 'minute')  # minute, hour, day

        # Use database-agnostic queries (works with both SQLite and PostgreSQL)
        if interval == 'day':
            # Daily aggregation - cast timestamp to date
            rows = db.execute('''
                SELECT CAST(timestamp AS DATE) as day,
                       MAX(total_value) as total_value,
                       MAX(cash) as cash,
                       MAX(positions_value) as positions_value
                FROM performance
                GROUP BY CAST(timestamp AS DATE)
                ORDER BY day DESC
                LIMIT %s
            ''', (limit,), fetch='all') or []
        elif interval == 'hour':
            # Hourly - just get raw data and aggregate in Python for compatibility
            rows = db.execute('''
                SELECT timestamp, total_value, cash, positions_value
                FROM performance
                ORDER BY timestamp DESC
                LIMIT %s
            ''', (limit * 60,), fetch='all') or []  # Get more data for aggregation
        else:
            # Raw minute data
            rows = db.execute('''
                SELECT timestamp, total_value, cash, positions_value
                FROM performance
                ORDER BY timestamp DESC
                LIMIT %s
            ''', (limit,), fetch='all') or []

        # Reverse to get chronological order
        data = []
        for row in reversed(rows):
            # Handle both day aggregation (with 'day' key) and raw data (with 'timestamp' key)
            ts = row.get('day') or row.get('timestamp')
            data.append({
                "timestamp": str(ts) if ts else None,
                "totalValue": float(row.get('total_value', 0) or 0),
                "cash": float(row.get('cash', 0) or 0),
                "positionsValue": float(row.get('positions_value', 0) or 0)
            })

        return jsonify({
            "status": "success",
            "interval": interval,
            "count": len(data),
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/charts/pnl', methods=['GET'])
def get_pnl_chart():
    """Get P&L data by day/trade for charting"""
    try:
        # Get daily P&L from orders (database-agnostic)
        rows = db.execute('''
            SELECT CAST(timestamp AS DATE) as day,
                   SUM(CASE WHEN side='SELL' THEN pnl ELSE 0 END) as daily_pnl,
                   COUNT(*) as trade_count,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
            FROM orders
            WHERE side = 'SELL'
            GROUP BY CAST(timestamp AS DATE)
            ORDER BY day DESC
            LIMIT 30
        ''', fetch='all') or []

        daily_data = []
        cumulative_pnl = 0
        rows_list = list(reversed(rows))

        for row in rows_list:
            daily_pnl = float(row.get('daily_pnl', 0) or 0)
            cumulative_pnl += daily_pnl
            daily_data.append({
                "date": str(row.get('day')) if row.get('day') else None,
                "dailyPnL": daily_pnl,
                "cumulativePnL": cumulative_pnl,
                "tradeCount": int(row.get('trade_count', 0) or 0),
                "wins": int(row.get('wins', 0) or 0),
                "losses": int(row.get('losses', 0) or 0)
            })

        return jsonify({
            "status": "success",
            "data": daily_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/charts/strategy-comparison', methods=['GET'])
def get_strategy_comparison():
    """Get strategy performance comparison data"""
    try:
        # Get strategy metrics (table may not exist in PostgreSQL yet)
        rows = db.execute('''
            SELECT strategy,
                   total_trades,
                   winning_trades,
                   losing_trades,
                   win_rate,
                   total_return,
                   max_drawdown,
                   sharpe_ratio,
                   avg_profit
            FROM strategy_metrics
            ORDER BY total_return DESC
        ''', fetch='all') or []

        strategies = []
        for row in rows:
            strategies.append({
                "strategy": row.get('strategy'),
                "totalTrades": row.get('total_trades'),
                "winningTrades": row.get('winning_trades'),
                "losingTrades": row.get('losing_trades'),
                "winRate": row.get('win_rate'),
                "totalReturn": row.get('total_return'),
                "maxDrawdown": row.get('max_drawdown'),
                "sharpeRatio": row.get('sharpe_ratio'),
                "avgProfit": row.get('avg_profit')
            })

        return jsonify({
            "status": "success",
            "strategies": strategies
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/charts/trade-distribution', methods=['GET'])
def get_trade_distribution():
    """Get trade size and P&L distribution data"""
    try:
        # Get trade P&L distribution
        rows = db.execute('''
            SELECT pnl, symbol, strategy, timestamp
            FROM orders
            WHERE side = 'SELL' AND pnl != 0
            ORDER BY timestamp DESC
            LIMIT 100
        ''', fetch='all') or []

        trades = []
        for row in rows:
            trades.append({
                "pnl": float(row.get('pnl', 0) or 0),
                "symbol": row.get('symbol'),
                "strategy": row.get('strategy'),
                "timestamp": str(row.get('timestamp')) if row.get('timestamp') else None
            })

        # Get summary stats
        stats = db.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM orders
            WHERE side = 'SELL'
        ''', fetch='one')

        return jsonify({
            "status": "success",
            "trades": trades,
            "summary": {
                "total": int(stats.get('total', 0) or 0) if stats else 0,
                "profitable": int(stats.get('profitable', 0) or 0) if stats else 0,
                "avgPnL": float(stats.get('avg_pnl', 0) or 0) if stats else 0,
                "bestTrade": float(stats.get('best_trade', 0) or 0) if stats else 0,
                "worstTrade": float(stats.get('worst_trade', 0) or 0) if stats else 0
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analytics')
@app.route('/analytics.html')
def serve_analytics():
    """Serve the analytics dashboard"""
    return send_file('www/analytics.html')


@app.route('/backtest-analytics')
@app.route('/backtest-analytics.html')
def serve_backtest_analytics():
    """Serve the backtest analytics dashboard"""
    return send_file('www/backtest-analytics.html')


# ============================================
# SCREENER PROFILE ENDPOINTS
# ============================================

@app.route('/api/screener/profiles', methods=['GET'])
def list_screener_profiles():
    """List all screener profiles"""
    try:
        from screener.profile_manager import ProfileManager
        manager = ProfileManager(db)

        enabled_only = request.args.get('enabled', 'false').lower() == 'true'
        theme = request.args.get('theme')

        profiles = manager.list_profiles(enabled_only=enabled_only, theme=theme)

        return jsonify({
            "status": "success",
            "profiles": [p.to_dict() for p in profiles]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles', methods=['POST'])
def create_screener_profile():
    """Create a new screener profile"""
    try:
        from screener.profile_manager import ProfileManager, ScreenerProfile
        manager = ProfileManager(db)

        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        if not data.get('name') or not data.get('theme'):
            return jsonify({"error": "name and theme are required"}), 400

        profile = ScreenerProfile.from_dict(data)
        profile_id = manager.create_profile(profile)

        return jsonify({
            "status": "success",
            "message": "Profile created",
            "profile_id": profile_id
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/<int:profile_id>', methods=['GET'])
def get_screener_profile(profile_id):
    """Get a specific screener profile"""
    try:
        from screener.profile_manager import ProfileManager
        manager = ProfileManager(db)

        profile = manager.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404

        return jsonify({
            "status": "success",
            "profile": profile.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/<int:profile_id>', methods=['PUT'])
def update_screener_profile(profile_id):
    """Update a screener profile"""
    try:
        from screener.profile_manager import ProfileManager, ScreenerProfile
        manager = ProfileManager(db)

        existing = manager.get_profile(profile_id)
        if not existing:
            return jsonify({"error": "Profile not found"}), 404

        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        # Merge with existing
        data['id'] = profile_id
        data['created_at'] = existing.created_at
        profile = ScreenerProfile.from_dict(data)

        manager.update_profile(profile)

        return jsonify({
            "status": "success",
            "message": "Profile updated"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/<int:profile_id>', methods=['DELETE'])
def delete_screener_profile(profile_id):
    """Delete a screener profile"""
    try:
        from screener.profile_manager import ProfileManager
        manager = ProfileManager(db)

        if not manager.get_profile(profile_id):
            return jsonify({"error": "Profile not found"}), 404

        manager.delete_profile(profile_id)

        return jsonify({
            "status": "success",
            "message": "Profile deleted"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/<int:profile_id>/run', methods=['POST'])
def run_screener_profile(profile_id):
    """Manually trigger a profile screening run"""
    try:
        from screener.profile_manager import ProfileManager
        from screener.multi_screener import MultiScreener
        manager = ProfileManager(db)

        profile = manager.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404

        data = request.get_json() or {}
        run_type = data.get('run_type', 'morning')
        use_ai = data.get('use_ai', True)

        screener = MultiScreener(manager)
        result = screener.run_profile(profile, run_type, use_ai)

        return jsonify({
            "status": "success",
            "result": result.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/<int:profile_id>/performance', methods=['GET'])
def get_screener_profile_performance(profile_id):
    """Get performance metrics for a profile"""
    try:
        from screener.profile_manager import ProfileManager
        from screener.feedback_loop import WeightFeedbackLoop
        manager = ProfileManager(db)

        profile = manager.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404

        days = request.args.get('days', 30, type=int)

        # Get aggregate performance
        aggregate = manager.get_aggregate_performance(profile_id, days)

        # Get detailed performance history
        feedback = WeightFeedbackLoop(manager)
        try:
            metrics = feedback.calculate_profile_performance(profile_id, days)
            detailed = {
                "win_rate": metrics.win_rate,
                "avg_return": metrics.avg_return,
                "best_category": metrics.best_category,
                "worst_category": metrics.worst_category,
                "suggested_adjustments": metrics.suggested_adjustments,
            }
        except Exception:
            detailed = {}

        return jsonify({
            "status": "success",
            "profile_name": profile.name,
            "aggregate": aggregate,
            "detailed": detailed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/export', methods=['GET'])
def export_screener_profiles():
    """Export all profiles as YAML"""
    try:
        from screener.profile_manager import ProfileManager
        from screener.yaml_serializer import ProfileYAMLSerializer
        manager = ProfileManager(db)

        enabled_only = request.args.get('enabled', 'false').lower() == 'true'
        profiles = manager.list_profiles(enabled_only=enabled_only)

        yaml_content = ProfileYAMLSerializer.export_profiles(profiles)

        return app.response_class(
            yaml_content,
            mimetype='text/yaml',
            headers={'Content-Disposition': 'attachment; filename=screener_profiles.yaml'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/import', methods=['POST'])
def import_screener_profiles():
    """Import profiles from YAML"""
    try:
        from screener.profile_manager import ProfileManager
        from screener.yaml_serializer import ProfileYAMLSerializer
        manager = ProfileManager(db)

        # Get YAML content from request body
        yaml_content = request.get_data(as_text=True)
        if not yaml_content:
            return jsonify({"error": "Missing YAML content in request body"}), 400

        update_existing = request.args.get('update', 'true').lower() == 'true'

        profiles = ProfileYAMLSerializer.import_profiles(yaml_content)
        stats = ProfileYAMLSerializer.sync_to_database(profiles, manager, update_existing)

        return jsonify({
            "status": "success",
            "message": f"Imported {len(profiles)} profiles",
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/<int:profile_id>/weights/adjust', methods=['POST'])
def adjust_profile_weights(profile_id):
    """Apply weight adjustments to a profile"""
    try:
        from screener.profile_manager import ProfileManager
        from screener.feedback_loop import WeightFeedbackLoop
        manager = ProfileManager(db)

        profile = manager.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404

        data = request.get_json() or {}
        adjustments = data.get('adjustments', {})
        auto = data.get('auto', False)

        if not adjustments:
            # Auto-calculate adjustments
            feedback = WeightFeedbackLoop(manager)
            metrics = feedback.calculate_profile_performance(profile_id, days=30)
            adjustments = metrics.suggested_adjustments

        if not adjustments:
            return jsonify({
                "status": "success",
                "message": "No adjustments needed"
            })

        feedback = WeightFeedbackLoop(manager)
        applied = feedback.apply_weight_adjustments(profile_id, adjustments, auto=auto)

        return jsonify({
            "status": "success",
            "applied": applied,
            "adjustments": adjustments
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/screener/profiles/init', methods=['POST'])
def init_default_profiles():
    """Initialize default screener profiles"""
    try:
        from screener.profile_manager import ProfileManager
        from screener.profile_templates import initialize_default_profiles
        manager = ProfileManager(db)

        data = request.get_json() or {}
        force = data.get('force', False)

        created_ids = initialize_default_profiles(manager, force=force)

        return jsonify({
            "status": "success",
            "message": f"Initialized {len(created_ids)} profiles",
            "profile_ids": created_ids
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def initialize_bot(massive_api_key, claude_api_key=None, symbols=None, initial_balance=10000.0):
    """Initialize the trading bot"""
    global bot
    
    if symbols is None:
        symbols = ["SPY", "QQQ"]
    
    # Import here to avoid circular imports
    from falcon_trader.paper_trading_bot import PaperTradingBot
    
    bot = PaperTradingBot(
        symbols=symbols,
        massive_api_key=massive_api_key,
        claude_api_key=claude_api_key,
        initial_balance=initial_balance,
        update_interval=60
    )

    # Start the bot's background market data update thread
    bot.start()

    print(f"Bot initialized with symbols: {symbols}")
    return bot

def main():
    """CLI entry point for falcon-dashboard"""
    import sys

    # Get API keys from environment variables (preferred) or command line
    MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '')
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')

    # Command line overrides (for backwards compatibility)
    if len(sys.argv) > 1:
        MASSIVE_API_KEY = sys.argv[1]
    if len(sys.argv) > 2:
        CLAUDE_API_KEY = sys.argv[2]

    # Initialize YouTube strategy extractor if Claude API key is available
    if YOUTUBE_AVAILABLE and CLAUDE_API_KEY:
        print("Initializing YouTube strategy extractor with Claude API...")
        global strategy_extractor
        strategy_extractor = YouTubeStrategyExtractor(CLAUDE_API_KEY)
        print("[OK] Strategy extractor ready")
    elif not YOUTUBE_AVAILABLE:
        print("NOTE: YouTube strategy extraction not available (install with: pip install falcon-trader[youtube])")
    else:
        print("WARNING: CLAUDE_API_KEY not set - YouTube strategy extraction disabled")

    # Try to initialize the trading bot (optional - dashboard works without it)
    try:
        if MASSIVE_API_KEY and MASSIVE_API_KEY != 'YOUR_MASSIVE_API_KEY':
            print("Initializing trading bot...")
            initialize_bot(
                massive_api_key=MASSIVE_API_KEY,
                claude_api_key=CLAUDE_API_KEY,
                symbols=["SPY", "QQQ", "AAPL"],
                initial_balance=10000.0
            )
        else:
            print("WARNING: MASSIVE_API_KEY not configured - running dashboard without trading bot")
            print("Strategy management endpoints will still work")
    except ImportError as e:
        print(f"WARNING: Could not initialize trading bot: {e}")
        print("Dashboard running in standalone mode - strategy management available")

    print("\n" + "="*80)
    print("Dashboard Server Starting")
    print("="*80)
    print(f"Dashboard URL: http://localhost:5000")
    print(f"API Endpoints:")
    print(f"  - GET  /api/account            - Account information")
    print(f"  - GET  /api/positions          - Current positions")
    print(f"  - GET  /api/trades             - Recent trades")
    print(f"  - GET  /api/performance        - Performance history")
    print(f"  - GET  /api/signals            - Current signals")
    print(f"  - GET  /api/bot/status         - Bot status")
    print(f"  - GET  /api/bot/start          - Start bot")
    print(f"  - GET  /api/bot/stop           - Stop bot")
    print(f"  - GET  /api/analysis           - AI analysis")
    print(f"  - GET  /api/recommendations    - AI stock picks")
    print(f"  - GET  /api/recommendations/history - Screening history")
    print(f"  - GET  /api/youtube-strategies - List all YouTube strategies")
    print(f"  - GET  /api/youtube-strategies/<id> - Get specific strategy")
    print(f"  - POST /api/youtube-strategies/submit - Submit YouTube URL")
    print("="*80 + "\n")

    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
