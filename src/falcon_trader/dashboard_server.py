import os
from flask import Flask, jsonify, send_file, request, redirect
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
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

# DAS Trader CMD-API execution backend (SIM only, dry-run default)
try:
    from falcon_trader.das_execution import DASExecutionClient, health_check as das_health_check
    DAS_AVAILABLE = True
except ImportError:
    DASExecutionClient = None
    das_health_check = None
    DAS_AVAILABLE = False

# Runtime-mutable config (editable from the website via /api/config — no restart needed).
# "execution_backend": "das" routes orders through DAStrader (SIM); else the in-memory paper bot.
RUNTIME_CONFIG = {
    "execution_backend": os.getenv("FALCON_EXECUTION", "paper").lower(),
}

# Import your paper trading bot
# Assuming the previous code is in a file called paper_trading_bot.py
# from paper_trading_bot import PaperTradingBot, MassiveRealTimeFeed

app = Flask(__name__)
CORS(app)  # Enable CORS for web dashboard


@app.errorhandler(Exception)
def handle_uncaught_exception(e):
    """Return JSON (never an HTML error page) for any error raised on an /api/*
    route, so JSON clients never choke on a '<!doctype html>' body. Non-API
    routes keep Flask's default HTML error behavior."""
    if request.path.startswith('/api/'):
        code = e.code if isinstance(e, HTTPException) else 500
        return jsonify({"status": "error", "error": str(e)}), code
    if isinstance(e, HTTPException):
        return e
    raise e

# Global bot instance
bot = None

# Initialize database manager (uses environment variables for config)
db = get_db_manager()

# --- DB-persisted runtime config (editable from the website via /api/config) ---
def _config_init():
    """Create the app_config table and load persisted values into RUNTIME_CONFIG."""
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS app_config (
            key TEXT PRIMARY KEY, value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        for r in (db.execute("SELECT key, value FROM app_config", fetch='all') or []):
            RUNTIME_CONFIG[r['key']] = r['value']
        print(f"app_config loaded: {RUNTIME_CONFIG}")
    except Exception as e:
        print(f"Warning: app_config init failed (using env defaults): {e}")

def config_set_db(key, value):
    """Persist a config key to the DB and update the in-memory RUNTIME_CONFIG."""
    db.execute("""INSERT INTO app_config (key, value, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value,
        updated_at = CURRENT_TIMESTAMP""", (key, str(value)))
    RUNTIME_CONFIG[key] = value

_config_init()

# Initialize backtest results store and register API routes
backtest_results_store = None
if BACKTEST_RESULTS_AVAILABLE:
    try:
        # Use shared DatabaseManager (PostgreSQL) so backtest results are in the main DB
        db.init_schema()
        backtest_results_store = BacktestResultsStore(db_manager=db)
        create_api_routes(app, backtest_results_store)
        print("Backtest analytics API routes registered (using main database)")
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
    """Place a buy or sell order. Routes to the DAS SIM backend when FALCON_EXECUTION=das,
    otherwise to the in-memory paper bot. DAS orders are dry-run unless FALCON_DAS_LIVE=1."""
    data = request.json or {}
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

    # --- DAS SIM backend ---
    if RUNTIME_CONFIG.get("execution_backend") == "das":
        if not DAS_AVAILABLE:
            return jsonify({"error": "DAS backend unavailable"}), 503
        client = None
        try:
            client = DASExecutionClient(live=(os.getenv("FALCON_DAS_LIVE") == "1"))
            client.connect()
            ok, resp = client.login()
            if not ok:
                return jsonify({"error": "DAS login failed", "detail": resp[:200]}), 502
            result = client.place_order(symbol, "B" if side == "buy" else "S",
                                        int(quantity), price if order_type == "limit" else None)
            result["backend"] = "das"
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"DAS order failed: {e}", "backend": "das"}), 500
        finally:
            if client:
                client.close()

    # --- paper bot backend ---
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503
    result = bot.place_order(symbol, side, int(quantity), order_type, price)
    result["backend"] = "paper"
    if result.get('status') == 'success':
        return jsonify(result), 200
    return jsonify(result), 400


@app.route('/api/das/health')
def das_health():
    """DAS SIM connection status (connect + login + BP/positions read; no orders)."""
    if not DAS_AVAILABLE:
        return jsonify({"connected": False, "error": "das_execution unavailable"}), 503
    info = das_health_check()
    info["execution_backend"] = RUNTIME_CONFIG.get("execution_backend")
    return jsonify(info), (200 if info.get("connected") else 502)


def _parse_das_buying_power(raw):
    """Extract a float buying power from the raw `GET BP` response string.
    DAS streams free-form text (e.g. "BP 50000.00" or "$BP ..."); grab the first
    number that looks like a dollar amount."""
    import re
    if not raw:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", raw.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_das_positions(raw):
    """Parse `POSREFRESH` output into [{symbol, qty, avgPrice, side}].
    Server response format (das-cmd-api): %POS symbol type qty avgcost initqty initprice realized.
    `type` is position-STATE, not account (memory das-pos-type-not-account)."""
    out = []
    if not raw:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line.upper().startswith("%POS"):
            continue
        parts = line.split()
        # %POS symbol type qty avgcost ...
        if len(parts) < 5:
            continue
        try:
            symbol = parts[1]
            qty = float(parts[3])
            avg_price = float(parts[4])
        except (ValueError, IndexError):
            continue
        out.append({
            "symbol": symbol,
            "qty": qty,
            "avgPrice": avg_price,
            "side": "long" if qty >= 0 else "short",
        })
    return out


@app.route('/api/das/account')
def get_das_account():
    """READ-ONLY DAS account/positions view (distinct from the paper-bot /api/account).
    Connects to DAStrader, logs in, reads BP + positions via the existing read-only
    methods (buying_power(), positions()) — never places or touches any order. The real
    SIM/LIVE account id + numeric buying power + open positions, not the static paper value.
    503 when DAS is unavailable; 502 on login failure."""
    if not DAS_AVAILABLE:
        return jsonify({"error": "das_execution unavailable"}), 503
    client = None
    try:
        client = DASExecutionClient(live=(os.getenv("FALCON_DAS_LIVE") == "1"))
        client.connect()
        ok, resp = client.login()
        if not ok:
            return jsonify({"error": "DAS login failed", "detail": resp[:200]}), 502
        bp_raw = client.buying_power()
        pos_raw = client.positions()
        return jsonify({
            "account": client.account,
            "buyingPower": _parse_das_buying_power(bp_raw),
            "positions": _parse_das_positions(pos_raw),
        }), 200
    except Exception as e:
        return jsonify({"error": f"DAS account read failed: {e}"}), 503
    finally:
        if client:
            client.close()


@app.route('/api/config', methods=['GET'])
def get_config():
    """Current runtime config (DB-persisted) + metadata for the website settings panel."""
    return jsonify({
        "execution_backend": RUNTIME_CONFIG.get("execution_backend", "paper"),
        "das_available": DAS_AVAILABLE,
        "das_account": os.getenv("DAS_ACCOUNT", ""),
        "das_live_ceiling": os.getenv("FALCON_DAS_LIVE") == "1",
        "options": {"execution_backend": ["paper", "das"]},
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    """Persist config edits from the website to the DB (survives restarts)."""
    data = request.json or {}
    updated = {}
    if "execution_backend" in data:
        v = str(data["execution_backend"]).lower()
        if v not in ("paper", "das"):
            return jsonify({"error": "execution_backend must be 'paper' or 'das'"}), 400
        if v == "das" and not DAS_AVAILABLE:
            return jsonify({"error": "DAS backend unavailable on this host"}), 400
        config_set_db("execution_backend", v)
        updated["execution_backend"] = v
    if not updated:
        return jsonify({"error": "no recognized config keys"}), 400
    return jsonify({"status": "success", "updated": updated, "config": RUNTIME_CONFIG}), 200

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
    """Current trading signals (hybrid). Primary source: the latest per-symbol
    strategy signal the orchestrator logs to strategy_signals. Fallback: directional
    signals derived from the persisted intraday-scanner setups (same source as
    /api/recommendations) when no strategy signals have been logged yet.

    The dashboard process has no in-process market-data feed, so signals come from
    persisted data, never live in-process TA."""
    if not bot:
        return jsonify({"error": "Bot not initialized"}), 503

    try:
        # --- Primary: orchestrator-logged strategy signals, latest per symbol. ---
        try:
            rows = db.execute(
                """SELECT DISTINCT ON (symbol) symbol, signal_type, signal_reason,
                          confidence, market_price, action_taken, timestamp
                     FROM strategy_signals
                    ORDER BY symbol, timestamp DESC""",
                fetch='all') or []
        except Exception:
            rows = []  # table missing/empty -> fall through to scanner

        if rows:
            signals = [{
                "symbol": r.get('symbol'),
                "signal": (r.get('signal_type') or '').upper(),
                "reason": r.get('signal_reason'),
                "confidence": float(r.get('confidence')) if r.get('confidence') is not None else None,
                "price": float(r.get('market_price')) if r.get('market_price') is not None else None,
                "action_taken": r.get('action_taken'),
                "timestamp": str(r.get('timestamp')) if r.get('timestamp') else None,
            } for r in rows]
            return jsonify({"signals": signals, "status": "success", "source": "strategy_signals"})

        # --- Fallback: derive directional signals from persisted scanner setups. ---
        from falcon_screener.profile_manager import ProfileManager
        manager = ProfileManager(db)
        best = {}        # symbol -> highest-confidence candidate
        recency = None
        for profile in manager.list_profiles(enabled_only=True):
            runs = manager.get_profile_runs(profile.id, days=1)
            if not runs:
                continue
            run_data = runs[0].get('run_data', {}) or {}
            for rec in run_data.get('recommendations', []):
                ticker = rec.get('ticker')
                if not ticker:
                    continue
                recency = recency or rec.get('data_recency')
                direction = (rec.get('direction') or '').lower()
                signal = 'BUY' if direction in ('long', 'buy') else \
                         'SELL' if direction in ('short', 'sell') else 'WATCH'
                cand = {
                    "symbol": ticker,
                    "signal": signal,
                    "reason": rec.get('trigger_detail') or rec.get('reasoning'),
                    "confidence": rec.get('confidence_score'),
                    "setup_type": rec.get('setup_type'),
                    "entry": rec.get('entry') or rec.get('entry_price_range'),
                    "stop": rec.get('stop') or rec.get('stop_loss'),
                    "target": rec.get('target') or rec.get('target_price'),
                    "data_recency": rec.get('data_recency'),
                }
                existing = best.get(ticker)
                if not existing or (cand.get('confidence') or 0) > (existing.get('confidence') or 0):
                    best[ticker] = cand

        signals = sorted(best.values(), key=lambda s: (s.get('confidence') or 0), reverse=True)
        if signals:
            return jsonify({"signals": signals, "status": "success",
                            "source": "scanner", "data_recency": recency})

        return jsonify({"signals": [], "status": "no_data",
                        "reason": "no strategy signals logged and no recent scanner setups"})
    except Exception as e:
        return jsonify({"signals": [], "status": "error", "error": str(e)}), 500

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

    # Graceful degradation: get_ai_analysis() is not implemented on the current
    # PaperTradingBot. Return an explicit "unavailable" payload instead of a 500
    # until the AI analysis feature is implemented.
    if not hasattr(bot, 'get_ai_analysis'):
        return jsonify({
            "analysis": None,
            "status": "unavailable",
            "reason": "AI analysis not implemented on bot"
        })

    try:
        analysis = bot.get_ai_analysis()
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"analysis": None, "status": "error", "error": str(e)}), 500


@app.route('/api/recommendations')
def get_recommendations():
    """Get AI stock recommendations from the screener database"""
    try:
        from falcon_screener.profile_manager import ProfileManager
        manager = ProfileManager(db)

        # Get all profile runs from today, merge recommendations
        all_recommendations = []
        latest_timestamp = None
        total_stocks = 0
        profiles_run = []

        for profile in manager.list_profiles(enabled_only=True):
            runs = manager.get_profile_runs(profile.id, days=1)
            if runs:
                latest_run = runs[0]  # Most recent
                run_data = latest_run.get('run_data', {})
                recommendations = run_data.get('recommendations', [])

                # Track metadata
                if not latest_timestamp or latest_run.get('run_timestamp', '') > latest_timestamp:
                    latest_timestamp = latest_run.get('run_timestamp')

                total_stocks += latest_run.get('stocks_found', 0)
                profiles_run.append({
                    'profile_name': profile.name,
                    'theme': profile.theme,
                    'stocks_found': latest_run.get('stocks_found', 0),
                    'run_type': latest_run.get('run_type')
                })

                # Add recommendations with profile source
                for rec in recommendations:
                    rec['_profile_source'] = profile.name
                    rec['_theme'] = profile.theme
                    all_recommendations.append(rec)

        # Deduplicate by ticker, keeping highest confidence. (No early no_data
        # return here: the intraday scanner below can still produce setups even
        # when the swing screener has no rows — the status is decided after merge.)
        ticker_map = {}
        for rec in all_recommendations:
            ticker = rec.get('ticker', '')
            if not ticker:
                continue
            existing = ticker_map.get(ticker)
            if not existing or rec.get('confidence_score', 0) > existing.get('confidence_score', 0):
                ticker_map[ticker] = rec

        # ---- INTRADAY SETUP SCANNER (falcon-trader #6) ----------------------
        # Lazily merge real, ranked, structured day-trade setups computed from
        # flat-file minute bars (FLAT FILES ONLY — never REST). In an image with
        # boto3 this runs in-process; in the dashboard image (no boto3) the scan
        # returns no in-process setups but the persisted "Intraday Scanner"
        # profile rows are already picked up by the ProfileManager loop above.
        # Either way we surface a data_recency label and never pass stale as live.
        scan = {"session_date": None, "last_bar_ts": None,
                "data_source": "flatfiles", "data_recency": None, "setups": []}
        try:
            from falcon_trader import intraday_scanner
            scan = intraday_scanner.scan_intraday_setups()
            for s in scan.get("setups", []):
                s['_theme'] = 'intraday_setup'
                s['_profile_source'] = 'intraday_scan'
                t = s.get('ticker', '')
                if not t:
                    continue
                existing = ticker_map.get(t)
                if (not existing
                        or s.get('edge_score', 0) > existing.get('edge_score', 0)
                        or s.get('confidence_score', 0) > existing.get('confidence_score', 0)):
                    ticker_map[t] = s
        except Exception as scan_err:
            # Never let the scanner break the existing endpoint.
            print(f"Warning: intraday scan skipped: {scan_err}")

        # If the in-process scan could not read flat files (boto3 absent in the
        # dashboard image), recover the honest STALE recency / session metadata
        # from any persisted intraday setup so staleness is still labeled, never
        # silently dropped or mistaken for live.
        # A data-bearing in-process scan carries one of the honest tier labels
        # (#9 added DELAYED/DEGRADED above STALE). Any of these means the scan ran
        # in-process and its recency must be passed through verbatim — no recovery,
        # no re-labeling. Only when NONE is present (boto3 absent -> scan empty) do
        # we recover the honest recency from a persisted setup.
        _DATA_TIERS = ("STALE", "DELAYED", "DEGRADED", "LIVE")
        in_process_ok = bool(scan.get("data_recency")) and any(
            t in str(scan.get("data_recency")) for t in _DATA_TIERS
        )
        if not in_process_ok:
            for rec in ticker_map.values():
                rec_recency = str(rec.get('data_recency') or '')
                if rec.get('_theme') == 'intraday_setup' and any(t in rec_recency for t in _DATA_TIERS):
                    scan['data_recency'] = rec.get('data_recency')
                    scan['session_date'] = rec.get('session_date')
                    scan['last_bar_ts'] = scan.get('last_bar_ts') or rec.get('last_bar_ts')
                    scan['data_source'] = rec.get('data_source') or scan.get('data_source')
                    break

        # Re-sort the merged set by the edge proxy first, confidence second.
        merged = sorted(
            ticker_map.values(),
            key=lambda x: (x.get('edge_score', 0), x.get('confidence_score', 0)),
            reverse=True,
        )

        # status='success' whenever the scan OR the screener produced anything.
        if not merged and not scan.get("setups"):
            return jsonify({
                "status": "no_data",
                "message": "No screening results available yet",
                "recommendations": [],
                "data_source": scan.get("data_source", "flatfiles"),
                "session_date": scan.get("session_date"),
                "last_bar_ts": scan.get("last_bar_ts"),
                "data_recency": scan.get("data_recency"),
            })

        # ---- RECENCY / EXPIRY GATE (falcon-trader #7) -----------------------
        # Never serve a setup whose validity window has closed (valid_until in
        # the past) or whose data is flagged STALE as a LIVE, actionable
        # suggestion. Acting on an expired/stale signal is the exact failure
        # this gate prevents. Non-actionable setups are still returned under a
        # separate key (not silently dropped) so the UI can show them as DEAD.
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo as _ZI
            _now_et = _dt.now(_ZI("America/New_York"))
        except Exception:
            from datetime import timezone as _tz
            _now_et = _dt.now(_tz.utc)

        def _is_expired(rec):
            vu = rec.get('valid_until')
            if not vu:
                return False
            try:
                end = _dt.fromisoformat(str(vu))
            except Exception:
                return False
            if end.tzinfo is None:
                return False
            return _now_et > end

        _top_stale = 'STALE' in str(scan.get('data_recency') or '')
        for _rec in merged:
            _exp = _is_expired(_rec)
            _stale = _top_stale or 'STALE' in str(_rec.get('data_recency') or '')
            _rec['expired'] = _exp
            _rec['actionable'] = not (_exp or _stale)

        live = [r for r in merged if r.get('actionable')]
        expired_recs = [r for r in merged if not r.get('actionable')]

        return jsonify({
            "status": "success",
            "timestamp": latest_timestamp,
            "screen_type": "multi-profile",
            "total_stocks_screened": total_stocks,
            "profiles_run": profiles_run,
            # Only LIVE, in-window setups are served as actionable suggestions.
            "recommendations": live,
            "expired_recommendations": expired_recs,
            "actionable_count": len(live),
            "expired_count": len(expired_recs),
            "message": (None if live else
                        "No live setups — all are past their validity window or "
                        "flagged STALE. Do NOT trade these."),
            # Top-level recency/provenance (mandatory honest-staleness labeling).
            "data_source": scan.get("data_source", "flatfiles"),
            "session_date": scan.get("session_date"),
            "last_bar_ts": scan.get("last_bar_ts"),
            "data_recency": scan.get("data_recency"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommendations/history')
def get_recommendations_history():
    """Get historical screening results from database"""
    try:
        from falcon_screener.profile_manager import ProfileManager
        manager = ProfileManager(db)

        days = int(request.args.get('days', 7))

        # Get runs from all profiles
        history = []
        for profile in manager.list_profiles():
            runs = manager.get_profile_runs(profile.id, days=days)
            for run in runs:
                history.append({
                    "timestamp": run.get('run_timestamp'),
                    "profile_name": profile.name,
                    "theme": profile.theme,
                    "run_type": run.get('run_type'),
                    "stocks_found": run.get('stocks_found', 0),
                    "recommendation_count": run.get('recommendations_generated', 0)
                })

        # Sort by timestamp descending
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/squawk')
def get_squawk():
    """Benzinga-style SQUAWK (falcon-trader #8) — a SHORT, ranked, de-duped,
    reverse-chronological stream of REAL Polygon breaking headlines, gated to the
    in-play / watchlist universe and high-impact catalyst types.

    PRIVATE dashboard surface only — this endpoint NEVER writes to Slack / Notion
    / any public channel. News is the ONE sanctioned Polygon REST use (publisher
    publish-time, NOT the 15-min bars/quotes lag). Separate from
    /api/recommendations — squawk headlines are never merged into the ranked
    trade-setup contract. Degrades gracefully (last-good cache) on 429/error and
    never raises a 500 that would break the page.
    """
    try:
        from falcon_trader import squawk_feed
        # Universe = FALCON_DASHBOARD_SYMBOLS unioned with the latest scan / merged
        # rec tickers (shared context, no forked list); resolver falls back to the
        # scanner's universe only when both are empty.
        universe = squawk_feed.resolve_universe(
            extra_tickers=squawk_feed._latest_scan_tickers())
        result = squawk_feed.fetch_squawk(universe)

        if not result.get("items"):
            return jsonify({
                "status": result.get("status", "no_data"),
                "message": result.get("message", "No squawk headlines available."),
                "items": [],
                "universe": result.get("universe", universe),
                "fetched_at": result.get("fetched_at"),
                "data_recency": result.get(
                    "data_recency", "LIVE (Polygon news, publisher-time)"),
                "source": result.get("source", "polygon_news"),
            })

        return jsonify({
            "status": result.get("status", "success"),
            "items": result.get("items", []),
            "count": result.get("count", len(result.get("items", []))),
            "universe": result.get("universe", universe),
            "fetched_at": result.get("fetched_at"),
            "data_recency": result.get(
                "data_recency", "LIVE (Polygon news, publisher-time)"),
            "source": result.get("source", "polygon_news"),
            "message": result.get("message"),
        })
    except Exception as e:
        # NEVER 500 the dashboard — return an empty, clearly-labeled envelope.
        print(f"Warning: squawk endpoint degraded: {e}")
        return jsonify({
            "status": "error",
            "items": [],
            "data_recency": "DEGRADED (squawk error)",
            "source": "polygon_news",
            "message": str(e),
        })


@app.route('/')
def serve_index():
    """Serve the landing page"""
    return send_file('www/index.html')


@app.route('/dashboard')
@app.route('/trading')
def serve_trading_dashboard():
    """Serve the main trading dashboard with account, positions, and trades"""
    return send_file('www/trading.html')


@app.route('/orchestrator')
@app.route('/orchestrator.html')
def serve_orchestrator():
    """Serve the orchestrator dashboard"""
    return send_file('www/orchestrator.html')


@app.route('/discipline-workshop')
@app.route('/discipline-workshop.html')
def serve_discipline_workshop():
    """Serve the MIC Discipline Workshop calendar page."""
    return send_file('www/discipline-workshop.html')


@app.route('/api/discipline-workshop/<month>')
def get_discipline_workshop(month):
    """A month of Discipline Workshop records (plan-adherence WIN/LOSS), read from postgres.
    Canonical store is DynamoDB (falcon-trades, DW# namespace) + S3; this is the dashboard mirror."""
    import re
    if not re.fullmatch(r'\d{4}-\d{2}', month or ''):
        return jsonify({"error": "month must be YYYY-MM"}), 400
    try:
        rows = db.execute(
            "SELECT date, dow, discipline_result, adherence_pct, combined_r, realized_pnl, "
            "plan, tickers, goal, learned, changes, overview, submitted_to_mic, "
            "graduation_status, s3_artifact "
            "FROM discipline_workshop WHERE month=%s ORDER BY date", (month,), fetch='all') or []
        days, wins, losses = [], 0, 0
        for r in rows:
            res = r.get('discipline_result')
            if res == 'WIN':
                wins += 1
            elif res == 'LOSS':
                losses += 1
            num = lambda k: float(r[k]) if r.get(k) is not None else None
            days.append({
                "date": str(r.get('date')), "dow": r.get('dow'), "discipline_result": res,
                "adherence_pct": num('adherence_pct'), "combined_r": num('combined_r'),
                "realized_pnl": num('realized_pnl'), "plan": r.get('plan'), "tickers": r.get('tickers'),
                "goal": r.get('goal'), "learned": r.get('learned'), "changes": r.get('changes'),
                "overview": r.get('overview'), "submitted_to_mic": r.get('submitted_to_mic'),
                "graduation_status": r.get('graduation_status'), "s3_artifact": r.get('s3_artifact'),
            })
        graded = wins + losses
        return jsonify({
            "status": "success", "month": month, "days": days,
            "tally": {"wins": wins, "losses": losses, "graded": graded,
                      "adherence_rate": round(wins / graded, 3) if graded else None},
            "graduation": {"method": "moderator-vote",
                           "status": days[-1]["graduation_status"] if days else "in-progress"},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/strategies')
@app.route('/strategies.html')
def serve_strategies_page():
    """Serve the strategies list page"""
    return send_file('www/strategies.html')


@app.route('/diagnostics')
@app.route('/diagnostics.html')
def serve_diagnostics():
    """Serve the diagnostics / API console page (surfaces every endpoint — issue #4)"""
    return send_file('www/diagnostics.html')


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
        try:
            from falcon_trader.strategy_analytics import StrategyAnalytics
            analytics = StrategyAnalytics(DB_PATH)
            leaderboard = analytics.get_all_strategies_leaderboard()
        except (ImportError, ConnectionError) as e:
            return jsonify({"error": "analytics unavailable", "details": str(e)}), 503

        # Also get all strategies (not just those with performance data).
        # The active_strategies table is created lazily on first activation; if
        # it doesn't exist yet there simply are no active strategies.
        try:
            rows = db.execute('''
                SELECT id, strategy_name, status, allocation_pct,
                       performance_weight, created_at, activated_at
                FROM active_strategies
                WHERE status IN ('active', 'paused')
                ORDER BY activated_at DESC
            ''', fetch='all') or []
        except Exception:
            rows = []

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
        try:
            from falcon_trader.strategy_analytics import StrategyAnalytics
            analytics = StrategyAnalytics(DB_PATH)
            summary = analytics.get_strategy_summary(strategy_id)
        except (ImportError, ConnectionError) as e:
            return jsonify({"error": "analytics unavailable", "details": str(e)}), 503

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
        try:
            from falcon_trader.strategy_analytics import StrategyAnalytics
            analytics = StrategyAnalytics(DB_PATH)
            leaderboard = analytics.get_all_strategies_leaderboard()
        except (ImportError, ConnectionError) as e:
            return jsonify({"error": "analytics unavailable", "details": str(e)}), 503

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
        try:
            from falcon_trader.strategy_analytics import StrategyAnalytics
            analytics = StrategyAnalytics(DB_PATH)
            stats = analytics.get_aggregate_statistics()
        except (ImportError, ConnectionError) as e:
            return jsonify({"error": "analytics unavailable", "details": str(e)}), 503

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


@app.route('/signals-risk')
@app.route('/signals-risk.html')
def serve_signals_risk():
    """Serve the Signal/Risk analyzer page (sangre-signal integration)"""
    return send_file('www/signals-risk.html')


# Spanish translations for sangre-signal risk-flag messages, keyed on the exact
# English emission from sangre_signal/analyzers/risk_analyzer.py. The float message
# interpolates a number, so it is matched by prefix and re-interpolated below.
RISK_FLAG_ES = {
    "Country of origin is in red-flag list": "País de origen en lista de alto riesgo",
    "Country of origin is non-US": "País de origen no estadounidense",
    "Headquarters location includes red-flag keywords": "La sede incluye ubicaciones de alto riesgo",
    "ADR/listed foreign issuer": "ADR/emisor extranjero cotizado",
}


def _localize_flag_message_es(message_en):
    """Return the Spanish flag message for an English emission, or None if unmapped.
    Handles the templated 'Float below {N}M shares' message by re-interpolating N."""
    if message_en in RISK_FLAG_ES:
        return RISK_FLAG_ES[message_en]
    if message_en.startswith("Float below ") and message_en.endswith(" shares"):
        amount = message_en[len("Float below "):-len(" shares")]
        return f"Flotante inferior a {amount} acciones"
    return None


@app.route('/api/risk-analysis', methods=['GET'])
def get_risk_analysis():
    """Stock risk analysis via sangre-signal.
    Query: tickers (required, comma-sep), lang ('en'|'es', narrative only).
    Returns: {lang, ai_enabled, results:[{ticker, ok, has_risks, flags:[{flag_type,message,severity}], narrative} | {ticker, ok:false, error}]}
    """
    try:
        import os
        from sangre_signal.fetchers import fetch_stock_info, fetch_vix
        from sangre_signal.analyzers import analyze_stock_risks
        from sangre_signal.formatters import get_formatter
        from sangre_signal.config import RED_FLAGS

        raw = (request.args.get('tickers') or '').strip()
        if not raw:
            return jsonify({"error": "Missing 'tickers' query parameter"}), 400
        lang = (request.args.get('lang') or 'en').lower()
        if lang not in ('en', 'es'):
            lang = 'en'

        tickers = []
        for t in raw.split(','):
            t = t.strip().upper()
            if t and t not in tickers:
                tickers.append(t)
        tickers = tickers[:10]
        if not tickers:
            return jsonify({"error": "No valid tickers provided"}), 400

        ai_enabled = bool(os.getenv('ANTHROPIC_API_KEY') or os.getenv('PERPLEXITY_API_KEY'))
        if os.getenv('ANTHROPIC_API_KEY'):
            fmt_name = 'claude'
        elif os.getenv('PERPLEXITY_API_KEY'):
            fmt_name = 'perplexity'
        else:
            fmt_name = 'claude'
        formatter = get_formatter(fmt_name, language=lang)

        try:
            vix_value = fetch_vix()
        except Exception:
            vix_value = None

        results = []
        for ticker in tickers:
            try:
                stock_info = fetch_stock_info(ticker)
                if stock_info is None:
                    results.append({"ticker": ticker, "ok": False, "error": f"Could not fetch data for ticker '{ticker}'"})
                    continue
                # yfinance soft-fails to a populated "Unknown" shell rather than empty info.
                # No name AND no price => treat as not-found; never render a false-safe "no risks" card.
                if not (stock_info.long_name or stock_info.short_name) and stock_info.regular_market_price is None:
                    msg = ("Símbolo no encontrado" if lang == "es" else "Ticker not found")
                    results.append({"ticker": ticker, "ok": False, "error": msg})
                    continue
                risk = analyze_stock_risks(stock_info)
                flags = []
                for f in risk.flags:
                    message_en = f.message
                    if lang == 'es':
                        message = _localize_flag_message_es(message_en) or message_en
                    else:
                        message = message_en
                    flags.append({"flag_type": f.flag_type, "message": message, "message_en": message_en, "severity": f.severity.value.upper()})
                narrative = formatter.format(stock_info, risk, RED_FLAGS.min_free_float, vix_value)
                results.append({"ticker": ticker, "ok": True, "has_risks": risk.has_risks, "flags": flags, "narrative": narrative})
            except Exception as e:
                results.append({"ticker": ticker, "ok": False, "error": str(e)})

        return jsonify({"lang": lang, "ai_enabled": ai_enabled, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# SCREENER PROFILE ENDPOINTS
# ============================================

@app.route('/api/screener/profiles', methods=['GET'])
def list_screener_profiles():
    """List all screener profiles"""
    try:
        from falcon_screener.profile_manager import ProfileManager
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
        from falcon_screener.profile_manager import ProfileManager, ScreenerProfile
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
        from falcon_screener.profile_manager import ProfileManager
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
        from falcon_screener.profile_manager import ProfileManager, ScreenerProfile
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
        from falcon_screener.profile_manager import ProfileManager
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
        from falcon_screener.profile_manager import ProfileManager
        from falcon_screener.multi_screener import MultiScreener
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
        from falcon_screener.profile_manager import ProfileManager
        from falcon_screener.feedback_loop import WeightFeedbackLoop
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
        from falcon_screener.profile_manager import ProfileManager
        from falcon_screener.yaml_serializer import ProfileYAMLSerializer
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
        from falcon_screener.profile_manager import ProfileManager
        from falcon_screener.yaml_serializer import ProfileYAMLSerializer
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
        from falcon_screener.profile_manager import ProfileManager
        from falcon_screener.feedback_loop import WeightFeedbackLoop
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
        from falcon_screener.profile_manager import ProfileManager
        from falcon_screener.profile_templates import initialize_default_profiles
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


# ============================================
# AI ADVISOR ENDPOINTS
# ============================================

@app.route('/advisor')
@app.route('/advisor.html')
def serve_advisor():
    """Serve the AI advisor dashboard"""
    return send_file('www/advisor.html')


@app.route('/api/advisor/proposals', methods=['GET'])
def list_advisor_proposals():
    """List advisor proposals, optionally filtered by status."""
    status = request.args.get('status', 'pending')
    try:
        if status == 'all':
            rows = db.execute(
                '''SELECT id, strategy_name, proposal_type, change_description,
                          analysis_summary, expected_improvement,
                          status, api_cost_usd, created_at,
                          current_sharpe, proposed_sharpe,
                          current_win_rate, proposed_win_rate,
                          current_total_return, proposed_total_return
                   FROM strategy_proposals ORDER BY created_at DESC''',
                fetch='all'
            )
        else:
            rows = db.execute(
                '''SELECT id, strategy_name, proposal_type, change_description,
                          analysis_summary, expected_improvement,
                          status, api_cost_usd, created_at,
                          current_sharpe, proposed_sharpe,
                          current_win_rate, proposed_win_rate,
                          current_total_return, proposed_total_return
                   FROM strategy_proposals WHERE status = %s
                   ORDER BY created_at DESC''',
                (status,), fetch='all'
            )
        proposals = []
        for row in (rows or []):
            if isinstance(row, dict):
                proposals.append(row)
            else:
                proposals.append({
                    'id': row[0], 'strategy_name': row[1],
                    'proposal_type': row[2], 'change_description': row[3],
                    'analysis_summary': row[4], 'expected_improvement': row[5],
                    'status': row[6], 'api_cost_usd': row[7],
                    'created_at': row[8],
                    'current_sharpe': row[9], 'proposed_sharpe': row[10],
                    'current_win_rate': row[11], 'proposed_win_rate': row[12],
                    'current_total_return': row[13], 'proposed_total_return': row[14],
                })
        return jsonify(proposals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/advisor/proposals/<int:proposal_id>', methods=['GET'])
def get_advisor_proposal(proposal_id):
    """Get full proposal detail including code."""
    try:
        row = db.execute(
            '''SELECT * FROM strategy_proposals WHERE id = %s''',
            (proposal_id,), fetch='one'
        )
        if not row:
            return jsonify({"error": "Proposal not found"}), 404
        proposal = dict(row) if hasattr(row, 'keys') else {
            'id': row[0], 'strategy_name': row[1],
            'proposal_type': row[2], 'current_code': row[3],
            'proposed_code': row[4], 'analysis_summary': row[5],
            'change_description': row[6], 'expected_improvement': row[7],
            'current_sharpe': row[8], 'proposed_sharpe': row[9],
            'current_win_rate': row[10], 'proposed_win_rate': row[11],
            'current_total_return': row[12], 'proposed_total_return': row[13],
            'status': row[14], 'reviewed_by': row[15],
            'reviewed_at': row[16], 'review_notes': row[17],
            'api_cost_usd': row[18], 'created_at': row[19],
            'applied_at': row[20],
        }
        return jsonify(proposal)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/advisor/proposals/<int:proposal_id>/approve', methods=['POST'])
def approve_advisor_proposal(proposal_id):
    """Approve and apply a proposal."""
    try:
        row = db.execute(
            'SELECT strategy_name, proposed_code, status FROM strategy_proposals WHERE id = %s',
            (proposal_id,), fetch='one'
        )
        if not row:
            return jsonify({"error": "Proposal not found"}), 404

        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        proposed_code = row['proposed_code'] if isinstance(row, dict) else row[1]
        status = row['status'] if isinstance(row, dict) else row[2]

        if status != 'pending':
            return jsonify({"error": f"Proposal is already '{status}'"}), 400

        now = datetime.now().isoformat()
        data = request.get_json(silent=True) or {}
        reviewer = data.get('reviewed_by', 'dashboard')

        # Update proposal status
        db.execute(
            '''UPDATE strategy_proposals SET
               status = %s, reviewed_by = %s, reviewed_at = %s, applied_at = %s
               WHERE id = %s''',
            ('approved', reviewer, now, now, proposal_id)
        )

        # Apply to strategy_roster
        db.execute(
            '''UPDATE strategy_roster SET
               strategy_code = %s, updated_at = %s
               WHERE strategy_name = %s''',
            (proposed_code, now, name)
        )

        # Log rotation
        db.execute(
            '''INSERT INTO strategy_rotation_log
               (strategy_name, from_status, to_status, reason, rotated_at)
               VALUES (%s, %s, %s, %s, %s)''',
            (name, 'advisor_proposal', 'code_updated',
             f'Approved proposal #{proposal_id}', now)
        )

        # Record improvement
        try:
            from falcon_core.backtesting.advisor import CostTracker
            CostTracker(db).record_improvement(name, True)
        except Exception:
            pass

        return jsonify({"status": "approved", "strategy_name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/advisor/proposals/<int:proposal_id>/reject', methods=['POST'])
def reject_advisor_proposal(proposal_id):
    """Reject a proposal."""
    try:
        row = db.execute(
            'SELECT strategy_name, status FROM strategy_proposals WHERE id = %s',
            (proposal_id,), fetch='one'
        )
        if not row:
            return jsonify({"error": "Proposal not found"}), 404

        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        status = row['status'] if isinstance(row, dict) else row[1]

        if status != 'pending':
            return jsonify({"error": f"Proposal is already '{status}'"}), 400

        now = datetime.now().isoformat()
        data = request.get_json(silent=True) or {}

        db.execute(
            '''UPDATE strategy_proposals SET
               status = %s, reviewed_by = %s, reviewed_at = %s, review_notes = %s
               WHERE id = %s''',
            ('rejected', data.get('reviewed_by', 'dashboard'), now,
             data.get('review_notes', ''), proposal_id)
        )

        # Record no-improvement
        try:
            from falcon_core.backtesting.advisor import CostTracker
            CostTracker(db).record_improvement(name, False)
        except Exception:
            pass

        return jsonify({"status": "rejected", "strategy_name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/advisor/proposals/<int:proposal_id>/backtest', methods=['POST'])
def backtest_advisor_proposal(proposal_id):
    """Trigger backtest comparison for a proposal."""
    try:
        from falcon_core.backtesting.advisor import StrategyAdvisor
        advisor = StrategyAdvisor(db)
        result = advisor.backtest_proposal(proposal_id)
        if result:
            return jsonify(result)
        return jsonify({"error": "Backtest failed"}), 500
    except ImportError:
        return jsonify({"error": "Backtesting dependencies not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/advisor/budgets', methods=['GET'])
def get_advisor_budgets():
    """Get budget status for all strategies."""
    try:
        from falcon_core.backtesting.advisor import CostTracker
        tracker = CostTracker(db)
        budgets = tracker.get_all_budgets()
        return jsonify(budgets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/advisor/usage', methods=['GET'])
def get_advisor_usage():
    """Get API usage history."""
    try:
        limit = int(request.args.get('limit', 100))
        rows = db.execute(
            '''SELECT id, service, model, strategy_name,
                      input_tokens, output_tokens, cost_usd,
                      request_type, created_at
               FROM api_usage ORDER BY created_at DESC LIMIT %s''',
            (limit,), fetch='all'
        )
        usage = []
        for row in (rows or []):
            if isinstance(row, dict):
                usage.append(row)
            else:
                usage.append({
                    'id': row[0], 'service': row[1], 'model': row[2],
                    'strategy_name': row[3], 'input_tokens': row[4],
                    'output_tokens': row[5], 'cost_usd': row[6],
                    'request_type': row[7], 'created_at': row[8],
                })
        return jsonify(usage)
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

    # Signal/Risk tab: ANTHROPIC_API_KEY (preferred) / PERPLEXITY_API_KEY (fallback)
    # are read at request time in /api/risk-analysis; if both are absent the
    # structured flags still render and the narrative uses the library fallback.
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
            _syms_env = os.getenv("FALCON_DASHBOARD_SYMBOLS", "SPY,QQQ,AAPL")
            _symbols = [s.strip().upper() for s in _syms_env.split(",") if s.strip()]
            print(f"Bot symbol universe ({len(_symbols)}): {_symbols}")
            initialize_bot(
                massive_api_key=MASSIVE_API_KEY,
                claude_api_key=CLAUDE_API_KEY,
                symbols=_symbols,
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
