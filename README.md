# Falcon Trader

Paper trading bot and strategy orchestrator for the Falcon Trading Platform.

## Installation

```bash
pip install git+https://github.com/TradingAsBuddies/falcon-trader.git
```

## Features

- **Multi-Strategy Orchestrator**: Route stocks to optimal strategies
- **Paper Trading**: Simulate trades with real market data
- **Strategy Engines**: Momentum, RSI, Bollinger Band strategies
- **Dashboard API**: REST API for portfolio management
- **Web UI**: Real-time dashboard

## Usage

### Start Trading Bot

```bash
falcon-trader
```

### Start Dashboard

```bash
falcon-dashboard
# Dashboard available at http://localhost:5000
```

### Python API

```python
from falcon_trader.orchestrator.routers import StrategyRouter
from falcon_trader.orchestrator.engines import MomentumEngine

router = StrategyRouter()
engine = MomentumEngine()

# Route stock to strategy
strategy = router.route_stock("AAPL")
signal = engine.generate_signal(stock_data)
```

## Strategies

Built-in strategies:
- **Momentum Breakout**: Volume + price breakout
- **RSI Mean Reversion**: Oversold/overbought conditions
- **Bollinger Bands**: Band squeeze and expansion
- **One Candle**: Single-bar momentum patterns

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/portfolio` | GET | Current positions |
| `/api/orders` | GET/POST | Order management |
| `/api/strategy` | GET/POST | Strategy configuration |
| `/api/performance` | GET | Performance metrics |

## Configuration

### Required

| Variable | Purpose |
|---|---|
| `FALCON_API_TOKEN` | **Required.** Shared secret for the dashboard. The server refuses to start without it. Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`. Minimum 24 characters. |
| `MASSIVE_API_KEY` | Polygon.io API key. Also accepted as `POLYGON_API_KEY`. **No longer accepted as a command-line argument** — a key on `argv` is visible in `ps` to every user on the host. |

### Network surface

| Variable | Default | Purpose |
|---|---|---|
| `FALCON_BIND_HOST` | `127.0.0.1` | Listen address. Set to `0.0.0.0` **only** inside a container where Traefik and its auth middleware are the boundary. |
| `FALCON_BIND_PORT` | `5000` | Listen port. |
| `FALCON_CORS_ORIGINS` | *(unset)* | Comma-separated allowed origins. Unset means no cross-origin access is granted at all. The dashboard is same-origin and does not need this. |
| `FALCON_COOKIE_SECURE` | `0` | Set to `1` when served over HTTPS so the session cookie is HTTPS-only. |

### Dangerous operations — off by default

| Variable | Default | Purpose |
|---|---|---|
| `FALCON_ALLOW_DEPLOY` | `0` | Enables `POST /api/strategy/deploy`, which writes caller-supplied Python into the installed package and executes it. Leave off unless you are deliberately deploying a strategy. |
| `FALCON_DAS_LIVE` | *(unset)* | `1` sends real broker orders through DAS. |
| `FALCON_TRADING_ENABLED` | `1` | Kill switch. `0` halts every trading loop. Also honored: the halt file at `FALCON_HALT_FILE` (default `/var/lib/falcon/TRADING_HALTED`) — `touch` it to stop trading from any shell. |
| `FALCON_ALLOW_EXTENDED_HOURS` | `0` | `1` permits fills outside 09:30–16:00 ET. |
| `FALCON_ALLOW_STALE_FILLS` | `0` | `1` permits fills priced off a bar that is not from the current session. |

### Other

| Variable | Purpose |
|---|---|
| `DB_TYPE` | Database type |
| `DB_PATH` | Database path |
| `FALCON_INITIAL_BALANCE` | Fallback starting balance when the account row has none. |
| `FALCON_DASHBOARD_SYMBOLS` | Watchlist for the market-data thread. Does **not** limit which positions get marked. |

### Authenticating

Browsers sign in at `/login` and receive an HttpOnly `SameSite=Strict` session
cookie; the existing dashboard pages then work unchanged. Scripts send the token
as a header:

```bash
curl -H "Authorization: Bearer $FALCON_API_TOKEN" http://localhost:5000/api/account
```

`/health` is the only unauthenticated endpoint, so container and gateway health
checks keep working.

## License

MIT
