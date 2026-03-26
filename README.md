# Falcon Trader

Paper trading bot and strategy orchestrator for the Falcon Trading Platform.

## Installation

```bash
pip install git+https://github.com/TradingAsBuddies/falcon-trader.git
```

## Features

- **Multi-Strategy Orchestrator**: Routes stocks to optimal strategies based on real market data (price, volatility, classification)
- **Intraday Trading**: Minute-level bars from Polygon.io with per-strategy intervals
- **Paper Trading**: Simulates trades with real market data, tracks P&L in PostgreSQL
- **Strategy Engines**: Momentum Breakout (1m), RSI Mean Reversion (5m), Bollinger Bands (5m)
- **Market Dashboard**: Real-time market indicators, gainers/losers, trending tickers, breaking news
- **Sentinel Health Checks**: `/sentinels` page and `falcon-sentinel` CLI
- **Resilient HTTP**: All API calls use incremental backoff (no silent failures)
- **Timezone Safe**: All timestamps are US/Eastern (market time)

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

### Health Checks

```bash
falcon-sentinel
```

## Dashboard Pages

| Path | Purpose |
|------|---------|
| `/` | Landing page |
| `/dashboard` | Account, positions, trades |
| `/market` | Market indicators (red-to-green vs prev close), top gainers/losers, trending tickers, breaking news |
| `/sentinels` | Health check status for all subsystems |
| `/orchestrator` | Multi-strategy orchestration |
| `/advisor` | AI strategy advisor proposals |
| `/quote/<SYMBOL>` | Redirect to Finviz Elite quote page |

## Market Page

Auto-refreshes every 60 seconds. Shows:
- **Indicators**: SPY, TQQQ, VIX (VIXY), Brent Crude (BNO) — green/yellow/red vs yesterday's close
- **Top Gainers / Losers**: From Polygon snapshots
- **In The News**: Trending tickers extracted from headlines
- **Breaking News**: Polygon.io, Finviz Elite, CNBC, CNBC World, Yahoo Finance (US + global + commodities)
- **Alerts**: Banner when a new ticker appears in any list

## Strategy Routing

The executor classifies stocks using daily Polygon data, then routes to the best engine:

| Stock Type | Strategy | Interval |
|-----------|----------|----------|
| Penny stocks (<$5) | Momentum Breakout | 1m |
| High volatility (>30%) | Momentum Breakout | 1m |
| ETFs (SPY, QQQ, etc.) | RSI Mean Reversion | 5m |
| Stable large caps | RSI Mean Reversion | 5m |
| Energy sector | Bollinger Mean Reversion | 5m |

## Strategies

Strategy code is **not in this repo**. Strategies live in:
- The database (`strategy_roster` table) — loaded at runtime
- The private [falcon-strategies](https://github.com/TradingAsBuddies/falcon-strategies) repo

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market` | GET | Market indicators, gainers, losers |
| `/api/market/news` | GET | Breaking news from Polygon, Finviz, CNBC, Yahoo |
| `/api/sentinels` | GET | Run all health checks |
| `/api/account` | GET | Account balance |
| `/api/positions` | GET | Open positions |
| `/api/trades` | GET | Trade history |
| `/api/performance` | GET | Performance metrics |
| `/quote/<SYMBOL>` | GET | Redirect to Finviz Elite |

## Configuration

Environment variables (via `/etc/falcon/falcon.env`):
- `MASSIVE_API_KEY` / `POLYGON_API_KEY` — Polygon.io API key
- `MASSIVE_ACCESS_KEY` / `MASSIVE_SECRET_KEY` — Massive S3 keys for flat files
- `FINVIZ_AUTH_KEY` — Finviz Elite (news + quote links)
- `CLAUDE_API_KEY` — AI strategy advisor
- `DATABASE_URL` — PostgreSQL connection string

## License

MIT
