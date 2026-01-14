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

Environment variables:
- `MASSIVE_API_KEY` - Polygon.io API key
- `DB_TYPE` - Database type
- `DB_PATH` - Database path

## License

MIT
