# falcon-trader — Handoff

**Role:** the trading engine (paper trading orchestrator) and — for now — the Flask operator dashboard.

**Canonical platform handoff:** `FALCON_HANDOFF.md` in `davdunc/falcon`. **Cross-cutting audit:** `falcon-platform/docs/GAP-ANALYSIS-2026-07.md`. **Strategy lifecycle:** `falcon-strategies/docs/STRATEGY-LIFECYCLE.md`.

## What it does (System A — the live path)
- Entry point `falcon-trader` = `run_orchestrator.py:main` (modes `--process/--monitor/--performance/--status/--once`, default daemon loops every 5 min).
- `TradeExecutor` reads AI screener JSON (`screened_stocks.json` under `FALCON_DATA_DIR`), then per stock: classify → route → fetch bars (`MarketDataFetcher`: Polygon → flat files → yfinance) → validate → engine signal → execute.
- **Broker = internal paper broker only.** `execute_buy/sell` write directly to `orders`/`positions`/`account` (cents-based cash accounting). **No Alpaca/IBKR integration exists.**
- Engines are hardcoded: `RSIEngine`, `MomentumEngine`, `BollingerEngine`.
- `falcon-dashboard` = `dashboard_server.py` (Flask :5000): `/`, `/api/market`, `/api/account`, `/api/positions`, `/api/trades`, `/orchestrator`, `/advisor`, `/strategies.html`.

## Runtime state (2026-07-20)
- Actively paper-trading in PostgreSQL: account ≈ +179% on a $10k paper start, 890 orders, 10 open positions. Orders tagged `momentum`/`rsi`.
- Trader logs `AI screener file not found: screened_stocks.json` every cycle — screener→trader handoff unwired (#13).

## Open issues (this repo)
- #13 orchestrator can't find `screened_stocks.json` (screener→trader handoff).
- #14 dead `StrategyOrchestrator`/`StrategyExecutor` path (missing modules `strategy_optimizer`/`strategy_analytics`, synthetic random-walk data).
- #15 no container healthcheck; port 5000 unpublished on the trader container.
- #16 move the Flask dashboard into `falcon-signal-web` (handoff §6.7).

## The big picture gap
The roster lifecycle in `falcon-core` and this executor are **decoupled**: promoting a strategy in `strategy_roster` does not change what trades here (engines are hardcoded, not roster-loaded). Wiring `TradeExecutor` to load `status='paper_trading'` roster strategies via `falcon_core...strategy_loader` is the single change that makes "submit a strategy" meaningful — tracked as **falcon-core#8**. Note the remote branch `feat/backtest-exec-endpoint` on falcon-core may be related work.

## System B (legacy, dead)
`strategy_orchestrator.py` / `strategy_executor.py` / `paper_trading_bot.py` — do not import cleanly and/or use synthetic data. Do not build on these; see #14.
