# falcon-trader — Component Spec

_Spec v1 · 2026-06-14 · living doc._

## Purpose
The execution half: real-time paper trading bot, strategy orchestrator, the **Flask web
dashboard + JSON API**, and the **DAStrader SIM execution backend**.

## Responsibilities
- **Dashboard / API** — `dashboard_server.py` (Flask, 70+ routes) serving 8 pages from `www/`
  + a `/diagnostics` API console. Endpoints: account/positions/trades, `/api/order`,
  `/api/bot/*`, `/api/config`, `/api/das/health`, `/api/advisor/*`, strategy lifecycle, charts.
- **DAS SIM execution** — `das_execution.py` `DASExecutionClient`: CMD-API socket, SIM account
  TR4425, authoritative positional `NEWORDER`, dry-run by default (`FALCON_DAS_LIVE`), SIM-only.
- **Runtime config** — `app_config` DB table + `/api/config` (execution backend paper↔das,
  web-editable, persisted, `RUNTIME_CONFIG`).
- **Paper bot** — `paper_trading_bot.py` `PaperTradingBot` (in-memory fills, real-time data).
- **Orchestrator** — `orchestrator/` (engines, validators, execution, routers); `strategy_*`,
  `youtube_strategies.py` (strategy extraction/activation), AI advisor.

## Interfaces
- Console scripts: `falcon-dashboard` (web), `falcon-trader` (orchestrator).
- HTTP: http://localhost:5000. Tools: `tools/website_evaluator.py` (consistency CI).

## Dependencies
`falcon-core`, Flask + flask-cors, PostgreSQL, DAStrader (CMD API, loopback), Polygon/Massive,
`anthropic` (advisor/extraction).

## Deployment
Image `localhost/falcon-trader:latest` (Fedora). Quadlet `dashboard.container` →
`dashboard.service`, **`Network=host`** (DAS loopback), override `dashboard.env`
(host-loopback DB + DAS creds + `FALCON_EXECUTION`). `das_execution.py` + `www/` bind-mounted.

## Status / notes
- DAS orders DRY-RUN until `FALCON_DAS_LIVE=1` + first RTH test. Stops blocked pre/post-market.
- Should consult `falcon-strategies` promotion guard before activating a strategy (open item).
- Related: `[[reference_falcon_das_execution]]`, falcon-platform HANDOFF.md, issues #3/#4 (closed).

## Roadmap / decisions (PM)
- **DAS account view (#7) — ADDRESSED.** Read-only `/api/das/account` surfaces real SIM/LIVE
  buying power + open positions via `DASExecutionClient.buying_power()`/`positions()`, parsed
  to JSON. Selectable alongside the paper-bot `/api/account`. No order paths touched.
- **strategy_analytics import crash (#5) — ADDRESSED.** Ship `strategy_analytics.py` as a thin
  adapter over falcon-core `BacktestResultsStore` (`get_all_strategies_summary` →
  `get_all_strategies_leaderboard`, `get_strategy_summary`); imports made package-qualified;
  endpoints return graceful 503 when analytics unavailable instead of raw 500.
- **Live intraday setup scan (#6) — IN PROGRESS (MVP scoped).** `/api/recommendations` returned
  `no_data` because the only source it reads (`profile_runs.run_data.recommendations`, a Finviz/AI
  **swing** screener) is unpopulated on this host. The fix is a **structured intraday setup
  scanner** — NOT a raw %-mover list (already evaluated: no edge, invites overtrading).

  **MVP shipped THIS run (honest scope):**
  - New module `falcon-trader/src/falcon_trader/intraday_scanner.py` builds a **ranked shortlist of
    <=5 day-trade setups** from minute bars via `falcon-core` `DataFeed`. ~~The code path MUST NOT
    reach the Polygon REST API~~ — **SUPERSEDED, see Freshness-tier refinement (#9) below.**

  - **FRESHNESS-TIER REFINEMENT (#9, user directive 2026-06-16).** The scanner / `/api/recommendations`
    is a **SUGGESTION / DISCOVERY surface**, not an execution surface. The standing
    `feedback_das_over_polygon` hard rule ("never REST / Polygon lags 15m / use DAS") governs
    **EXECUTION / live-trading PRICE decisions only** (order placement, real-time bars feeding a
    trade). It does **NOT** bind discovery surfaces. For the scanner, the data-source hierarchy is
    (prefer the FRESHEST AVAILABLE):
      1. **`"LIVE (DAS)"`** — DAS loopback, RTH top tier. **RESERVED HOOK, DEFERRED.** Must NEVER be
         emitted until real DAS bars are wired; emitting it from any Polygon/flat-file path is the
         worst possible mislabel (claims execution-grade freshness over delayed/stale data).
      2. **`"DELAYED ~15m (Polygon REST)"`** — Polygon REST aggregates, reused via
         `DataFeed.get_historical_data(..., source="polygon")` → `_try_polygon`. Attempted FIRST
         during RTH (>= ~09:45 ET so the ~15m-delayed window has real bars). This account's
         aggregates endpoint returns `status="DELAYED"` (data-bearing), so the polygon_client
         status-gate must accept `("OK","DELAYED",None)` exactly as `squawk_feed.py:346` does (the
         current `!= "OK"` break at `polygon_client.py:155` discards 958 valid bars — that is the
         enabling bug).
      3. **`"STALE (EOD flat-file, as of <session_date>)"`** — Massive/Polygon flat files via
         `DataFeed(source="flatfiles")`. **PRESERVED** as fallback / pre-market prep / multi-day
         indicator warm-up / symbols Polygon has no bars for. Removing Polygon entirely yields
         identical behavior to today (purely additive change).
    Failure strings reuse the squawk vocabulary verbatim: `"UNAVAILABLE (<reason>)"` /
    `"DEGRADED (<reason>)"`. **No other strings.** The flat-file string MUST always interpolate the
    actual session date.
  - **HONEST LABELING IS NON-NEGOTIABLE.** `data_recency` reflects the tier ACTUALLY USED for the
    data in hand, **never the tier attempted**. If Polygon is called and returns empty/error/rate-limited,
    the code falls back to flat files and the label is the **STALE/DEGRADED flat-file string — never
    DELAYED**. Distinguish the two empty cases: pre-open / no-session-yet → `STALE (… as of <prior
    session>)`; RTH-but-Polygon-failed → `DEGRADED (Polygon … → flat-file, as of <date>)`.
  - **Two-grain labeling that must agree:** a coarse `data_source` / `data_recency` on the payload
    envelope **AND** a per-setup `data_recency` + machine-readable `last_bar_ts` on **every** row.
    A mixed-tier scan shows the true per-symbol tier on each row; the envelope shows the **WORST
    (least-fresh)** tier present so the headline never over-promises. `last_bar_ts` is ISO-8601,
    timezone-aware **America/New_York** at both grains (Polygon bars are tz-naive UTC out of the
    client — normalize before surfacing). A tier label without `last_bar_ts` is non-compliant.
  - **Tier selection is TIME-and-availability aware.** Pre-open the freshest meaningful data IS the
    prior session's flat file (labeled STALE honestly, never dressed up as DELAYED just because
    Polygon was queried). The frontend MUST visibly render the tier (green LIVE / amber DELAYED /
    grey STALE badge); for DELAYED rows show age `(now_ET - last_bar_ts)` in minutes.
    **Rendering stale data as live, or DELAYED over flat-file data, is a FAILING condition.**
    `/api/recommendations` propagates the scanner's exact `data_recency` + `last_bar_ts` with **no
    re-labeling, defaulting, or field loss.**
  - **Execution still routes through DAS only.** Polygon is sanctioned here **only** for surfacing
    candidates the trader then confirms in DAS before any order. Polygon/Massive-sourced recency
    stays **PRIVATE to the dashboard** — never flows into any auto-published channel (Slack
    `#watchlist` / Notion).
  - **Setups reuse existing strategy signal logic, no new indicators:** `bella_fade`,
    `offside_scalp`, `microstructure_momentum` (`generate_signals(df)->List[Signal]`) plus a new
    **ORB (opening-range break)** and **VWAP-reclaim** trigger codified in the scanner. Take the
    **last entry `Signal` of the session** per symbol as the setup; map `Signal.to_dict()` →
    recommendation dict.
  - **Bounded universe (no full-universe scan):** seed candidates from a small in-play list
    (gappers / rel-vol, ~20-40 tickers) — respects cost speed-limit + anti-overtrading.
  - **Ranking = edge proxy, not raw %/volume:** `edge_score = R:R × signal.confidence`, where
    `R:R = (target-entry)/(entry-stop)`. **Gates BEFORE ranking:** min price, min rel/dollar
    volume, R:R ≥ 1.5, and a **cost speed-limit** (expected move must clear modeled
    spread+slippage by a wide margin, same slippage model as `backtest_intraday.py`). Trend
    setups (gap-and-go, ORB, VWAP reclaim) weighted above counter-trend fades. Dedup by ticker
    (keep highest edge). **Hard-cap 5** (Bellafiore "1 entry, 1 exit"). Each setup stamped with a
    `valid_until` / kill-time (time-of-day decay; suppress last 30 min).

  **Contract — extended, not replaced (backward-compatible):** existing keys retained
  (`status`, `timestamp`, `screen_type`, `total_stocks_screened`, `profiles_run`,
  `recommendations[]`); existing rec keys retained (`ticker`, `entry_price_range`,
  `target_price`, `stop_loss`, `risk_level`, `confidence_score`, `reasoning`,
  `_profile_source`, `_theme`). New **additive** rec fields: `setup_type`, `trigger_detail`,
  `entry`, `stop`, `target`, `rr` (R:R), `edge_score`, `rank`, `valid_until`, `data_recency`.
  New **additive** top-level fields: `data_source`, `session_date`, `last_bar_ts`,
  `data_recency`. The existing ProfileManager merge/dedup is preserved; intraday setups are
  merged into the same list tagged `_theme="intraday_setup"` then re-sorted by `edge_score`.
  `status="success"` only when real setups are computed from real bars; `status="no_data"` only
  when no qualifying setup exists — never because the pipeline is unwired.

  **LAST-MILE WIRING (#10, 2026-06-16) — scan→persist→read split, RTH worker.** The lazy
  in-process scan inside `get_recommendations` is **structurally insufficient** for the DELAYED
  tier: the thin dashboard image installs `falcon-core[advisor,postgresql]` (NO boto3), so its
  flat-file tier returns empty, and an HTTP handler is not a scheduler. The fresh DELAYED row
  therefore comes from a **worker that runs `intraday_scanner` on an RTH timer and persists via
  `persist_scan()`**; the dashboard stays THIN and surfaces those rows for free through its
  existing ProfileManager merge loop (`dashboard_server.py:538-561`).
  - **Write side (worker).** `persist_scan()` (`intraday_scanner.py:827-855`) needs BOTH
    `falcon_core.get_db_manager` (boto3 → flat-file tier) AND
    `falcon_screener.profile_manager.ProfileManager` in one process. No prior image had both, so
    the **falcon-screener image is thickened**: install `falcon-core[data-sync]` (boto3+pyarrow)
    + `falcon-trader` (provides `intraday_scanner`, pulls `falcon_screener`). It runs
    `python3 -m falcon_trader.intraday_scanner`, which scans (Polygon DELAYED during RTH, else
    flat-file STALE/DEGRADED) and writes one `profile_runs` row under the **"Intraday Scanner"**
    profile (`_ensure_intraday_profile`, run_type `intraday_scan`).
  - **Schedule.** A new oneshot quadlet `intraday-scan.container` + `intraday-scan.timer` fires
    every ~15 min Mon-Fri ~09:45-16:00 ET (mirrors `data-sync-minute`), so `last_bar_ts` stays
    within ~20 min through the session — the suggestion is fresh, not frozen.
  - **Read side UNCHANGED.** The dashboard image is NOT modified (no boto3 added). Its lazy
    in-process scan remains a best-effort Polygon-REST fallback only; the honest-recency recovery
    branch (`dashboard_server.py:615-623`) fires only when `in_process_ok` is False and never
    re-labels a worker-sourced DELAYED row.

  **Deferred (NOT this run):** the **DAS loopback live feed** (LIVE tier reserved hook); folding
  the swing-screener RTH quadlets (`falcon-screener/deploy/quadlet/*`, SQLite/wrong cadence) into
  the PG/falcon.env wiring.

- **Squawk / News feed (#8) — IN PROGRESS (MVP scoped).** A Benzinga-style **SQUAWK**: a SHORT,
  ranked, de-duped, reverse-chronological stream of REAL breaking headlines for the in-play
  universe, surfaced on the dashboard so the trader sees catalysts the moment they hit. This is a
  **PRIVATE dashboard surface only** — never auto-published to Slack/Notion/public (Polygon/Massive
  news is an individual subscription).

  **Source — Polygon REST news (verified live, this is its ONLY sanctioned REST use):**
  `GET https://api.polygon.io/v2/reference/news?ticker=&order=desc&sort=published_utc&limit=&apiKey=$POLYGON_API_KEY`
  (fallback `MASSIVE_API_KEY`). Each `results[]` carries `id`, `publisher.name`, `title`,
  `published_utc` (ISO8601 Z), `article_url`, `tickers[]`, `description`, `keywords[]`, and
  `insights[]` (per-ticker `{ticker, sentiment, sentiment_reasoning}` — present on this key).
  **Polygon flat files have NO news prefix (confirmed via `s3 ls`)** — REST news is the only
  source. The **DAS-over-Polygon hard rule applies to BARS/QUOTES, not news**: news is an
  event/text feed (publisher publish-time, not a 15-min market-data delay), so the REST news
  endpoint does NOT violate `feedback_das_over_polygon`. ~~Do NOT add Polygon REST to
  `intraday_scanner.py`~~ — **SUPERSEDED by the Freshness-tier refinement (#9):** the scanner MAY
  use Polygon REST as the DELAYED tier above flat files, labeled honestly. Squawk (news) and the
  scanner (bars) remain separate code paths but now share the same Polygon-as-discovery rationale.

  **MVP shipped THIS run (honest scope):**
  - New module `falcon-trader/src/falcon_trader/squawk_feed.py` with `fetch_squawk(universe,
    lookback_min)`: per-ticker/batched GET, **dedup by `results[].id` plus a normalized
    (ticker + event-type + time-bucket) fingerprint** (collapse multi-publisher duplicates to one
    row with a source count), classify catalyst type, score, hard-cap. **Never raises** (mirror
    the scanner's never-raise contract) and **120s TTL cache** (mirror `intraday_scanner._CACHE`,
    `_CACHE_TTL_SECONDS`). Reads the key via `os.getenv("POLYGON_API_KEY")` / fallback
    `MASSIVE_API_KEY` — **never hardcoded**.
  - **Universe = shared context, NOT a forked list.** Default to `FALCON_DASHBOARD_SYMBOLS`
    (today's real in-play set), **unioned** with tickers from the latest scan / merged
    recommendations; fall back to `intraday_scanner.DEFAULT_UNIVERSE` /
    `FALCON_INTRADAY_UNIVERSE` only if both empty. Bounded set (Bellafiore: not a firehose) to
    cap Polygon calls. An "all market" toggle is allowed but **default is in-play**.
  - **Ranking — one score, one stream (mirrors the scanner's `edge_score` idiom):**
    `squawk_score = catalyst_tier_weight × publisher_quality × recency_decay × in_play_bonus`,
    sort desc, **hard-cap <=30 rendered items**, drop below a score floor (not paginate),
    **dedup before ranking**. Publisher quality tiers (primary newswire/exchange > aggregator >
    PR wire > sponsored=drop) drive both score and NOISE suppression. Catalyst-type chip:
    `EARNINGS / GUIDANCE / M&A / FDA / HALT / RATING / OFFERING / SEC-8K / EXEC / MACRO / NOISE`
    from keyword+publisher rules over `title`. TIER-1 (HALT/M&A/FDA/GUIDANCE) on an in-play
    ticker sorts above all TIER-2/3 and renders in the HIGH visual tier.
  - **Freshness is the most important field** — render `published_utc` (parsed with `ZoneInfo`,
    sorted strictly reverse-chronological by publish-time, NEVER fetch-time) as **age** ("0m12s
    ago" / NEW badge), color-decaying after 5 min, graying past 30 min; **auto-expire/mute items
    older than the session window (~60-90 min)** unless re-confirmed. Same honest-staleness
    contract as the scanner: every item carries `published_utc` + a `data_recency` /
    source-provenance label (`"LIVE (Polygon news, publisher-time)"`). **A headline with no
    visible freshness is a FAILING condition** (stale rehash mistaken for breaking).
  - **Sentiment per ticker, not gospel:** map `insights[]` to an up/down/flat arrow by selecting
    the entry where `insight.ticker == queried symbol` (NOT `insights[0]`); show "no sentiment"
    when no matching insight exists. Marked model-derived; it orients, it does not decide.
  - **Endpoint — additive, separate:** new `@app.route('/api/squawk')` in `dashboard_server.py`
    inserted near L654 (after `get_recommendations_history`), returning
    `{status, items:[{id, ticker(s), title (verbatim), published_utc, age, publisher,
    article_url, catalyst_type, sentiment, sentiment_reasoning, squawk_score, tier, in_play,
    source_count, source:'polygon_news'}], fetched_at, data_recency}` following the L637-649
    envelope, with a `no_data` branch like L626-635. **Never merged into `/api/recommendations`**
    (headlines must never be mistaken for ranked trade setups). Headline text is **verbatim from
    `results[].title`** — never paraphrased in the headline slot.
  - **Frontend — additive panel:** a new `<div class="recommendations" id="squawk">` reusing the
    existing `.recommendations` CSS (L72-78) and the **data_recency banner** pattern (L235-250);
    `loadSquawk()` mirrors `loadRecommendations()` (fetch `/api/squawk`, render a headline
    **list/cards** — ticker badge + verbatim headline linking to `article_url` (new tab) +
    catalyst chip + sentiment pill + relative age — NOT a setup table) with its own
    `setInterval(loadSquawk, 60000)` (faster than the 5-min recs refresh). Stable `id` drives the
    NEW badge / new-vs-seen diff.
  - **Resilience:** on 429/error, degrade gracefully (last-good cached stream + clear status),
    back off, never crash the dashboard.

  **Deferred (NOT this run, strong follow-ups):** TTS/**audio squawk** (one announcement per
  `results[].id`, gated by stable id), a websocket/continuous-push delta-only stream (MVP uses
  client polling), and an AI "why it matters" tag (`insights[].sentiment_reasoning` is the cheap
  stand-in for now).
