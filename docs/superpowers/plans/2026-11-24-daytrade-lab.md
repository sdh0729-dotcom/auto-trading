# Day-trade Lab (Plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect Toss 1-minute bars for the daily universe and screen / validate the spec's day-trade rule families (same-day exit, regular session) with Toss costs, the KR VI rule and the existing integrity machinery (periods, experiment log, PBO, seal refusal).

**Architecture:** Minute bars live per symbol per month under `data/{market}/1m/`, trimmed to the regular session at load. A new intraday executor turns per-symbol rule candidates into trades (`TRADE_COLUMNS`, with timestamps in `entry_date` / `exit_date`); the existing `CostModel.net_returns` and `trade_metrics` score them unchanged. The pipeline gains a `daytrade` pocket whose periods split the minute-data trading days; validate uses a new intraday portfolio (all positions flat each night). `verify --final` for day trading (Nautilus on minute bars) is **Plan 2b**, not this plan: no seal is opened here.

**Families in this plan:** spec 7.5 #0 ORB stocks-in-play (paper + wide stop), #1 ORB general, #4 gap go / fade, #5 KR opening overreaction, #6 intraday momentum, #8 close momentum / reversal. Deferred until the first results are in: #2 VWAP reversion, #3 volume-surge momentum, #9 intraday seasonality, #7 Kronos 1m (needs the probe first, like swing).

**Tech Stack:** Python 3.12 (uv), pandas, numpy, pyarrow, typer, httpx; existing `autotrade` modules.

**Spec:** `docs/superpowers/specs/2026-11-19-auto-trading-design.md` (sections 4.1, 4.2, 4.3, 5, 7.2, 7.3, 7.5, 7.6, 7.8; decisions of 2026-11-24 in 7.9).

## Global Constraints

- `uv` only; `uv run pytest -q` must pass with no warnings before every commit; commits local, push only on the user's OK.
- Never print, log or commit `.env` / tokens; `data/` (Toss prices) is never committed or published.
- No order API calls; the LLM is not in any path here.
- Do not hand-edit `research/seal-registry.json`, `research/periods-*.json`, `research/experiments.jsonl`. Screen / validate never read a sealed period (existing guards apply to the daytrade pocket too).
- 1-minute timestamps are **bar end** times; a signal may use a bar only after it ends; fills happen at a later bar.
- Regular session only: KR bars ending in (09:00, 15:30] Asia/Seoul; US bars ending in (09:30, 16:00] America/New_York (a half-day ends at its last bar).
- KR: no new entry after 15:20; anything open is closed at the closing-auction price (close of the day's last regular bar). US: closed at the last regular bar's close; the real (50,000 KRW, fractional) scenario closes at the last bar ending <= 1 h before the session's last bar.
- Minute history starts: kr 2022-11-23, us 2021-12-01 (Toss); 200 candles per call; MARKET_DATA_CHART limit 20/s.
- Cost precondition: a parameter set whose median round-trip cost ÷ median stop (cost R) exceeds **0.15** is not screened (logged as stage `precheck`).
- Risk (pocket `daytrade`, `config/pockets/daytrade.yaml`): risk_per_trade 0.01, max_position_weight 0.25, daily_loss_stop 0.03; same-day exit; long only.
- KR VI (spec 7.6): static ±10% of the reference price, dynamic 3% (KOSPI200 stocks) / 6% (others) of the previous trade; a triggered VI is a 2-minute single-price auction: orders in it fill at the open of the first bar after it.
- Slippage as in the market configs (KR 1 tick, US max(0.05%, $0.02)).
- Talk to the user in Korean; code, tests and commit messages in English.

## Review Focus

1. **Missing minutes:** illiquid symbols have no bar in minutes without trades — the executor must use the next existing bar for fills and the day's last existing regular bar for the session exit, never a fixed clock row. (Task 3: `test_fills_skip_missing_minutes`, `test_session_exit_uses_last_existing_bar`.)
2. **US daylight saving and half-days:** session bounds come from local ET time per day, and a 13:00 close day exits at its last bar. (Task 1: `test_us_session_follows_dst`; Task 3: `test_half_day_exits_at_last_bar`.)
3. **Stop gapped through inside a bar or across the VI auction:** a stop is filled at min(stop, bar open) when the bar opens below it. (Task 3: `test_stop_gap_fills_at_open`; Task 4: `test_stop_inside_vi_fills_after_auction`.)
4. **Collect interrupted (sleep, IP change) mid-symbol:** month files stay whole, resume continues backwards from the earliest stored bar without a hole, and a changed adjustment (split) refetches the symbol. (Task 2: `test_resume_after_abort_has_no_gap`, `test_split_refetches_minutes`.)
5. **The daily job and the minute collector share one Toss client id:** the minute collector pauses between symbols while `data/logs/.lock` exists. (Task 2: `test_pauses_while_daily_job_runs`.)

---

### Task 1: Minute store and session trimming

**Files:**
- Create: `src/autotrade/data/minute_store.py`
- Modify: `config/markets/kr.yaml`, `config/markets/us.yaml`, `src/autotrade/config.py` (MarketConfig: `timezone: str`, `regular_open: str`, `regular_close: str`, `minute_start: str`)
- Test: `tests/test_minute_store.py`

**Interfaces:**
- Produces:
  - `MINUTE_COLUMNS = ["open", "high", "low", "close", "volume"]`; frames indexed by tz-aware `DatetimeIndex` named `ts` in the market timezone (bar end).
  - `minute_dir(market, symbol) -> Path` = `data_root()/market/"1m"/symbol`
  - `save_minutes(market, symbol, df) -> None` — merges into month files `YYYY-MM.parquet` (atomic per file, `write_parquet_atomic`), later rows win on duplicate `ts`.
  - `load_minutes(market, symbol, start: date, end: date, regular_only=True) -> pd.DataFrame` (empty frame with the columns when nothing stored)
  - `regular_session(df, market_cfg) -> pd.DataFrame` — keeps bars whose local end time is in (regular_open, regular_close].
  - `minute_bounds(market, symbol) -> tuple[pd.Timestamp, pd.Timestamp] | None` — earliest / latest stored `ts`.
  - Config values: kr `timezone: Asia/Seoul`, `regular_open: "09:00"`, `regular_close: "15:30"`, `minute_start: 2022-11-23`; us `America/New_York`, `"09:30"`, `"16:00"`, `2021-12-01`.

- [ ] **Step 1: Write the failing tests**
  - `test_month_partition_round_trip`: bars across 2024-01-31 / 2024-02-01 → two files `2024-01.parquet`, `2024-02.parquet`; `load_minutes` returns all rows sorted, tz preserved.
  - `test_save_merges_and_dedupes`: saving an overlapping frame twice keeps one row per `ts`, values from the second save.
  - `test_kr_trim_drops_nxt_hours`: bars ending 08:01, 08:59, 09:00, 09:01, 15:20, 15:30, 15:31, 20:00 KST → kept: 09:01, 15:20, 15:30.
  - `test_us_session_follows_dst`: bars ending 09:31 and 16:00 ET on 2024-03-08 (EST) and 2024-03-11 (EDT) are kept, 09:30 and 16:01 dropped, on both dates (input given in UTC).
  - `test_bounds_and_empty`: `minute_bounds` is None and `load_minutes` empty for an unknown symbol.

- [ ] **Step 2: Run** `uv run pytest tests/test_minute_store.py -v` — FAIL (module missing).
- [ ] **Step 3: Implement** the functions above; trimming compares `ts.tz_convert(tz).time()`.
- [ ] **Step 4: Run** the test file and the full suite — PASS, no warnings.
- [ ] **Step 5: Commit** `feat(data): month-partitioned 1-minute store with regular-session trim`

### Task 2: Minute collector and `lab collect-minute`

**Files:**
- Create: `src/autotrade/data/collect_minute.py`
- Modify: `src/autotrade/pipeline.py` (`collect_minute`), `src/autotrade/cli.py` (`collect-minute`)
- Test: `tests/test_collect_minute.py`

**Interfaces:**
- Consumes: `TossClient.get_candles(symbol, "1m", before=..., count=200)` → `CandlePage(df, next_before)` (newest first pages; `timestamp` column tz-aware), Task 1 store.
- Produces:
  - `collect_minutes(client, market, symbol, since: date, today: pd.Timestamp) -> MinuteReport(added: int, complete: bool)` — (1) forward part: pages from now back to the latest stored bar; if the overlapping bar differs from the stored one (adjustment changed) the symbol's minute files are deleted and refetched from scratch; (2) backward part: from the earliest stored bar back to `since` or an empty page; saves after every page (only whole pages, atomic), never stores bars of `today`'s session; writes `data/{market}/1m/{symbol}/_complete` when the backward part reached `since` / empty.
  - `minute_symbols(market, start: date, end: date, top_n: int) -> list[str]` — union of symbols in the daily **paper** universe (`eligible_mask`, top `top_n`) on any day in [start, end] plus `always_include`, ordered by number of universe days, descending.
  - `pipeline.collect_minute(market, limit: int | None) -> dict` (counts: symbols, added, complete, not_found, failed); reuses the daily collect's `stock-not-found` skip and `MAX_CONSECUTIVE_FAILURES` abort; before each symbol waits (sleep 30 s loop) while `data/logs/.lock` exists.
  - CLI: `lab collect-minute --market kr|us [--limit N]` prints the counts.

- [ ] **Step 1: Write the failing tests** (fake client serving minute pages, like `tests/test_data.py::FakeClient`)
  - `test_backward_paging_stops_at_since`: 3 days of bars, `since` = day 2 → only days 2-3 stored, `_complete` written.
  - `test_incremental_fetch_stops_at_stored_last_bar`: second run after one new day requests only the new pages (count pages).
  - `test_resume_after_abort_has_no_gap`: client raises after 2 pages; second run fills back to `since`; stored `ts` equal the full fake series.
  - `test_split_refetches_minutes`: fake rescales all prices by 1/2 between runs → stored series equals the new scale everywhere.
  - `test_today_session_not_stored`: bars of `today` are absent.
  - `test_minute_symbols_union_ordered_by_days`: synthetic daily panel where A is in the universe 10 days, B 3 days → `["A", "B", ...]`, always_include present.
  - `test_pauses_while_daily_job_runs`: lock dir present → sleep called (monkeypatched) until it disappears; then collection proceeds.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** file + full suite — PASS.
- [ ] **Step 5: Commit** `feat(data): resumable 1-minute collector for the daily universe`
- [ ] **Step 6 (operational, after commit):** start `uv run lab collect-minute --market kr` in the background (log to the scratchpad); then `--market us`. Expect days of runtime; the Mac must stay awake. Report progress to the user in Korean.

### Task 3: Intraday executor

**Files:**
- Create: `src/autotrade/screen/intraday.py`
- Modify: `src/autotrade/types.py` (`EXIT_REASONS` += `"target"`, `"session_end"`)
- Test: `tests/test_intraday.py`

**Interfaces:**
- Produces:
  - `CANDIDATE_COLUMNS = ["day", "signal_ts", "stop_pct", "target_pct", "exit_ts", "confidence"]` — one long entry per row: signal from the bar ending `signal_ts`; `target_pct` NaN = none; `exit_ts` NaT = session end; `day` is the local session date.
  - `execute(symbol: str, bars: pd.DataFrame, cands: pd.DataFrame, rules: SessionRules) -> pd.DataFrame` — `TRADE_COLUMNS` rows (`entry_date` / `exit_date` hold timestamps).
  - `SessionRules(last_entry: str | None, exit_lead_minutes: int, vi: VIConfig | None)` — kr: `last_entry="15:20"`; us: None; `exit_lead_minutes` 0 (paper) / 60 (us real).
  - Semantics: entry at the open of the first bar after `signal_ts` (same day, before `last_entry`); stop price = entry × (1 − stop_pct), target = entry × (1 + target_pct); from the entry bar on, per bar: if open ≤ stop → exit at open (`stop`); elif low ≤ stop → exit at stop (`stop`); elif target and open ≥ target → exit at open (`target`); elif target and high ≥ target → exit at target (`target`) — stop is checked before target in the same bar (conservative); at `exit_ts` → exit at that bar's close (`signal`); at the day's last regular bar (minus `exit_lead_minutes`) → its close (`session_end`). One open trade per symbol at a time; later candidates while in a trade are dropped.

- [ ] **Step 1: Write the failing tests** (hand-built bars)
  - `test_entry_is_next_bar_open_never_the_signal_bar`
  - `test_fills_skip_missing_minutes`: no bar for the minute after the signal → entry at the next existing bar's open.
  - `test_stop_gap_fills_at_open`
  - `test_stop_checked_before_target_in_one_bar`
  - `test_session_exit_uses_last_existing_bar` (no 15:30 bar → last bar's close)
  - `test_half_day_exits_at_last_bar` (US day ending 13:00 ET)
  - `test_no_entry_after_last_entry_kr` (signal 15:19 → entry bar 15:20 allowed; signal 15:20 → no trade)
  - `test_us_real_exits_one_hour_early` (`exit_lead_minutes=60`)
  - `test_overlapping_candidates_dropped`
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (loop per day; numpy arrays per day are fine). **Step 4: Run** — PASS. **Step 5: Commit** `feat(screen): intraday executor with same-day exits`

### Task 4: KR VI emulation

**Files:**
- Modify: `src/autotrade/screen/intraday.py` (`VIConfig`, `vi_halts`), `config/markets/kr.yaml` (`vi: {static_pct: 0.10, dynamic_pct_kospi200: 0.03, dynamic_pct_other: 0.06, halt_minutes: 2, etf_exempt: true}`)
- Test: `tests/test_intraday.py`

**Interfaces:**
- Produces: `vi_halts(bars_day: pd.DataFrame, prev_close: float, dynamic_pct: float, cfg: VIConfig) -> pd.Series[bool]` (index = bar ts; True for bars inside a 2-minute auction). A bar triggers when its high/low crosses ±static_pct of the reference (prev close; the reference resets to the auction's end price after a static VI) or moves ≥ dynamic_pct from the previous bar's close; the next `halt_minutes` minutes are halted. `execute` defers any fill (entry, stop, target) that falls on a halted bar to the open of the first non-halted bar. KOSPI200 membership is unknown → `dynamic_pct_other` for every stock (limitation line in the report); ETFs exempt.
- [ ] **Step 1: Tests** `test_static_vi_halts_two_minutes`, `test_dynamic_vi_uses_previous_close`, `test_stop_inside_vi_fills_after_auction`, `test_etf_never_halts`.
- [ ] **Steps 2-4** fail → implement → pass. **Step 5: Commit** `feat(screen): KR VI auction delay in the intraday executor`

### Task 5: Day-trade pocket, periods and screen

**Files:**
- Create: `src/autotrade/rules/daytrade/__init__.py`, `src/autotrade/rules/daytrade/base.py`
- Modify: `src/autotrade/rules/registry.py` (`DAYTRADE_FAMILIES`), `src/autotrade/pipeline.py` (`FAMILIES["daytrade"]`; `_setup` / `_new_periods` use the minute calendar for `daytrade`; `screen` dispatches to `screen_intraday`), `src/autotrade/screen/run.py` (`screen_intraday`), `config/pockets/daytrade.yaml`
- Test: `tests/test_daytrade_pipeline.py`

**Interfaces:**
- Produces:
  - `IntradayRule` (base): `family`, `PARAM_KEYS`, `grid()`, `rule_id` (same hashing as swing `BaseRule`), `select(day_features: pd.DataFrame, daily: Panel, eligible: pd.DataFrame) -> pd.Index` (MultiIndex day × symbol to trade; default: every eligible symbol-day), `day_features(symbol, bars) -> pd.DataFrame` (index day; default empty), `candidates(symbol, bars, days, daily: Panel) -> pd.DataFrame[CANDIDATE_COLUMNS]`.
  - `minute_calendar(market) -> pd.DatetimeIndex` — trading days with regular bars of the market benchmark (069500 / SPY) from `minute_start`.
  - `screen_intraday(family, market, symbols, start, end, daily, eligible, cost_model, rules, log, pocket, sealed_start) -> pd.DataFrame` (same columns as `screen_family`); per grid point: cost R precheck first (median `cost_model.round_trip_cost_pct` at entry ÷ median `stop_pct` over its candidates > 0.15 → log stage `precheck` with `{"cost_r": x}` and skip), else execute, `net_returns`, `trade_metrics`, log stage `screen`.
  - `lab screen --pocket daytrade --market kr|us [--family ...]` works; periods file `research/periods-daytrade-<market>.json` from the minute calendar, split 0.6/0.2/0.2.
- [ ] **Step 1: Tests** (synthetic minute store via Task 1 + a toy rule registered in a monkeypatched `DAYTRADE_FAMILIES`): `test_daytrade_periods_use_the_minute_calendar`, `test_screen_never_loads_bars_after_train_end`, `test_cost_r_precheck_logs_and_skips`, `test_screen_logs_one_trial_per_grid_point`.
- [ ] **Steps 2-4.** **Step 5: Commit** `feat(pipeline): daytrade pocket with minute-calendar periods and intraday screen`

### Task 6: ORB families (spec 7.5 #0, #1)

**Files:** Create `src/autotrade/rules/daytrade/orb.py`; register; Test `tests/test_daytrade_rules.py`
**Interfaces:**
- `OrbStocksInPlay` family `orb_sip`: `day_features` = first-5-minute volume, first-5-minute direction (close of 5th bar vs day open), ATR14 from daily, 14-day mean first-5-minute volume; `select` = eligible, price > $5 (us) / tick-cost filter (kr), 14-day avg daily volume ≥ 1,000,000 (us), ATR > $0.50 (us) / > 1% of price (kr), relative first-5-min volume ≥ 1.0, top 20 by relative volume per day; `candidates`: only up days (first 5-minute bar up), signal when a later bar's high crosses the opening-range high → stop_pct = `stop_atr` × ATR / entry-ish (range high); exit at session end. Grid: `stop_atr ∈ {0.10, 0.5, 0.75, 1.0}` (0.10 = paper version a; others = wide-stop variant b), `or_minutes = 5`.
- `Orb` family `orb`: general opening range on every eligible symbol: `or_minutes ∈ {5, 15, 30}`, `stop ∈ {"or_low", 0.5 ATR, 1.0 ATR}`, `target_r ∈ {nan, 2.0}`.
- [ ] Tests: `test_orb_entry_only_after_the_range_ends`, `test_orb_sip_picks_top_relative_volume`, `test_orb_paper_stop_is_ten_percent_atr`, `test_orb_signal_uses_completed_bars_only` (changing bars after `signal_ts` never changes earlier candidates).
- [ ] Fail → implement → pass → commit `feat(rules): opening-range breakout families`

### Task 7: Momentum families (spec 7.5 #6, #8)

**Files:** Create `src/autotrade/rules/daytrade/momentum.py`; register; tests in `tests/test_daytrade_rules.py`
**Interfaces:**
- `IntradayMomentum` (`intraday_mom`): first 30 minutes' return (prev close → bar ending open+30 min) > `threshold` → enter for the last 30 minutes (kr: signal at the bar ending 14:50, exit 15:20 bar close via `exit_ts`; us: signal at the bar ending 15:30 ET, exit session end). Universe: index ETFs (`always_include`) + eligible ETFs. Grid `threshold ∈ {0.0, 0.002, 0.005}`.
- `CloseMomentum` (`close_mom`, Baltussen et al.): return from prev close to the signal bar (kr 14:50, us 15:30) sign; `direction ∈ {"momentum", "reversal"}` (long when positive / when negative), index ETFs only; exits as above. Grid `direction × min_move ∈ {0.0, 0.005}`.
- [ ] Tests: `test_intraday_mom_signal_bar_and_exit_kr`, `test_close_mom_reversal_goes_long_after_a_down_day`, `test_momentum_families_trade_index_etfs_only`.
- [ ] Fail → implement → pass → commit `feat(rules): intraday and close momentum families`

### Task 8: Gap and KR opening-overreaction families (spec 7.5 #4, #5)

**Files:** Create `src/autotrade/rules/daytrade/gap.py`; register; tests in `tests/test_daytrade_rules.py`
**Interfaces:**
- `GapGo` (`gap_go`): gap up ≥ `gap` (open vs prev close) and the first `confirm` minutes close above the open → long at the next bar, stop = day low so far (as pct), exit session end. Grid `gap ∈ {0.02, 0.04}`, `confirm ∈ {5, 15}`.
- `GapFade` (`gap_fade`, long only): gap down ≤ −`gap` → long after the first `confirm` minutes if price is above the first bar's low; stop = `stop_pct`; target = fill of half the gap (target_pct = gap/2). Grid `gap ∈ {0.02, 0.04}`, `confirm ∈ {5, 15}`, `stop_pct ∈ {0.01, 0.02}`.
- `OpenReversal` (`kr_open_rev`, kr only): first 30-minute drop ≤ −`drop` with volume ≥ 2 × its 14-day average first-30-minute volume → long at the next bar, exit `hold` minutes later or session end. Grid `drop ∈ {0.03, 0.05}`, `hold ∈ {60, 120}`, `stop_pct ∈ {0.02, 0.03}`.
- [ ] Tests: `test_gap_go_needs_confirmation`, `test_gap_fade_target_is_half_gap`, `test_kr_open_rev_skips_us`.
- [ ] Fail → implement → pass → commit `feat(rules): gap and opening-overreaction families`

### Task 9: Intraday portfolio and daytrade validate

**Files:**
- Create: `src/autotrade/sizing/intraday.py`
- Modify: `src/autotrade/pipeline.py` (`validate` dispatch for `daytrade`), `src/autotrade/cli.py` (daytrade validate columns), `config/pockets/daytrade.yaml` (`benchmark_gate: false`), `src/autotrade/config.py` (`PocketConfig.benchmark_gate: bool = True`)
- Test: `tests/test_intraday_portfolio.py`, `tests/test_daytrade_pipeline.py`

**Interfaces:**
- `run_intraday_portfolio(trades, cost_model, pocket, capital, currency_per_krw, fractional) -> PortfolioResult` — per day: sizing equity = previous day's close equity; events in time order (exits before entries at the same `ts`); qty = min(equity × risk_per_trade × confidence / (entry × stop_pct), equity × max_position_weight / entry), floored unless fractional; skip reasons `daily_loss_stop` (realized day P&L ≤ −daily_loss_stop × day-start equity blocks later entries that day), `too_small`, `cash`; equity marked once per day after all exits (flat book). Reuses `CostModel.fill_cash`.
- `validate("daytrade", market, top)`: same pool / PBO / segments / experiment logging as swing, portfolio = `run_intraday_portfolio`; adds `valid_mean_daily_return`, `valid_positive_day_share`, `valid_worst_day` to metrics and the printed table; recommendation = valid net pnl > 0 (no benchmark gate when `benchmark_gate` is false).
- [ ] Tests: `test_daily_loss_stop_blocks_later_entries_same_day_only`, `test_sizing_uses_previous_day_equity`, `test_exit_before_entry_at_same_ts`, `test_validate_daytrade_reports_daily_metrics`, `test_daytrade_has_no_benchmark_gate`.
- [ ] Fail → implement → pass → commit `feat(validate): intraday portfolio and daytrade validate`

### Task 10: Limitations, journal, real run

**Files:** Modify `src/autotrade/report/limitations.py` (daytrade lines: KR minute bars are KRX+NXT combined; KOSPI200 membership unknown → 6% dynamic VI for all stocks; ETF VI exempt; US bars not NBBO), `README.md`, `research/journal/<date>.md`
- [ ] Test: `test_daytrade_limitations_listed` (report for pocket daytrade contains the four lines).
- [ ] After the minute collection finishes: `lab screen --pocket daytrade --market kr`, then `us`; `lab validate --pocket daytrade --market kr|us`; write the journal (numbers, cost-R exclusions, daily metrics); report to the user in Korean. **Do not run `verify --final`** (Plan 2b; user confirmation required anyway).
- [ ] Commit `docs: daytrade limitations and first daytrade results`
