# Disclosure Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect KR (DART) and US (SEC EDGAR) filings since 2015, classify them by a config table into dated good/bad events, and add event-driven swing rule families (`disclosure`, `<family>_negf`) to the existing lab without breaking its period/seal/trial discipline.

**Architecture:** A `disclosure/` package holds the two HTTP clients and the resumable collectors that write raw filings (+ size details) under `data/{market}/events/`. `load_events` classifies raw filings with `config/events/{market}.yaml` at load time and maps each event to its **signal date** (the trading day whose next open is the first tradable open). The events ride on `Panel.events` (long form) and are sliced with prices, so screen / validate / verify / forward see exactly the events their price window sees.

**Tech Stack:** Python 3.12 (uv), httpx, pandas/parquet, typer, pytest (httpx.MockTransport for HTTP fakes).

**Spec:** `docs/superpowers/specs/2026-11-24-disclosure-lab-design.md` (parent: `docs/superpowers/specs/2026-11-19-auto-trading-design.md`)

## Global Constraints
- `uv` only; `uv run pytest -q` passes with **no warnings** before every commit. Commit locally; never push.
- Never print, log, commit or put in an exception message the DART key (`DART_API_KEY`) or the SEC contact (`SEC_USER_AGENT`) from `.env`. The DART key travels in the query string → every error path must scrub URLs.
- Tests never touch the network: HTTP goes through an injectable `httpx.Client` (tests use `httpx.MockTransport`).
- `data/` is never committed. Never hand-edit `research/seal-registry.json`, `research/periods-*.json`, `research/experiments.jsonl`. Never run `lab verify --final`.
- No LLM anywhere in this plan. No order API.
- KR availability: filing date D (no time) → signal date = last trading day ≤ D (entry at the next open). US: accepted before 09:30 New York time on trading day D → signal date = trading day before D; otherwise last trading day ≤ D (New York date).
- Event families use `entry_at="next_open"` only. Families with `entry_at="close"` (overnight, tom) never get events.
- Corrections (`[기재정정]`, `[첨부정정]`, `[첨부추가]`, `[변경등록]` … any leading `[...]` tag) are not new events. Withdrawals are recorded with direction 0 and never delete the original event.
- Same symbol + same event within `repeat_days: 20` trading days → only the first is an event signal.
- Size unknown → a rule with `sized=True` does **not** enter. Size minimums (fixed before any return is seen): KR supply contract revenue ratio ≥ 0.10, KR buyback amount ≥ 1.0 × 20-day average traded value, KR rights offering dilution ≥ 0.10; US insider buy amount ≥ 100,000 USD (designer's call, spec §10-1 #2 adds US insider size).
- Bundles per market (spec §10-1 #2): kr `bundle_a`=buyback, `bundle_b`=supply_contract; us `bundle_a`=8k_101, `bundle_b`=insider_buy; `positive` = every direction +1 event of the market.
- `disclosure` grid = bundle {bundle_a, bundle_b, positive} × hold {5, 10, 20} × ATR stop k {2, 3} × confirm {none, ma20, high20} × sized {True, False} = 108 (US `bundle_a` duplicates its sized axis — accepted, it only makes DSR stricter).
- `_negf` wraps exactly breakout, ma_trend, momentum, reversal: no new entry while a direction −1 event has a signal date within the last 20 trading days (inclusive of t); a held position exits at the next open when a −1 event's signal date is t.
- Event rules live in the existing `swing` pocket and existing period files; every grid point is a logged trial. Seal cap (3 per pocket × market) and pass criteria unchanged.
- Event-table integrity: screen/validate rows of event families carry `metrics["events_table"]` = hash of the market's events config; validate, verify and forward refuse an event-family rule whose logged hash differs from the current one.
- Event families refuse to screen/validate while events are incomplete: collector running, any target symbol not `complete`, or delisted-target corp-code coverage < 0.80.
- ETFs have no company filings: event targets are `product == "stock"` symbols only. The KR ETF forward test (`research/forward-swing-kr.json`) must keep producing the same numbers.

## Review Focus
1. DART/SEC error paths (HTTP 4xx/5xx, timeouts, DART status `020` rate limit, `800` maintenance) leaking the key via `httpx` exception text or URL → Task 1/6 tests assert the key string never appears in any raised message.
2. A weekend/holiday KR filing and a US 08:00 / 12:00 / 17:00 New York acceptance landing on the right signal date, including a US early-close day → Task 4 tests.
3. A screen whose train window ends before an event's signal date must not see that event (and a sealed-period event never reaches screen/validate) → Task 4/5 tests on `Panel.slice` and `_panel`.
4. Existing families produce identical signals with and without `Panel.events` → Task 4 regression test over every swing family on the synthetic panel.
5. An interrupted collector (KeyboardInterrupt / rate-limit stop mid-symbol) resumes without duplicate filings and without marking the symbol complete → Task 2 test.

---

### Task 1: DART client and corp-code map

**Files:**
- Create: `src/autotrade/disclosure/__init__.py`, `src/autotrade/disclosure/dart.py`
- Test: `tests/test_dart.py`

**Interfaces:**
- Produces: `class DartError(Exception)` with `.status: str`, `.message: str` (message never contains the key or a URL).
  `class DartClient(key: str, http: httpx.Client | None = None, min_interval: float = 0.1)` with
  `corp_codes() -> pd.DataFrame` (columns `corp_code, stock_code, corp_name`; `stock_code` "" when absent),
  `list_filings(corp_code: str, start: date, end: date, kind: str) -> list[dict]` (all pages, `page_count=100`, `pblntf_ty=kind`; status `013` → `[]`),
  `detail(endpoint: str, corp_code: str, start: date, end: date) -> list[dict]` (structured 주요사항 APIs; `013` → `[]`),
  `document(rcept_no: str) -> bytes` (raw zip from `document.xml`).
  `make_dart_client_from_env() -> DartClient` (reads `DART_API_KEY`; raises `DartError("config", "DART_API_KEY missing in .env")`).
  Status `020` raises `DartRateLimited(DartError)`; any other non-`000`/`013` raises `DartError`.

- [ ] **Step 1: Write the failing tests** — `test_list_filings_pages_until_total` (MockTransport serves 2 pages, total_count 150 → 150 rows, second request has `page_no=2`), `test_no_data_status_is_empty` (`013` → `[]`), `test_rate_limit_raises_rate_limited` (`020` → `DartRateLimited`), `test_errors_never_contain_key` (key `"SECRETKEY123"`; HTTP 500, `httpx.ConnectTimeout`, status `900` each raise an error whose `str()` and `repr()` lack `"SECRETKEY123"` and `"crtfc_key"`), `test_corp_codes_parses_zip` (zip with a 3-entry CORPCODE.xml incl. one blank stock_code → 3 rows, blank → ""), `test_missing_key_message`.
- [ ] **Step 2: Run** `uv run pytest tests/test_dart.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `dart.py`. Base URL `https://opendart.fss.or.kr/api/`. Catch `httpx.HTTPError` and re-raise `DartError("http", type(e).__name__)` `from None`. Respect `min_interval` between calls (monotonic sleep).
- [ ] **Step 4: Run** `uv run pytest tests/test_dart.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(disclosure): DART client (paged list, details, documents, key-safe errors)`.

### Task 2: Event store and KR filing collector (`lab collect-events --market kr`)

**Files:**
- Create: `src/autotrade/disclosure/store.py`, `src/autotrade/disclosure/collect_kr.py`
- Modify: `src/autotrade/pipeline.py` (new `collect_events`), `src/autotrade/cli.py` (new command), `.env.example` (keep the empty `DART_API_KEY=` line already in the working tree; add `SEC_USER_AGENT=`)
- Test: `tests/test_disclosure_collect.py`

**Interfaces:**
- Consumes: Task 1 `DartClient`, `DartRateLimited`.
- Produces:
  - `events_root(market) -> Path` = `data_root()/market/"events"`; files: `filings.parquet` (raw rows), `sizes.parquet` (Task 3), `targets.json`, `state.json`, `.collect.lock`.
  - `load_filings(market) -> pd.DataFrame`, `append_filings(market, df)` (dedupe on `filing_id`; atomic write via temp + rename).
  - Raw KR filing columns: `filing_id` (= rcept_no), `symbol`, `corp_code`, `filed_date` (date), `title` (report_nm stripped), `kind` (B/I), `filer`, `remark` (rm).
  - `event_targets(market) -> list[str]`: `product == "stock"` symbols in the swing paper universe on any day of [swing_start, today] (reuse `eligible_mask` with `load_pocket("swing")`, like `minute_symbols`).
  - `collect_events_kr(client, symbols, start=date(2015,1,1), today) -> dict` with counts `{"symbols","filings","complete","unmapped","rate_limited"}`; per-symbol state `{"done_through": iso, "complete": bool}`; kinds fetched: `B` and `I`.
  - `targets.json`: `{"targets": n, "mapped": n, "delisted_targets": n, "delisted_mapped": n, "unmapped": [symbols]}`. Mapping: `stock_code` match in `corp_codes()`; a delisted target with no match is unmapped (no name guessing).
  - `pipeline.collect_events(market, limit: int | None = None) -> dict` under an flock on `events_root/.collect.lock` (same pattern as `_collect_lock`); `events_running(market) -> bool`.
  - CLI: `lab collect-events --market kr [--limit N]`.

- [ ] **Step 1: Write the failing tests** — fake client object (not HTTP): `test_collect_writes_filings_and_marks_complete`, `test_resume_skips_complete_and_continues_from_done_through` (second run requests only `start=done_through`), `test_rate_limit_stops_cleanly_keeps_state` (fake raises `DartRateLimited` on symbol 2 → symbol 1 complete, symbol 2 not complete, return `rate_limited=True`, no exception), `test_interrupt_mid_symbol_no_duplicates` (KeyboardInterrupt after first kind of symbol 1; rerun → `filings.parquet` has no duplicate `filing_id` and symbol 1 is complete only after rerun), `test_targets_exclude_etfs_and_record_coverage`, `test_second_collector_refused` (lock held → `LabError`), `test_cli_collect_events_runs` (cli_env + monkeypatched client factory).
- [ ] **Step 2: Run** `uv run pytest tests/test_disclosure_collect.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** A symbol is marked `complete` only after both kinds were fetched through `today`; filings are appended per symbol before the state write.
- [ ] **Step 4: Run** the test file, then `uv run pytest -q` — Expected: all PASS, no warnings.
- [ ] **Step 5: Commit** `feat(disclosure): resumable KR filing collector + lab collect-events`.

### Task 3: KR size details

**Files:**
- Create: `src/autotrade/disclosure/sizes_kr.py`, `config/events/kr.yaml` (the full first KR table in the Task 4 schema — Task 4 adds the loader and `us.yaml`)
- Modify: `src/autotrade/disclosure/collect_kr.py` (size pass after listing), `src/autotrade/disclosure/store.py`
- Test: `tests/test_sizes_kr.py`

**Interfaces:**
- Consumes: `DartClient.detail`, `DartClient.document`, `load_filings`.
- Produces: `sizes.parquet` columns `filing_id, measure, value` where `measure` ∈ `{"buyback_krw", "supply_revenue_ratio", "rights_dilution"}`; `value` NaN when parsing failed (still recorded so it is not refetched).
  `parse_supply_ratio(doc: bytes) -> float | None` (reads the "매출액 대비(%)" figure; returns a ratio, e.g. 12.5% → 0.125).
  `size_targets(filings) -> pd.DataFrame` picks filings whose title (after correction-tag strip) matches the three size events in `config/events/kr.yaml` (this task writes the file in the Task 4 schema and reads only its `size` entries via a small loader `size_patterns(market) -> dict[str, str]`).
  Structured endpoints: buyback and rights offering via `DartClient.detail` (the implementer confirms the exact endpoint and field names against the DART guide pages and records them in a module docstring); supply contract via `document()` + `parse_supply_ratio`.

- [ ] **Step 1: Write the failing tests** — `test_parse_supply_ratio_reads_percent` (HTML fixture text containing `매출액대비(%)` → `12.5` → 0.125; tolerate spaces/linebreaks inside the label), `test_parse_supply_ratio_missing_returns_none`, `test_size_pass_records_nan_on_failure_and_never_refetches`, `test_size_pass_resumes_after_rate_limit`, `test_rights_dilution_is_new_over_existing_shares`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** in fetch order buyback → supply contract → rights offering (spec §10-1 #8).
- [ ] **Step 4: Run** test file + full suite — PASS.
- [ ] **Step 5: Commit** `feat(disclosure): KR event sizes (buyback, supply-contract ratio, rights dilution)`.

### Task 4: Classification, signal dates, `Panel.events`

**Files:**
- Create: `config/events/us.yaml`, `src/autotrade/disclosure/events.py`
- Modify: `src/autotrade/types.py` (`Panel.events`, `Panel.slice`), `src/autotrade/pipeline.py` (`_panel` attaches events; new `attach_events`), `src/autotrade/forward.py` (attach events to its panel)
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `load_filings`, `sizes.parquet`, `store.calendar(market)`.
- Produces:
  - Config schema (both markets):
    ```yaml
    repeat_days: 20
    negative_window: 20
    events:            # first match wins, matched against the correction-stripped title (kr) / form+items (us)
      - {name: buyback, direction: 1, match: "자기주식취득(신탁계약체결)?결정", size: buyback_krw, size_min: 1.0}
      - ...
    bundles: {bundle_a: [buyback], bundle_b: [supply_contract]}   # positive is implicit
    ```
    KR first table = spec §6.1 rows; US rows: `8k_101` (8-K items contain 1.01, +1), `8k_301`, `8k_402` (−1), `offering_424b` (0), `insider_buy` (+1, size `insider_usd`, size_min 100000).
  - `events_table_hash(market) -> str` (sha1 of the config file bytes, 12 hex chars).
  - `load_events(market, calendar: pd.DatetimeIndex) -> pd.DataFrame | None` (None when no filings stored), columns: `symbol, signal_date, event, direction, size, size_ok, bundle_a, bundle_b, positive, filing_id`. Rules: correction rows dropped; withdrawals kept as direction 0; repeats within `repeat_days` trading rows dropped (first kept); `size_ok` = size ≥ size_min, False when size NaN, True when the event has no size criterion; KR buyback size = amount ÷ 20-day mean of close×volume ending at the signal date.
  - `Panel.events: pd.DataFrame | None = None`; `Panel.slice` keeps rows with start ≤ signal_date ≤ end.
  - `attach_events(panel: Panel, market: str, end: date) -> Panel` (events with signal_date ≤ end; used by `_panel` and `forward.run`).
  - Helpers for rules: `event_frame(panel, column: str, min_size: bool) -> pd.DataFrame` (bool date × symbol on `panel.close`, True where an event of that bundle/direction has signal_date = t) and `recent_negative(panel, window: int) -> pd.DataFrame` (True where a −1 event's signal date lies in the last `window` rows incl. t).
  - CLI: `lab events --market kr [--unmatched N]` prints counts by event × year, coverage from `targets.json`, and the N most frequent unmatched titles (for finalizing the table with the user).

- [ ] **Step 1: Write the failing tests**
  - `test_kr_weekend_filing_maps_to_friday` (filed Sat 2024-03-09 → signal 2024-03-08), `test_kr_weekday_filing_maps_to_same_day`.
  - `test_us_acceptance_times` (NY 08:00 on trading day D → trading day before D; 12:00 → D; 17:00 → D; 08:00 on a holiday → last trading day ≤ holiday).
  - `test_corrections_are_not_events`, `test_withdrawal_kept_with_direction_zero`, `test_repeat_within_20_days_dropped`, `test_size_unknown_is_not_ok`, `test_event_without_size_rule_is_ok`.
  - `test_slice_drops_events_outside_window` and `test_panel_attaches_only_events_up_to_end` (an event dated after the train end is absent from the screen panel).
  - `test_existing_families_unchanged_with_events` (every `SWING_FAMILIES` member: identical `Signals` frames on `make_panel` with and without a random `events` frame).
  - `test_events_table_hash_changes_with_config`.
- [ ] **Step 2: Run** `uv run pytest tests/test_events.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** US acceptance times are converted with `zoneinfo("America/New_York")`; "trading day" = a date in `calendar`.
- [ ] **Step 4: Run** test file + full suite (incl. `tests/test_forward.py`) — PASS.
- [ ] **Step 5: Commit** `feat(disclosure): event table, signal dates and Panel.events`.

### Task 5: `disclosure` rule family and lab guards

**Files:**
- Create: `src/autotrade/rules/swing/disclosure.py`
- Modify: `src/autotrade/rules/registry.py`, `src/autotrade/pipeline.py` (event-family guards in `screen`, `validate`, `verify_final`; `events_table` in logged metrics), `src/autotrade/screen/run.py` (pass-through of the hash into `log.record` metrics), `src/autotrade/forward.py` (register stores `events_table` per rule; run refuses a mismatch), `src/autotrade/report/limitations.py` (event lines)
- Test: `tests/test_disclosure_rule.py`, additions to `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 4 `event_frame`, `events_table_hash`, `Panel.events`.
- Produces:
  - `class Disclosure(BaseRule)`: `family = "disclosure"`, `PARAM_KEYS = ("bundle", "hold", "k", "confirm", "sized")`, grid as in Global Constraints (108). Entries = event_frame(bundle, sized) ∧ confirm(t) ∧ eligible; confirm `none` → True, `ma20` → close > SMA20, `high20` → close > prior 20-day high; exits none (hold + stop); stop `atr_stop(panel, k)`; `max_hold_days = hold`; entry/exit `next_open`; confidence = 1 for sized events passing, 0.5 otherwise. `signals` raises `ValueError("disclosure rules need Panel.events")` when `panel.events is None`.
  - `EVENT_FAMILIES: set[str]` in registry (`disclosure` + Task 6 names).
  - `pipeline._refuse_incomplete_events(market)`: LabError when `events_running`, any target not complete, or `delisted_mapped / delisted_targets < 0.80` (message gives the coverage).
  - Logged metrics of event families include `"events_table": events_table_hash(market)`; validate / verify_final / forward raise `LabError("events table changed since screen of <rule_id>: re-screen (new trials)")` on mismatch.
  - Limitations lines (only when the family is an event family): "국내 공시 시각 없음 → 공시일 다음 거래일 시가 가정", "공시 제목·양식 기반 분류(규칙 표)", "상폐 종목 공시 매핑률 NN%".

- [ ] **Step 1: Write the failing tests** — `test_grid_has_108_points`, `test_entry_only_on_signal_date_next_open` (one buyback event on day t → entry flag exactly at t), `test_sized_true_skips_unknown_size`, `test_confirm_high20_requires_breakout`, `test_no_events_raises`, `test_screen_refuses_incomplete_events` (cli_env: targets.json coverage 0.5 → exit 1 with "coverage"), `test_validate_refuses_changed_table` (screen, edit config bytes, validate → "events table changed"), `test_non_event_family_screen_unaffected_without_events` (breakout screens with no events store).
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** new tests + full suite — PASS, no warnings.
- [ ] **Step 5: Commit** `feat(rules): disclosure swing family + event-table and completeness guards`.

### Task 6: `_negf` negative-event filter variants

**Files:**
- Create: `src/autotrade/rules/swing/negf.py`
- Modify: `src/autotrade/rules/registry.py`
- Test: `tests/test_negf.py`

**Interfaces:**
- Consumes: Task 4 `recent_negative`, `event_frame`; base classes `Breakout, MaTrend, Momentum, Reversal`.
- Produces: `with_negative_filter(base: type[BaseRule], window: int = 20) -> type[BaseRule]` returning a subclass with `family = f"{base.family}_negf"`, same `PARAM_KEYS` and `grid()`; `signals` = base signals with `entries &= ~recent_negative(panel, window)` and `exits |= (negative event signal_date == t)`; requires `panel.events` (ValueError otherwise). Registry adds `breakout_negf, ma_trend_negf, momentum_negf, reversal_negf` to `SWING_FAMILIES` and `EVENT_FAMILIES`.

- [ ] **Step 1: Write the failing tests** — `test_negf_rule_ids_differ_from_base` (family in the hash input so ids never collide with the base rule's logged ids — assert `rule_id.startswith("breakout_negf:")`), `test_entry_blocked_for_20_rows_after_negative`, `test_entry_allowed_on_row_21`, `test_negative_event_forces_exit_next_open`, `test_without_negative_events_equals_base` (signals identical to the base family when the events frame has only +1 events), `test_momentum_negf_keeps_equal_weight`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** new tests + full suite — PASS.
- [ ] **Step 5: Commit** `feat(rules): negative-event filter variants of breakout/ma_trend/momentum/reversal`.

### Task 7: SEC EDGAR client and US collector

**Files:**
- Create: `src/autotrade/disclosure/edgar.py`, `src/autotrade/disclosure/collect_us.py`
- Modify: `src/autotrade/pipeline.py` (`collect_events` dispatches by market)
- Test: `tests/test_edgar.py`

**Interfaces:**
- Consumes: Task 2 store (`append_filings`, state, targets, lock).
- Produces:
  - `EdgarClient(user_agent: str, http: httpx.Client | None = None, min_interval: float = 0.12)`: `tickers() -> pd.DataFrame` (`ticker, cik`), `submissions(cik: str) -> list[dict]` (recent + every older `filings.files` page merged; fields `accessionNumber, filingDate, acceptanceDateTime, form, items`), `insider_quarter(year: int, q: int) -> pd.DataFrame` (SEC insider-transactions quarterly data set; columns needed: accession, transaction code, shares, price). `make_edgar_client_from_env()` raises `LabError` "SEC_USER_AGENT missing in .env" when unset; the UA value never appears in messages or logs.
  - Raw US filing columns: `filing_id` (accession), `symbol`, `cik`, `filed_date`, `accepted_at` (UTC timestamp), `form`, `items`, `insider_usd` (sum of code-P shares × price per accession, NaN otherwise).
  - Forms kept: `8-K`, `8-K/A` (amendment → correction, not an event), `4`, `424B1`–`424B8`.
  - Collector semantics identical to Task 2 (state, resume, lock, `targets.json`; US delisted coverage = mapped share of all targets).
- [ ] **Step 1: Write the failing tests** — `test_submissions_merges_older_pages`, `test_user_agent_header_sent_and_never_in_errors`, `test_missing_user_agent_refuses`, `test_insider_buy_amount_sums_code_p_only`, `test_us_collect_resume_and_lock` (fake client).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** If the insider data-set layout differs from the assumption, fall back to parsing each Form 4 XML (transaction code `P`, shares × price) and note it in the module docstring.
- [ ] **Step 4: Run** new tests + full suite — PASS.
- [ ] **Step 5: Commit** `feat(disclosure): SEC EDGAR client + US filing collector`.

### Task 8: Daily refresh, docs

**Files:**
- Modify: `scripts/daily_kr.sh`, `README.md`, `docs/superpowers/specs/2026-11-24-disclosure-lab-design.md` (§10-1 #2: add the US insider size minimum 100,000 USD as designer's call), `docs/superpowers/specs/2026-11-19-auto-trading-design.md` (7.4: item 10 "공시 규칙군" one paragraph pointing to the disclosure spec)
- Test: `tests/test_daily_script.py` (static checks only)

**Interfaces:**
- Produces: `daily_kr.sh` runs `lab collect-events --market kr` **after** `lab collect` and before `lab forward`; a failure of the events step is logged and notified but does **not** block the forward report or mark the day failed (the forward test must not depend on DART).
- [ ] **Step 1: Write the failing test** — `test_daily_script_collects_events_after_bars` (the script text has `collect-events --market kr` after `lab collect --market kr` and before `lab forward`, and the events command is not part of the `&&` chain guarding `kr-done`).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Edit** the script and docs.
- [ ] **Step 4: Run** `bash -n scripts/daily_kr.sh` and full suite — PASS.
- [ ] **Step 5: Commit** `chore(disclosure): daily KR filing refresh + docs`.

### After the tasks (controller, real data — not subagent work)
1. `uv run lab collect-events --market kr` detached with a log under `data/logs/`; rerun daily until every target is complete (DART daily limit).
2. `uv run lab events --market kr --unmatched 50` → finalize `config/events/kr.yaml` patterns from title frequencies (no prices/returns looked at) → **show the table to the user** before any screen.
3. `lab screen --family disclosure` and each `_negf` family, then `lab validate` — journal + report to the user in Korean.
4. US: once the user has put `SEC_USER_AGENT` in `.env`, the same for `--market us`.
