# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Undergraduate thesis (*skripsi*) project: LSTM time-series forecasting of sales revenue for **Eatstedi**, a canteen at DTEDI UGM. Data spans 24 Aug 2024 – 20 Jun 2026 (~316 active business days). Code comments, docs, and variable names are almost entirely in **Indonesian** — match that language when editing existing files.

There is no test suite, linter, or build step. This is a notebook + scripts + Flask API project.

## Commands

```bash
pip install -r requirements.txt          # torch comes from the CPU wheel index (see requirements.txt)

python app.py                            # Flask POS-integration API for Eatstedi, port 5000, debug=on
python app_public.py                     # Generic/multi-tenant variant of the same API

python Scripts/generate_notebook.py      # Regenerate main.ipynb from scratch (run from repo root)

# TimeGAN synthetic-data augmentation (PyTorch). NOTE: Data/Note.md says
# `python Data/TimeGAN.py` but the actual path is Data/TimeGAN/TimeGAN.py
python Data/TimeGAN/TimeGAN.py --mode all       # train + generate (default)
python Data/TimeGAN/TimeGAN.py --mode train     # train + save models only
python Data/TimeGAN/TimeGAN.py --mode generate  # generate from saved models
```

The LSTM research pipeline itself lives in `main.ipynb` and is run cell-by-cell in Jupyter, not from the CLI.

## Two independent tracks — do not conflate them

**This is the single most important thing to understand.** The project has two parts that are architecturally disconnected:

1. **Research pipeline (`main.ipynb`)** — trains the actual LSTM models. Produces `Models/*.keras` and `Models/scalers/*.pkl`.
2. **Production API (`app.py` / `app_public.py`)** — serves forecasts to the POS, but **does NOT load or use the trained LSTM models.** The endpoints return simple statistical baselines computed on the fly: Seasonal-Naive (k=4, by day-of-week) for daily, Mean-4 for weekly, Mean-3 for monthly. If you're asked to "improve the model's predictions in the API," note that the API never touches `.keras` files — wiring the LSTM into the API would be new work, not a bug fix.

`Models/TimeGAN/comparison_results.csv` reflects why: on this tiny dataset the Seasonal-Naive baseline is competitive with (often beats) the LSTM, so the API ships the robust baseline.

## Data pipeline (main.ipynb, Sections 0–7)

Flow: `eatstedi-*.sql` → `raw_transactions.csv` → daily/weekly/monthly aggregates → sequences → 3 LSTM models → eval → forecast.

- **Section 1 parses a ~32MB MariaDB dump directly in Python** (`_parse_sql_tuples`, a hand-rolled char-level VALUES parser) — no MySQL server involved. Only 4 tables are extracted: `invoices` (filtered `is_paid=1`), `product_sold`, `products`, `categories`. Categories are hardcoded (id→name map) because their SVG icon columns are hard to parse.
- **Every stage is cache-guarded**: if the output CSV already exists, the cell skips regeneration. To force a rebuild, delete the target CSV. The 32MB SQL parse is the slow step — this guard exists to avoid re-running it.
- Data lives in `Data/Ekstrak/` (note the subfolder): `daily_sales.csv`, `weekly_sales.csv`, `monthly_sales.csv`, `raw_transactions.csv`, and the `.sql` dump. The `.sql` dump is source-of-truth — do not edit it.

### Core domain concept: business-day filtering

Roughly half of calendar days have `revenue=0` (weekends, Jan/Jul academic breaks, Ramadan lull). The pipeline **drops all `revenue==0` rows** and trains on the sequence of *active days only* — the model learns "active day → next active day," not calendar days. When forecasting, output dates are mapped back onto the calendar by skipping Sat/Sun and months 1 & 7. Any new forecasting or windowing code must preserve this: never feed zero-revenue days into a sequence, and always skip closed days when generating future dates.

Binary context flags in the data: `is_weekend`, `is_holiday` (month in {1,7}), `is_ramadan` (Mar 2025 / Mar 2026).

### Model configs (source of truth is main.ipynb, not the Planning doc)

| Granularity | look_back | horizon | split | notes |
|---|---|---|---|---|
| Daily   | 14 active days | 7 | 70/15/15 chronological | 10 features incl. cyclical sin/cos encodings + `is_ramadan` |
| Weekly  | 5 weeks | 4 | 70/15/15 | reduced from 8 due to ~75 active weeks |
| Monthly | 3 months | 3 | 65/35, **no val set** | ~20 months only; EarlyStopping monitors `loss` not `val_loss` |

Shared `build_lstm_model` = stacked 2-layer LSTM (units, units//2) + Dropout(0.2) + L2(1e-4) + Dense(32,relu) + Dense(horizon). `create_sequences` predicts `target_col=0` (revenue). Splits are **chronological, never shuffled** (time-series leakage). `MinMaxScaler` is fit on train only; a separate revenue-only scaler is saved for `inverse_transform` back to Rupiah in evaluation. Metrics: RMSE, MAE, MAPE, all computed in Rupiah after inverse-transform.

## v2 API (M3 — `v2/`, `evaluation/`)

Implements the contract in `Dokumen/28 Juli - M3.md` §3 for the Flutter POS client's `LstmApiClient`. Both `app.py` and `app_public.py` call `register_v2_routes(app)` from `v2/routes.py` at import time, so v1 endpoints keep working unmodified (client falls back to v1 only on 404/405) while v2 lives at the same host/port.

- `v2/baselines.py` — naive, seasonal_naive(k=7), moving_average(7). Pure functions, no I/O.
- `v2/lstm_adapter.py` — wraps the pre-trained `Models/lstm_daily.keras` (look_back=14, 7-day native horizon, 10 features) with a recursive rollout to cover the 30-day contract horizon. Future `transactions`/`qty_sold` (unknown at inference time) are estimated from recent Rp/transaction and Rp/unit ratios, not invented. Confidence decays and the `revenue_low`/`revenue_high` band widens with rollout depth, per §4.5.
- `v2/forecast_service.py` — orchestrates the tiered fallback (`lstm → seasonal_naive → naive → empty`) from §5, and builds the full `/api/v2/forecast` response (hourly shares, product_demand, recommendations).
- `evaluation/backtest.py` — walk-forward backtest of the 5 models required by §6, writing `Models/backtest/backtest_summary.csv` + `backtest_folds.csv` and `Dokumen/M3 - Hasil Backtest.md` (the pass/fail decision). Re-run with `python evaluation/backtest.py` whenever `daily_sales.csv` or the trained model changes.

**Production gate on `model_used = "lstm"`**: `forecast_service.lstm_cleared_production_bar()` reads the LULUS/TIDAK LULUS verdict out of `Dokumen/M3 - Hasil Backtest.md`. As of the last run (see that file) the verdict is **TIDAK LULUS** — LSTM does not beat seasonal_naive on MAPE H+1 *and* H+7 in ≥60% of walk-forward folds on this single-store dataset — so `/api/v2/forecast` currently serves `seasonal_naive` even when a store has ≥45 days of history, with `fallback_reason: "backtest_not_passed"`. This is not a bug: don't "fix" it by hardcoding `model_used = "lstm"`; re-run the backtest after retraining/augmenting data and let the decision flip naturally.

`lstm_finetuned` in the backtest is a documented stand-in for `lstm_global` (same model) — this repo's data has only one store with a long series (Eatstedi itself), so there's nothing to fine-tune against yet (see §4.2 of the M3 doc and the caveat at the top of `evaluation/backtest.py`).

## app.py vs app_public.py

Same 7 endpoints and heuristics; they differ in tenant assumptions:

- **`app.py`** — hardcoded to Eatstedi: weekends + Jan/Jul treated as closed (`is_business_day`), category shares and top-products baked in as Eatstedi's real historical values, falls back to those if `raw_transactions.csv` is missing.
- **`app_public.py`** — generic storefront: open 7 days, no closed months by default (`OPEN_ON_WEEKENDS_DEFAULT`, `CLOSED_MONTHS_DEFAULT`, overridable per-request), placeholder category/product defaults.

Both auto-load product/category statistics from `raw_transactions.csv` at startup (`load_product_statistics`) and only fall back to hardcoded defaults if that file is absent or malformed. `/api/sales/record` mutates `daily_sales.csv` in place (upsert by date). Full endpoint reference: `Dokumen/API Documentation.md` and `Dokumen/API Documentation Public.md`.

## TimeGAN augmentation (`Data/TimeGAN/TimeGAN.py`)

Standalone PyTorch implementation (Yoon et al. 2019, NeurIPS) to synthesize extra daily sequences and combat the small-dataset problem. Trains 5 nets (embedder, recovery, generator, supervisor, discriminator) in 3 phases and writes `synthetic_daily.csv` (200 sequences × 24 days). The augmented-vs-baseline LSTM comparison is driven from `Eksperimen/lstm_timegan_augmented.ipynb`, with results in `Models/TimeGAN/`.

## Layout notes

- `Eksperimen/` — exploratory notebooks (simplified LSTM, hybrid-residual, recursive-forecast fix, multi-seed evaluation, TimeGAN augmentation). These are experiments, not the canonical pipeline; `main.ipynb` is canonical.
- `Dokumen/` — Indonesian planning/analysis docs and API references. `Planning.md` is an early design doc; where it disagrees with `main.ipynb` (e.g. feature list, look-back values), **the notebook wins**.
- `Models/` holds the canonical `.keras` models + `scalers/`; `Models/TimeGAN/` holds the comparison-experiment models and CSVs.
