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
