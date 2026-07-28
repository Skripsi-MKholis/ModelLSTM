"""Orkestrasi kontrak POST /api/v2/forecast (Dokumen/28 Juli - M3.md §3.2, §5).

Fallback berjenjang:
    LSTM (>= MIN_DAYS_REQUIRED hari aktif)
      -> seasonal_naive (>= 14 hari)
          -> naive (>= 1 hari)
              -> daily: [] (tidak ada data sama sekali)

`metadata.model_used` HANYA salah satu dari: lstm, lstm_finetuned,
seasonal_naive, naive — nilai ini menentukan label di UI klien.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from . import baselines
from . import lstm_adapter

MODEL_VERSION = "lstm-v2.1.0"
MIN_DAYS_REQUIRED = lstm_adapter.MIN_DAYS_REQUIRED
SEASONAL_MIN_DAYS = 14

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKTEST_SUMMARY_PATH = os.path.join(_REPO_ROOT, "Models", "backtest", "backtest_summary.csv")
_BACKTEST_DECISION_PATH = os.path.join(_REPO_ROOT, "Dokumen", "M3 - Hasil Backtest.md")
_metrics_cache: dict | None = None
_lstm_cleared_cache: bool | None = None


def lstm_cleared_production_bar() -> bool:
    """§6: LSTM hanya boleh dilabeli/dipakai di produksi bila backtest LULUS.

    Bila laporan belum ada (backtest belum pernah dijalankan), anggap belum
    lulus — jangan optimis secara default.
    """
    global _lstm_cleared_cache
    if _lstm_cleared_cache is not None:
        return _lstm_cleared_cache
    if not os.path.exists(_BACKTEST_DECISION_PATH):
        _lstm_cleared_cache = False
        return False
    with open(_BACKTEST_DECISION_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    _lstm_cleared_cache = "Keputusan: LULUS" in text
    return _lstm_cleared_cache


def _load_backtest_metrics() -> dict | None:
    """Baca hasil backtest lstm_global (§6) untuk mengisi `metrics` di respons.

    Bukan sekadar kosmetik: field ini dipakai laporan skripsi untuk menyertakan
    MAPE/RMSE aktual dari model yang sedang melayani permintaan, bukan angka statis.
    """
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache
    if not os.path.exists(_BACKTEST_SUMMARY_PATH):
        return None
    try:
        df = pd.read_csv(_BACKTEST_SUMMARY_PATH)
        lstm_row = df[(df["model"] == "lstm_global") & (df["horizon"] == "H+1")]
        sn_row = df[(df["model"] == "seasonal_naive") & (df["horizon"] == "H+1")]
        if lstm_row.empty or sn_row.empty:
            return None
        _metrics_cache = {
            "backtest_mape": float(lstm_row.iloc[0]["mape"]),
            "backtest_rmse": float(lstm_row.iloc[0]["rmse"]),
            "baseline_mape": float(sn_row.iloc[0]["mape"]),
        }
        return _metrics_cache
    except Exception:
        return None


def _build_daily_dataframe(history_daily: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(history_daily)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"tx_count": "transactions", "item_count": "qty_sold"})
    for col in ("transactions", "qty_sold"):
        if col not in df.columns:
            df[col] = 0
    df_active = df[df["revenue"] > 0].copy()
    df_active["day_of_week"] = df_active["date"].dt.weekday
    return df_active


def _make_future_date_generator(open_weekdays: list[int], closed_months: list[int]):
    open_set = set(open_weekdays) if open_weekdays else set(range(1, 8))

    def _is_open(date: pd.Timestamp) -> bool:
        iso_dow = date.isoweekday()  # 1=Senin..7=Minggu, sama seperti kontrak
        if iso_dow not in open_set:
            return False
        if date.month in closed_months:
            return False
        return True

    def generate(start_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
        out = []
        cur = pd.Timestamp(start_date)
        # batas pengaman: jangan mencari lebih dari 10x n hari kalender
        guard = 0
        while len(out) < n and guard < n * 20 + 60:
            cur += pd.Timedelta(days=1)
            guard += 1
            if _is_open(cur):
                out.append(cur)
        return out

    return generate


def _fallback_reason_for(input_days: int) -> str:
    return "insufficient_history"


def _forecast_daily_baseline(df_active: pd.DataFrame, horizon: int, generate_future_dates) -> tuple[str, np.ndarray, list[pd.Timestamp]]:
    revenue = df_active["revenue"].to_numpy(dtype=float)
    future_dates = generate_future_dates(pd.Timestamp(df_active["date"].max()), horizon)
    if len(revenue) >= SEASONAL_MIN_DAYS:
        model_used = "seasonal_naive"
        pred = baselines.seasonal_naive(revenue, horizon)
    else:
        model_used = "naive"
        pred = baselines.naive(revenue, horizon)
    return model_used, pred, future_dates


def _product_demand(products: list[dict], horizon_days: int) -> list[dict]:
    out = []
    for p in products:
        avg_daily = float(p.get("avg_daily_qty") or 0.0)
        predicted_qty = avg_daily
        predicted_qty_week = avg_daily * 7
        recommended_qty = predicted_qty_week * 1.15  # safety buffer, konsisten dgn v1
        out.append({
            "product_id": p.get("product_id"),
            "product_name": p.get("product_name"),
            "category": p.get("category"),
            "predicted_qty": int(round(predicted_qty)),
            "predicted_qty_week": int(round(predicted_qty_week)),
            "recommended_qty": int(round(recommended_qty)),
            "confidence": 0.5,
            "trend": "stable",
        })
    return out


def _hourly_forecast(history_hourly: list[dict]) -> list[dict]:
    if not history_hourly:
        return []
    df = pd.DataFrame(history_hourly)
    if df.empty or "hour" not in df.columns:
        return []
    by_hour = df.groupby("hour").agg(tx_count=("tx_count", "mean")).reset_index()
    total = by_hour["tx_count"].sum()
    out = []
    for _, row in by_hour.sort_values("hour").iterrows():
        share = float(row["tx_count"] / total) if total > 0 else 0.0
        out.append({
            "hour": int(row["hour"]),
            "tx_count": int(round(row["tx_count"])),
            "share": round(share, 4),
            "confidence": 0.6,
        })
    return out


def _recommendations(df_active: pd.DataFrame, hourly: list[dict]) -> list[dict]:
    recs = []
    if not df_active.empty:
        last_15 = df_active.tail(15)["revenue"]
        mean_15 = float(last_15.mean())
        moderate = mean_15 * 1.10
        aggressive = mean_15 * 1.25
        recs.append({
            "kind": "target_omzet",
            "title": "Target Omzet Harian",
            "desc": "",
            "badge": "TARGET",
            "rationale": f"Dihitung dari rata-rata {len(last_15)} hari aktif terakhir.",
            "payload": {"moderate": int(round(moderate)), "aggressive": int(round(aggressive))},
        })
    if hourly:
        quiet = min(hourly, key=lambda h: h["share"])
        recs.append({
            "kind": "happy_hour",
            "title": "Promo Jam Sepi",
            "desc": "",
            "badge": "PROMO",
            "rationale": f"Jam {quiet['hour']:02d}.00 hanya menyumbang {quiet['share']*100:.1f}% transaksi harian.",
            "payload": {
                "discount_percent": 15,
                "hour_from": quiet["hour"],
                "hour_to": min(23, quiet["hour"] + 2),
                "product_ids": [],
            },
        })
    return recs


def build_forecast_response(payload: dict) -> dict:
    store_profile = payload.get("store_profile") or {}
    history = payload.get("history") or {}
    horizon = payload.get("horizon") or {}
    horizon_daily = int(horizon.get("daily", 30))

    open_weekdays = store_profile.get("open_weekdays") or []
    if not open_weekdays and store_profile.get("open_on_weekends") is not None:
        open_weekdays = list(range(1, 8)) if store_profile["open_on_weekends"] else list(range(1, 6))
    closed_months = store_profile.get("closed_months") or []

    generate_future_dates = _make_future_date_generator(open_weekdays, closed_months)
    df_active = _build_daily_dataframe(history.get("daily") or [])

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    input_days = int(len(df_active))

    if input_days == 0:
        metadata = {
            "model_used": "naive",
            "model_version": MODEL_VERSION,
            "fallback_reason": "insufficient_history",
            "input_days": 0,
            "min_days_required": MIN_DAYS_REQUIRED,
            "generated_at": generated_at,
        }
        return {
            "metadata": metadata,
            "metrics": None,
            "daily": [],
            "hourly": [],
            "product_demand": [],
            "recommendations": [],
        }

    model_used = None
    fallback_reason = None
    daily_points = None
    metrics = None

    if input_days >= MIN_DAYS_REQUIRED and not lstm_cleared_production_bar():
        model_used = None
        fallback_reason = "backtest_not_passed"
        print("[V2][INFO] LSTM tersedia tapi belum lulus kriteria produksi (§6) - memakai baseline.")

    if input_days >= MIN_DAYS_REQUIRED and model_used is None and fallback_reason is None:
        try:
            result = lstm_adapter.forecast_daily(df_active, horizon_daily, generate_future_dates)
            model_used = "lstm"
            fallback_reason = None
            metrics = _load_backtest_metrics()
            daily_points = []
            for i, date in enumerate(result["dates"]):
                daily_points.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "revenue": int(round(result["revenue"][i])),
                    "revenue_low": int(round(result["revenue_low"][i])),
                    "revenue_high": int(round(result["revenue_high"][i])),
                    "tx_count": None,
                    "confidence": round(float(result["confidence"][i]), 2),
                })
        except Exception as e:  # noqa: BLE001
            model_used = None
            fallback_reason = "model_unavailable"
            print(f"[V2][WARN] LSTM gagal, jatuh ke baseline: {e}")

    if daily_points is None:
        baseline_model_used, pred, future_dates = _forecast_daily_baseline(df_active, horizon_daily, generate_future_dates)
        model_used = baseline_model_used
        if fallback_reason is None:
            fallback_reason = _fallback_reason_for(input_days)
        daily_points = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "revenue": int(round(pred[i])),
                "revenue_low": None,
                "revenue_high": None,
                "tx_count": None,
                "confidence": None,
            }
            for i, date in enumerate(future_dates)
        ]

    hourly = _hourly_forecast(history.get("hourly") or [])
    product_demand = _product_demand(history.get("products") or [], horizon_daily)
    recommendations = _recommendations(df_active, hourly)

    metadata = {
        "model_used": model_used,
        "model_version": MODEL_VERSION,
        "fallback_reason": fallback_reason,
        "input_days": input_days,
        "min_days_required": MIN_DAYS_REQUIRED,
        "generated_at": generated_at,
    }

    return {
        "metadata": metadata,
        "metrics": metrics,
        "daily": daily_points,
        "hourly": hourly,
        "product_demand": product_demand,
        "recommendations": recommendations,
    }
