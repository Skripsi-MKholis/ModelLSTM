"""Adapter di atas model `Models/lstm_daily.keras` yang sudah dilatih di main.ipynb.

Model itu dilatih dengan look_back=14, horizon=7, dan 10 fitur:
[revenue, transactions, qty_sold, week_sin, week_cos, month_sin, month_cos,
 dow_sin, dow_cos, is_ramadan]  (lihat main.ipynb sel 22-23, 30).

Kontrak v2 (Dokumen/28 Juli - M3.md §3.2) minta horizon 30 hari, sehingga
di sini dipakai *recursive rollout*: prediksi 7-hari diulang lagi sebagai
bagian dari window input berikutnya sampai horizon terpenuhi. transactions
dan qty_sold masa depan tidak diketahui — diestimasi dari rasio historis
Rp/transaksi dan Rp/unit pada window terakhir, bukan dikarang bebas.

Model ini adalah satu-satunya model LSTM yang tersedia di repo (lihat
catatan §4.2 dokumen M3: hanya satu toko dengan seri panjang), sehingga ia
dipakai juga sebagai stand-in untuk "lstm_finetuned" pada backtest — bukan
model terpisah. Ini didokumentasikan secara terbuka di laporan backtest,
bukan disembunyikan.
"""
from __future__ import annotations

import os
import pickle
import threading

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "Models")

LOOK_BACK = 14
NATIVE_HORIZON = 7
FEATURES = [
    "revenue", "transactions", "qty_sold",
    "week_sin", "week_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "is_ramadan",
]
MIN_DAYS_REQUIRED = 45  # ambang aktivasi LSTM, per §4.3 dokumen M3

_lock = threading.Lock()
_model = None
_scaler = None
_scaler_rev = None
_load_error: str | None = None


def _load():
    global _model, _scaler, _scaler_rev, _load_error
    if _model is not None or _load_error is not None:
        return
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import tensorflow as tf  # import lokal: berat, hanya dipakai jalur LSTM
            model_path = os.path.join(MODELS_DIR, "lstm_daily.keras")
            scaler_path = os.path.join(MODELS_DIR, "scalers", "scaler_daily.pkl")
            scaler_rev_path = os.path.join(MODELS_DIR, "scalers", "scaler_revenue.pkl")
            _model = tf.keras.models.load_model(model_path)
            with open(scaler_path, "rb") as f:
                _scaler = pickle.load(f)
            with open(scaler_rev_path, "rb") as f:
                _scaler_rev = pickle.load(f)
        except Exception as e:  # noqa: BLE001 - dilaporkan ke pemanggil sebagai fallback_reason
            _load_error = str(e)


def is_ready() -> bool:
    _load()
    return _model is not None


def load_error() -> str | None:
    _load()
    return _load_error


def _row_features(date: pd.Timestamp, revenue: float, tx: float, qty: float) -> list[float]:
    week = date.isocalendar()[1]
    month = date.month
    dow = date.weekday()
    is_ramadan = 1.0 if month == 3 else 0.0  # heuristik: kalender Islam tidak tersedia untuk toko umum
    return [
        revenue, tx, qty,
        np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5),
        is_ramadan,
    ]


def forecast_daily(df_active: pd.DataFrame, horizon: int, generate_future_dates) -> dict:
    """Menghasilkan prediksi revenue harian sepanjang `horizon` via rollout rekursif.

    `df_active` wajib berkolom date/revenue/transactions/qty_sold, terurut kronologis,
    hanya hari aktif (revenue > 0). `generate_future_dates(start_date, n)` adalah
    fungsi pemanggil yang tahu kalender toko (weekend/closed_months).

    Returns dict: {"dates": [...], "revenue": [...], "revenue_low": [...],
                   "revenue_high": [...], "confidence": [...]}
    Raises RuntimeError bila model belum siap.
    """
    if not is_ready():
        raise RuntimeError(load_error() or "model belum dimuat")

    last_date = pd.Timestamp(df_active["date"].max())
    future_dates = generate_future_dates(last_date, horizon)

    window = df_active.tail(LOOK_BACK * 3) if len(df_active) >= LOOK_BACK * 3 else df_active.copy()
    rows = [
        _row_features(pd.Timestamp(r["date"]), float(r["revenue"]), float(r["transactions"]), float(r["qty_sold"]))
        for _, r in window.tail(LOOK_BACK).iterrows()
    ]

    # Rasio historis untuk mengestimasi transaksi/qty di hari yang belum terjadi.
    recent = df_active.tail(30) if len(df_active) >= 30 else df_active
    rev_per_tx = float((recent["revenue"] / recent["transactions"].replace(0, np.nan)).mean())
    rev_per_qty = float((recent["revenue"] / recent["qty_sold"].replace(0, np.nan)).mean())
    rev_per_tx = rev_per_tx if np.isfinite(rev_per_tx) and rev_per_tx > 0 else float(recent["revenue"].mean())
    rev_per_qty = rev_per_qty if np.isfinite(rev_per_qty) and rev_per_qty > 0 else float(recent["revenue"].mean())

    predicted_revenue: list[float] = []
    step_confidence: list[float] = []
    rollouts_done = 0

    while len(predicted_revenue) < horizon:
        arr = np.array(rows[-LOOK_BACK:], dtype=float)
        scaled = _scaler.transform(arr)
        x = scaled.reshape(1, LOOK_BACK, len(FEATURES))
        pred_scaled = _model.predict(x, verbose=0)[0]
        pred_rev = _scaler_rev.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        pred_rev = np.clip(pred_rev, 0, None)

        take = min(NATIVE_HORIZON, horizon - len(predicted_revenue))
        for i in range(take):
            step = len(predicted_revenue)
            date = future_dates[step]
            rev = float(pred_rev[i])
            predicted_revenue.append(rev)
            # Keyakinan menurun seiring makin jauh rollout (galat kumulatif §4.5).
            step_confidence.append(max(0.35, 0.85 - 0.03 * step))

            tx_est = max(1.0, rev / rev_per_tx)
            qty_est = max(1.0, rev / rev_per_qty)
            rows.append(_row_features(date, rev, tx_est, qty_est))

        rollouts_done += 1
        if rollouts_done > 20:  # safety valve, tidak seharusnya tercapai untuk horizon wajar
            break

    revenue = np.array(predicted_revenue[:horizon])
    confidence = np.array(step_confidence[:horizon])
    # Pita interval melebar bersama menurunnya keyakinan — bukan disembunyikan (§4.5).
    spread = revenue * (1.0 - confidence) * 0.6
    return {
        "dates": future_dates[:horizon],
        "revenue": revenue,
        "revenue_low": np.clip(revenue - spread, 0, None),
        "revenue_high": revenue + spread,
        "confidence": confidence,
    }
