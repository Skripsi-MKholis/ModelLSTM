"""Backtest walk-forward lima model pembanding (Dokumen/28 Juli - M3.md §6).

Protokol: latih/hitung dari data s.d. origin t, uji t+1..t+7, geser origin
7 hari, ulangi sampai habis data test. Metrik: MAE, RMSE, MAPE, sMAPE per
horizon H+1/H+3/H+7.

Model 1-3 (naive, seasonal_naive, moving_average) dihitung ulang murni dari
statistik pada tiap origin — tidak butuh training. Model 4-5 (lstm_global,
lstm_finetuned) memakai `Models/lstm_daily.keras` yang SUDAH dilatih di
main.ipynb, dievaluasi lewat rolling-origin forward-chaining TANPA retraining
per fold (retraining ratusan fold model deep learning di luar anggaran waktu
skripsi ini). Ini penyimpangan dari walk-forward yang ketat (yang melatih
ulang model di tiap origin) dan dicatat secara terbuka di laporan, bukan
disembunyikan.

Keterbatasan data (§4.2): hanya toko Eatstedi sendiri yang punya seri harian
panjang (~315 hari aktif) di seluruh dataset yang diperiksa. `lstm_global`
dan `lstm_finetuned` karena itu adalah MODEL YANG SAMA di sini — keduanya
hanya bisa dibedakan sungguhan bila tersedia data lintas-toko yang cukup
untuk fine-tuning per toko. Baris `lstm_finetuned` di laporan adalah
duplikat `lstm_global`, ditandai jelas, bukan model kedua yang independen.

Kriteria lulus (§6): LSTM harus mengalahkan seasonal_naive pada MAPE H+1 DAN
H+7 di >= 60% dari unit uji. Karena hanya ada SATU toko, unit uji di sini
adalah *fold* (origin walk-forward), bukan toko — substitusi ini dicatat
eksplisit di keputusan akhir, sebagaimana diarahkan dokumen M3 untuk
menyatakan keterbatasan secara terang-terangan.

Jalankan: python evaluation/backtest.py
Output  : Models/backtest/backtest_summary.csv, Models/backtest/backtest_folds.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from v2 import baselines  # noqa: E402
from v2.calendar_utils import is_ramadan  # noqa: E402

DAILY_CSV = os.path.join(BASE_DIR, "Data", "Ekstrak", "daily_sales.csv")
OUT_DIR = os.path.join(BASE_DIR, "Models", "backtest")
LOOK_BACK = 14
HORIZONS = {"H+1": 1, "H+3": 3, "H+7": 7}
FOLD_STEP = 7
MAX_HORIZON = 7


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if not mask.any():
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _load_lstm():
    import pickle
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, "Models", "lstm_daily.keras"))
    with open(os.path.join(BASE_DIR, "Models", "scalers", "scaler_daily.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "Models", "scalers", "scaler_revenue.pkl"), "rb") as f:
        scaler_rev = pickle.load(f)
    return model, scaler, scaler_rev


def _lstm_predict(model, scaler, scaler_rev, df_active, origin_idx):
    window = df_active.iloc[origin_idx - LOOK_BACK: origin_idx]
    week = window["date"].dt.isocalendar().week.astype(float)
    month = window["date"].dt.month.astype(float)
    dow = window["date"].dt.weekday.astype(float)
    ramadan_flags = window["date"].apply(is_ramadan).to_numpy(dtype=float)
    feats = np.column_stack([
        window["revenue"].to_numpy(dtype=float),
        window["transactions"].to_numpy(dtype=float),
        window["qty_sold"].to_numpy(dtype=float),
        np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5),
        ramadan_flags,
    ])
    scaled = scaler.transform(feats)
    x = scaled.reshape(1, LOOK_BACK, feats.shape[1])
    pred_scaled = model.predict(x, verbose=0)[0]
    pred_rev = scaler_rev.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return np.clip(pred_rev, 0, None)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    df_active = df[df["revenue"] > 0].sort_values("date").reset_index(drop=True)
    n = len(df_active)
    print(f"[BACKTEST] {n} hari aktif dimuat dari {DAILY_CSV}")

    print("[BACKTEST] Memuat model LSTM terlatih (Models/lstm_daily.keras)...")
    lstm_model, lstm_scaler, lstm_scaler_rev = _load_lstm()

    first_origin = max(LOOK_BACK, int(n * 0.70))
    origins = list(range(first_origin, n - MAX_HORIZON, FOLD_STEP))
    print(f"[BACKTEST] {len(origins)} fold walk-forward, origin pertama index {first_origin}")

    fold_rows = []
    for origin in origins:
        history_rev = df_active["revenue"].to_numpy(dtype=float)[:origin]
        y_true = df_active["revenue"].to_numpy(dtype=float)[origin: origin + MAX_HORIZON]
        if len(y_true) < MAX_HORIZON:
            continue

        preds = {
            "naive": baselines.naive(history_rev, MAX_HORIZON),
            "seasonal_naive": baselines.seasonal_naive(history_rev, MAX_HORIZON),
            "moving_average_7": baselines.moving_average(history_rev, MAX_HORIZON, window=7),
        }
        try:
            lstm_pred = _lstm_predict(lstm_model, lstm_scaler, lstm_scaler_rev, df_active, origin)
            preds["lstm_global"] = lstm_pred
            preds["lstm_finetuned"] = lstm_pred  # stand-in, lihat catatan modul
        except Exception as e:
            print(f"[BACKTEST][WARN] LSTM gagal di origin {origin}: {e}")
            continue

        for model_name, pred in preds.items():
            for h_label, h in HORIZONS.items():
                fold_rows.append({
                    "origin_date": df_active.iloc[origin]["date"].strftime("%Y-%m-%d"),
                    "model": model_name,
                    "horizon": h_label,
                    "mae": _mae(y_true[:h], pred[:h]),
                    "rmse": _rmse(y_true[:h], pred[:h]),
                    "mape": _mape(y_true[:h], pred[:h]),
                    "smape": _smape(y_true[:h], pred[:h]),
                })

    folds_df = pd.DataFrame(fold_rows)
    folds_path = os.path.join(OUT_DIR, "backtest_folds.csv")
    folds_df.to_csv(folds_path, index=False)
    print(f"[BACKTEST] Detail per-fold disimpan: {folds_path}")

    summary = (
        folds_df.groupby(["model", "horizon"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mape=("mape", "mean"),
             smape=("smape", "mean"), n_samples=("mae", "count"))
        .reset_index()
    )
    summary_path = os.path.join(OUT_DIR, "backtest_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[BACKTEST] Ringkasan disimpan: {summary_path}")
    print(summary.to_string(index=False))

    # ── Kriteria lulus (§6): LSTM vs seasonal_naive, MAPE H+1 DAN H+7, per fold ──
    pivot_h1 = folds_df[folds_df["horizon"] == "H+1"].pivot(index="origin_date", columns="model", values="mape")
    pivot_h7 = folds_df[folds_df["horizon"] == "H+7"].pivot(index="origin_date", columns="model", values="mape")
    wins_h1 = (pivot_h1["lstm_global"] < pivot_h1["seasonal_naive"]).sum()
    wins_h7 = (pivot_h7["lstm_global"] < pivot_h7["seasonal_naive"]).sum()
    total_folds = len(pivot_h1)
    wins_both = ((pivot_h1["lstm_global"] < pivot_h1["seasonal_naive"]) &
                 (pivot_h7["lstm_global"] < pivot_h7["seasonal_naive"])).sum()
    pass_rate = wins_both / total_folds if total_folds else 0.0
    passed = pass_rate >= 0.60

    decision_lines = [
        "# M3 — Hasil Backtest & Keputusan Kriteria Produksi",
        "",
        f"> Dihasilkan otomatis oleh `evaluation/backtest.py`. Total fold walk-forward: {total_folds}.",
        "",
        "## Ringkasan metrik (rata-rata lintas fold)",
        "",
        summary.to_markdown(index=False) if hasattr(summary, "to_markdown") else summary.to_string(index=False),
        "",
        "## Kriteria lulus (§6 dokumen M3)",
        "",
        "LSTM harus mengalahkan seasonal_naive pada MAPE H+1 **dan** H+7 di >= 60% unit uji.",
        "",
        "**Keterbatasan yang diketahui**: dataset ini hanya berisi satu toko (Eatstedi) dengan "
        "seri harian panjang, jadi 'unit uji' di sini adalah *fold* walk-forward, bukan toko "
        "seperti diasumsikan kriteria asli (§4.2 mencatat mengapa model lintas-toko tidak "
        "diandalkan dengan komposisi data yang ada). `lstm_finetuned` adalah duplikat "
        "`lstm_global` karena tidak ada model fine-tuned terpisah yang bisa dilatih dari satu "
        "toko saja — dicatat di sini, bukan disembunyikan.",
        "",
        f"- Fold LSTM menang MAPE H+1: {wins_h1}/{total_folds}",
        f"- Fold LSTM menang MAPE H+7: {wins_h7}/{total_folds}",
        f"- Fold LSTM menang **keduanya** (H+1 dan H+7): {wins_both}/{total_folds} = {pass_rate*100:.1f}%",
        "",
        f"## Keputusan: {'LULUS' if passed else 'TIDAK LULUS'}",
        "",
        (
            "LSTM mengalahkan seasonal_naive pada >= 60% fold untuk MAPE H+1 dan H+7. "
            "Server v2 boleh mengembalikan `metadata.model_used = \"lstm\"` untuk toko yang "
            "memenuhi ambang data (>= 45 hari aktif)."
            if passed else
            "LSTM TIDAK mengalahkan seasonal_naive secara konsisten (>= 60% fold) pada MAPE H+1 "
            "dan H+7. Sesuai §6 dokumen M3: **baseline (seasonal_naive) tetap dipakai di "
            "produksi**, dan server tidak boleh mengembalikan `model_used = \"lstm\"` sampai "
            "kriteria ini terpenuhi — meski `/api/v2/forecast` tetap mengaktifkan jalur LSTM "
            "bila data mencukupi, hasil negatif ini adalah temuan yang sah dan harus dilaporkan "
            "apa adanya di skripsi, bukan disembunyikan atau dipaksakan lulus."
        ),
    ]
    with open(os.path.join(BASE_DIR, "Dokumen", "M3 - Hasil Backtest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(decision_lines))
    print(f"[BACKTEST] Keputusan dicatat: Dokumen/M3 - Hasil Backtest.md -> {'LULUS' if passed else 'TIDAK LULUS'}")

    return passed


if __name__ == "__main__":
    run()
