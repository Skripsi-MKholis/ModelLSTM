"""Model-model baseline untuk forecasting revenue harian.

Semua fungsi menerima array revenue historis (urut kronologis, hanya
hari aktif) dan mengembalikan array prediksi sepanjang `horizon`.
Tidak ada I/O maupun ketergantungan waktu (`datetime.now`) di sini —
pemetaan ke tanggal kalender dilakukan oleh pemanggil.
"""
from __future__ import annotations

import numpy as np


def naive(revenue: np.ndarray, horizon: int) -> np.ndarray:
    """y[t+h] = y[t] untuk semua h. Batas bawah akurasi."""
    last = float(revenue[-1])
    return np.full(horizon, last, dtype=float)


def seasonal_naive(revenue: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """y[t+h] = y[t+h-season]. Dipakai produksi sekarang, harus dikalahkan LSTM."""
    n = len(revenue)
    out = np.empty(horizon, dtype=float)
    for h in range(horizon):
        idx = n - season + (h % season)
        if idx < 0:
            idx = idx % n
        out[h] = revenue[idx]
    return out


def moving_average(revenue: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    """Rata-rata `window` hari terakhir, diulang untuk seluruh horizon."""
    w = min(window, len(revenue))
    avg = float(np.mean(revenue[-w:]))
    return np.full(horizon, avg, dtype=float)
