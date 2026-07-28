"""Definisi kalender bisnis & fitur tanggal yang sama dengan yang dipakai saat
melatih model di `main.ipynb` / `Scripts/generate_notebook.py`.

Sebelumnya `v2/lstm_adapter.py` dan `evaluation/backtest.py` masing-masing
punya definisi `is_ramadan` sendiri (heuristik `month == 3` untuk tahun
berapa pun), berbeda dari definisi asli saat training (hanya Maret 2025 dan
Maret 2026, lihat `Scripts/generate_notebook.py`). Modul ini adalah satu
sumber kebenaran supaya training dan serving/backtest tidak bisa drift lagi.
"""
from __future__ import annotations

import pandas as pd

# Bulan-bulan yang secara historis merupakan Ramadan pada rentang data Eatstedi
# (24 Agu 2024 - 20 Jun 2026). Sama persis dengan Scripts/generate_notebook.py.
RAMADAN_YEAR_MONTHS = {(2025, 3), (2026, 3)}

# Kalender operasional Eatstedi: Senin-Jumat, tutup Januari & Juli (libur akademik).
EATSTEDI_OPEN_WEEKDAYS = [1, 2, 3, 4, 5]  # ISO weekday, 1=Senin..7=Minggu
EATSTEDI_CLOSED_MONTHS = [1, 7]


def is_ramadan(date: pd.Timestamp) -> float:
    """1.0 bila (tahun, bulan) termasuk periode Ramadan yang dipakai saat training, else 0.0."""
    return 1.0 if (date.year, date.month) in RAMADAN_YEAR_MONTHS else 0.0


def is_business_day_eatstedi(date: pd.Timestamp) -> bool:
    """Hari aktif Eatstedi: bukan weekend, bukan Januari/Juli."""
    if date.weekday() >= 5:
        return False
    if date.month in EATSTEDI_CLOSED_MONTHS:
        return False
    return True
