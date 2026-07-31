# Planning Improve — 28 Juli 2026

Dokumen ini mendata kesalahan, inkonsistensi, dan hal-hal *out of scope* yang ditemukan pada pipeline Model LSTM (`main.ipynb`), API produksi (`app.py`, `app_public.py`, `v2/`), backtest, dan subsistem TimeGAN — beserta rencana penyederhanaan.

Konteks dasar (sudah diketahui, lihat `CLAUDE.md`): repo ini punya dua jalur terpisah — (1) pipeline riset `main.ipynb` yang benar-benar melatih LSTM, dan (2) API produksi v1 (`app.py`/`app_public.py`) yang **tidak** memuat model LSTM sama sekali (hanya baseline statistik), sementara v2 (`v2/*`) memuat `Models/lstm_daily.keras` tapi saat ini selalu fallback ke `seasonal_naive` karena verdict backtest **TIDAK LULUS**.

---

## 1. Bug Korektnes (prioritas tertinggi)

### 1.1 `is_ramadan` dihitung beda antara training dan serving
- **Training** (`Scripts/generate_notebook.py:302-305`, sumber `daily_sales.csv` yang dipakai `main.ipynb`): `is_ramadan` di-gate per tahun — hanya `(2025, Maret)` dan `(2026, Maret)`.
- **`app.py`** meniru definisi ini dengan benar (baris 569, 584).
- **`v2/lstm_adapter.py:82`** dan **`evaluation/backtest.py:100`** malah memakai heuristik umum `month == 3` untuk **tahun berapa pun**.
- **Dampak**: fitur input LSTM saat inferensi (v2) dan saat backtest berbeda dari fitur saat training untuk setiap fold/prediksi yang jatuh di bulan Maret. Ini bukan sekadar keterbatasan "kalender Islam tidak tersedia" — definisi training itu sendiri adalah *fixed year-gated fact*, jadi seharusnya bisa disamakan persis seperti di `app.py`. Yang lebih serius: **verdict LULUS/TIDAK LULUS di `Dokumen/M3 - Hasil Backtest.md` yang menjadi gerbang produksi LSTM dihitung dari input yang salah untuk fold bulan Maret** — verdict itu perlu dihitung ulang setelah bug ini diperbaiki.
- **Rencana perbaikan**: samakan `is_ramadan` di `v2/lstm_adapter.py` dan `evaluation/backtest.py` dengan definisi year-gated di `app.py`/`generate_notebook.py`. Idealnya faktor keluarkan ke satu fungsi util (`utils/features.py` atau sejenis) yang dipanggil oleh ketiga tempat, supaya tidak bisa drift lagi. Lalu **jalankan ulang `evaluation/backtest.py`** dan perbarui `Dokumen/M3 - Hasil Backtest.md`.

### 1.2 `app_public.py` hardcode `is_ramadan = 0`
- Baris 528, 543: selalu 0, tidak pernah mengikuti definisi training.
- **Dampak saat ini**: tidak berbahaya karena `app_public.py` tidak pernah memanggil model LSTM. Tapi ini bom waktu laten jika endpoint ini nanti disambungkan ke LSTM.
- **Rencana**: perbaiki sekalian saat mengerjakan 1.1 agar konsisten, atau beri komentar eksplisit bahwa field ini sengaja diabaikan untuk tenant generik (karena Ramadan hanya relevan untuk kalender Eatstedi yang sudah diketahui, bukan multi-tenant umum).

---

## 2. Inkonsistensi Train/Serve

### 2.1 Logika hari-bisnis (business day) terduplikasi 4 kali dengan default berbeda
- `main.ipynb` (cell forecast_daily): hardcode `weekday()<5 and month not in [1,7]`.
- `app.py:117-125` (`is_business_day`): duplikat aturan yang sama, hardcode untuk Eatstedi.
- `app_public.py:24-25,111-117`: parametrized, tapi **default terbalik** — `OPEN_ON_WEEKENDS_DEFAULT=True`, `CLOSED_MONTHS_DEFAULT=[]` (asumsi toko buka 7 hari, tidak ada bulan libur).
- `v2/forecast_service.py:94-117` (`_make_future_date_generator`): ambil `open_weekdays`/`closed_months` sepenuhnya dari payload klien, tanpa fallback ke kalender Eatstedi (weekend+Jan/Jul tutup) sama sekali.
- **Dampak**: tidak ada satu sumber kebenaran untuk "kapan toko buka." Rollout rekursif LSTM di v2 bisa menghasilkan tanggal masa depan yang polanya berbeda dari pola training (yang selalu skip weekend & Jan/Jul), sehingga fitur siklikal (`week_sin`, `month_sin`, dst.) yang dibangun dari tanggal tersebut bisa keluar dari distribusi training.
- **Rencana**: buat satu fungsi `get_business_day_calendar(store_profile=None)` yang default ke aturan Eatstedi (weekday<5, month not in [1,7]) bila `store_profile` tidak diberikan, dipakai konsisten oleh `main.ipynb` (via ekspor Python module bila memungkinkan), `app.py`, dan `v2/forecast_service.py`. `app_public.py` boleh override lewat parameter request, tapi default-nya harus align dengan apa yang dilatih model, bukan asumsi generik.

### 2.2 Dokumentasi (`Dokumen/Planning.md`) basi vs. `main.ipynb`
- Planning.md menyebut weekly look_back=8/horizon=4 dan monthly look_back=6/horizon=3, serta tidak mendokumentasikan fitur `active_days`.
- Aktual di `main.ipynb`: `FEATURES_W = FEATURES_M = ['revenue','transactions','qty_sold','active_days']`, `LOOK_BACK_W=5, HORIZON_W=4`, `LOOK_BACK_M=3, HORIZON_M=3`.
- **Rencana**: update `Dokumen/Planning.md` §weekly/monthly agar match tabel di `CLAUDE.md`/`main.ipynb`. Tidak berdampak runtime, tapi penting untuk laporan skripsi agar konsisten dengan model yang benar-benar dilatih.

---

## 3. Out of Scope / Scope Creep

Tujuan skripsi ini adalah **prediksi penjualan berbasis LSTM** (pendapatan & jumlah produk terjual). Beberapa bagian API v2 melampaui itu dan berisiko membuat pembaca (atau penguji) mengira semuanya berasal dari model LSTM, padahal tidak:

### 3.1 `product_demand` (`v2/forecast_service.py:136-153`) bukan hasil model
```python
predicted_qty = avg_daily
predicted_qty_week = avg_daily * 7
recommended_qty = predicted_qty_week * 1.15
```
Ini ekstrapolasi linear dari rata-rata yang dikirim klien (`avg_daily_qty`), tidak melibatkan LSTM atau model time-series apa pun per produk. Ditempatkan di response JSON yang sama dengan `daily.revenue` (yang benar-benar dari LSTM/baseline), sehingga membingungkan mana yang "model" dan mana yang "heuristik biasa."
- **Rencana**: beri label eksplisit di response (`"method": "heuristic"` vs `"method": "lstm"`) atau pisahkan ke bagian dokumentasi yang jelas menyatakan fitur ini di luar cakupan model LSTM skripsi — murni fitur pendukung produk POS.

### 3.2 `recommendations` (`v2/forecast_service.py:176-206`) adalah logika bisnis, bukan forecasting
`target_omzet` (rata-rata 15 hari aktif terakhir × 1.10/1.25) dan `happy_hour` (diskon hardcode 15%, jam tersepi) murni heuristik agregat, tidak berhubungan dengan LSTM.
- **Rencana**: sama seperti 3.1 — pertahankan sebagai fitur produk (berguna untuk POS Flutter client), tapi tegaskan dalam dokumen skripsi bahwa ini bukan bagian dari kontribusi model, supaya evaluasi akademik fokus ke akurasi forecast revenue/qty saja.

### 3.3 Subsistem TimeGAN tidak terpakai dan tidak terbukti lebih baik
- Tidak ada referensi ke `Models/TimeGAN/*` atau `TimeGAN.py` di `v2/`, `app.py`, `app_public.py`, maupun `evaluation/backtest.py` — murni eksperimen berdiri sendiri.
- `Models/TimeGAN/comparison_results.csv` menunjukkan model augmented mengungguli LSTM baseline murni, tapi **masih kalah dari Seasonal-Naive-5** pada MASE (2.51 vs 2.14) — mengulang pola yang sama seperti LSTM v2 (TIDAK LULUS vs seasonal_naive).
- **Rencana**: dua opsi — (a) jika ingin dipertahankan sebagai bab eksperimen di skripsi, tulis eksplisit sebagai *rejected/inconclusive branch* dengan angka di atas sebagai bukti, jangan diklaim sebagai peningkatan; (b) jika tidak akan dibahas di laporan akhir, pertimbangkan untuk tidak menjalankannya lagi / keluarkan dari ruang lingkup utama agar tidak menambah kompleksitas dependency (lihat 3.4).

### 3.4 Dependency `torch` hanya untuk eksperimen yang tidak dipakai
- `requirements.txt` menarik wheel CPU PyTorch penuh, padahal satu-satunya pemakai adalah `Data/TimeGAN/TimeGAN.py`. Semua jalur serving (`app.py`, `app_public.py`, `v2/*`, `evaluation/backtest.py`) hanya pakai TensorFlow/Keras.
- **Rencana**: pindahkan `torch` ke `requirements-timegan.txt` (opsional) terpisah dari `requirements.txt` inti, supaya instalasi API produksi lebih ringan dan jelas scope-nya.

---

## 4. Ringkasan Prioritas Perbaikan

| # | Item | Tipe | Aksi |
|---|---|---|---|
| 1 | `is_ramadan` beda training vs `lstm_adapter.py`/`backtest.py` | Bug | Samakan definisi year-gated, re-run backtest, update verdict di M3 doc |
| 2 | `app_public.py` hardcode `is_ramadan=0` | Bug laten | Perbaiki bersamaan dengan #1, atau beri komentar penjelas |
| 3 | 4x duplikasi logika hari-bisnis dgn default beda | Inkonsistensi | Satukan jadi satu fungsi/util, default ikut kalender training Eatstedi |
| 4 | Planning.md weekly/monthly stale | Dokumentasi | Update angka agar match `main.ipynb` |
| 5 | `product_demand` heuristik dikira hasil model | Scope creep | Tandai `method: heuristic` di response / pisahkan di dokumen |
| 6 | `recommendations` (promo/target) heuristik bisnis | Scope creep | Sama seperti #5, tegaskan di luar kontribusi model |
| 7 | Subsistem TimeGAN tak terpakai, tak menang vs seasonal-naive | Scope creep | Nyatakan eksplisit sebagai eksperimen gagal, atau keluarkan dari scope utama |
| 8 | `torch` di requirements inti hanya utk TimeGAN | Cruft | Pisah ke requirements opsional |

**Rekomendasi urutan pengerjaan**: #1 → #3 → re-run backtest & update dokumen M3 → #4 (dokumentasi) → #5/#6 (kejelasan scope response API) → #7/#8 (beres-beres, bisa terakhir karena tidak mempengaruhi correctness).

Setelah #1 dan #3 diperbaiki dan backtest dijalankan ulang, ada kemungkinan nyata verdict berubah dari TIDAK LULUS — ini perlu dicatat sebagai temuan penting dalam skripsi, bukan sekadar bug fix administratif.
