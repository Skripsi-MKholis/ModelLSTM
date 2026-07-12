# About — Fitur API Prediksi Penjualan Eatstedi

Dokumen ini merangkum fitur yang disediakan oleh API (`app.py`, port 5000) beserta contoh input dan output-nya.

> **Catatan penting:** Endpoint API **tidak** memuat model LSTM (`Models/*.keras`). API menyajikan baseline statistik yang dihitung langsung dari data historis — Seasonal-Naive (k=4) untuk harian, Mean-4 untuk mingguan, Mean-3 untuk bulanan — karena pada dataset kecil ini baseline tersebut terbukti kompetitif dengan LSTM (lihat `Models/TimeGAN/comparison_results.csv`). Model LSTM dilatih dan dievaluasi di `main.ipynb`.

---

## 1. Cek Status API — `GET /api/status`

Memeriksa kesehatan API dan ketersediaan file data (daily/weekly/monthly/raw transactions).

**Contoh Input:**
```bash
curl http://localhost:5000/api/status
```

**Contoh Output:**
```json
{
  "status": "online",
  "timestamp": "2026-07-13T10:30:00.000000",
  "database_status": {
    "daily_sales_exists": true,
    "weekly_sales_exists": true,
    "monthly_sales_exists": true,
    "raw_transactions_exists": true,
    "daily_records_count": 316,
    "last_record_date": "2026-06-20"
  }
}
```

---

## 2. Prediksi Penjualan Harian — `GET/POST /api/predict/daily`

Meramalkan omzet harian ke depan menggunakan **Seasonal-Naive (k=4)**: rata-rata 4 kemunculan terakhir hari-yang-sama (Senin dgn Senin, dst.), plus pembanding **Naive** (omzet hari kerja terakhir). Tanggal prediksi otomatis melewati hari tutup (Sabtu/Minggu dan bulan Januari/Juli).

**Parameter (JSON, opsional):**
- `n_days` — jumlah hari prediksi (default 7)
- `k` — jumlah riwayat hari-sama yang dirata-rata (default 4)
- `history` — data transaksi terbaru dari POS (opsional, menggantikan CSV)

**Contoh Input:**
```bash
curl -X POST http://localhost:5000/api/predict/daily \
  -H "Content-Type: application/json" \
  -d '{"n_days": 3, "k": 4}'
```

**Contoh Output:**
```json
{
  "metadata": {
    "model_used": "Seasonal-Naive (k=4) - Paling Stabil",
    "baseline_model": "Naive (Kemarin)",
    "historical_last_date": "2026-06-19",
    "historical_last_revenue": 1450000.0,
    "n_days_forecasted": 3,
    "k_seasons": 4
  },
  "predictions": [
    { "date": "2026-06-22", "day": "Monday",    "predicted_revenue_seasonal_naive": 1520000, "predicted_revenue_naive": 1450000 },
    { "date": "2026-06-23", "day": "Tuesday",   "predicted_revenue_seasonal_naive": 1480000, "predicted_revenue_naive": 1450000 },
    { "date": "2026-06-24", "day": "Wednesday", "predicted_revenue_seasonal_naive": 1395000, "predicted_revenue_naive": 1450000 }
  ],
  "summary": {
    "total_predicted_revenue_seasonal_naive": 4395000,
    "average_predicted_revenue_seasonal_naive": 1465000,
    "total_predicted_revenue_naive": 4350000,
    "average_predicted_revenue_naive": 1450000
  }
}
```

---

## 3. Prediksi Penjualan Mingguan — `GET/POST /api/predict/weekly`

Meramalkan omzet mingguan menggunakan **Mean-4** (rata-rata 4 minggu terakhir) plus pembanding Naive.

**Parameter (JSON, opsional):** `n_weeks` (default 4), `history`.

**Contoh Input:**
```bash
curl -X POST http://localhost:5000/api/predict/weekly \
  -H "Content-Type: application/json" \
  -d '{"n_weeks": 2}'
```

**Contoh Output:**
```json
{
  "metadata": { "model_used": "Mean-4 Weekly Average", "baseline_model": "Naive Weekly" },
  "predictions": [
    { "week_index": 1, "predicted_revenue_mean4": 7250000, "predicted_revenue_naive": 7100000 },
    { "week_index": 2, "predicted_revenue_mean4": 7250000, "predicted_revenue_naive": 7100000 }
  ],
  "summary": {
    "total_predicted_revenue_mean4": 14500000,
    "total_predicted_revenue_naive": 14200000
  }
}
```

---

## 4. Prediksi Penjualan Bulanan — `GET/POST /api/predict/monthly`

Meramalkan omzet bulanan menggunakan **Mean-3** (rata-rata 3 bulan terakhir) plus pembanding Naive.

**Parameter (JSON, opsional):** `n_months` (default 3), `history`.

**Contoh Input:**
```bash
curl -X POST http://localhost:5000/api/predict/monthly \
  -H "Content-Type: application/json" \
  -d '{"n_months": 2}'
```

**Contoh Output:**
```json
{
  "metadata": { "model_used": "Mean-3 Monthly Average", "baseline_model": "Naive Monthly" },
  "predictions": [
    { "month_index": 1, "predicted_revenue_mean3": 29500000, "predicted_revenue_naive": 30200000 },
    { "month_index": 2, "predicted_revenue_mean3": 29500000, "predicted_revenue_naive": 30200000 }
  ],
  "summary": {
    "total_predicted_revenue_mean3": 59000000,
    "total_predicted_revenue_naive": 60400000
  }
}
```

---

## 5. Rekomendasi Stok — `GET/POST /api/recommendations/stock`

Menghitung alokasi anggaran belanja per kategori dan rekomendasi kuantitas stok top produk berdasarkan proyeksi omzet harian. Jika `predicted_revenue` tidak diberikan, dihitung otomatis dengan Seasonal-Naive. Termasuk safety buffer +15% dan peringatan khusus untuk kategori MAKANAN BASAH (perishable).

**Parameter (JSON, opsional):** `predicted_revenue`, `target_date` (default hari kerja besok).

**Contoh Input:**
```bash
curl -X POST http://localhost:5000/api/recommendations/stock \
  -H "Content-Type: application/json" \
  -d '{"predicted_revenue": 1500000}'
```

**Contoh Output (dipersingkat):**
```json
{
  "target_date": "2026-07-14",
  "predicted_revenue_base": 1500000,
  "category_allocations": [
    { "category": "MAKANAN BASAH", "revenue_share": 0.45, "allocated_budget_rupiah": 675000 },
    { "category": "MINUMAN",       "revenue_share": 0.25, "allocated_budget_rupiah": 375000 }
  ],
  "product_recommendations": [
    {
      "product_name": "Nasi Ayam Geprek",
      "category": "MAKANAN BASAH",
      "unit_price_rupiah": 10000,
      "qty_share": 0.08,
      "recommended_stock_qty": 38,
      "recommended_stock_with_safety_buffer": 44,
      "handling_instruction": "PERISHABLE! Segera habiskan/jangan simpan melebihi 24 jam"
    }
  ]
}
```

---

## 6. Rekomendasi Target Omzet — `GET/POST /api/recommendations/target`

Memberikan tiga level target penjualan berbasis statistik 15 hari aktif terakhir: **konservatif** (rata-rata), **moderat** (rata-rata × faktor), **agresif** (rata-rata + 1,5 × standar deviasi, dibatasi maksimum historis).

**Parameter (JSON, opsional):** `factor` — faktor kenaikan target moderat (default 1.10 = +10%).

**Contoh Input:**
```bash
curl -X POST http://localhost:5000/api/recommendations/target \
  -H "Content-Type: application/json" \
  -d '{"factor": 1.15}'
```

**Contoh Output:**
```json
{
  "historical_basis_15_active_days": {
    "average_revenue": 1480000,
    "std_deviation": 210000,
    "max_revenue": 1900000
  },
  "targets": {
    "konservatif_target": 1480000,
    "moderat_target": 1702000,
    "agresif_target": 1795000
  },
  "status_pesan": {
    "konservatif_pesan": "Cocok untuk hari biasa/sepi (misal: masa-masa ujian awal).",
    "moderat_pesan": "Target pertumbuhan standar (+15%). Direkomendasikan untuk target operasional harian.",
    "agresif_pesan": "Cocok untuk masa-masa ramai (seperti awal masuk semester/welcome week)."
  }
}
```

---

## 7. Pencatatan Penjualan Baru — `POST /api/sales/record`

Digunakan POS untuk mengirim rekap transaksi harian (lunas) secara real-time. Data di-*upsert* ke `Data/Ekstrak/daily_sales.csv` (jika tanggal sudah ada, baris diperbarui; jika belum, ditambahkan) beserta kolom turunan (`day_of_week`, `is_weekend`, `is_holiday`, `is_ramadan`, dll.).

**Parameter (JSON, wajib semua):** `date` (YYYY-MM-DD), `revenue`, `transactions`, `qty_sold`.

**Contoh Input:**
```bash
curl -X POST http://localhost:5000/api/sales/record \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-06-20", "revenue": 1620000, "transactions": 145, "qty_sold": 510}'
```

**Contoh Output:**
```json
{
  "message": "Data tanggal 2026-06-20 berhasil ditambahkan.",
  "recorded_data": {
    "date": "2026-06-20",
    "revenue": 1620000,
    "transactions": 145,
    "qty_sold": 510
  }
}
```

---

## Fitur Pendukung Lainnya

- **`GET /`** — halaman indeks berisi daftar endpoint (mencegah 404).
- **`app_public.py`** — varian generik/multi-tenant dari API yang sama (toko buka 7 hari, bulan tutup dapat dikonfigurasi per-request).
- **Model LSTM (riset, `main.ipynb`)** — 3 model LSTM 2-layer (harian look_back=14, mingguan=5, bulanan=3) dilatih pada hari aktif saja; hasil tersimpan di `Models/*.keras`.
- **Augmentasi TimeGAN (`Data/TimeGAN/TimeGAN.py`)** — sintesis data harian tambahan (200 sekuens × 24 hari) untuk mengatasi keterbatasan data.

Referensi lengkap: `Dokumen/API Documentation.md` dan `Dokumen/API Documentation Public.md`.
