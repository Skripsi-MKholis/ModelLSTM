# API Documentation - POS Integration Eatstedi

Dokumen ini menjelaskan spesifikasi antarmuka pemrograman (API) yang disediakan oleh [app.py](file:///e:/SKRIPSI/Kholis/ModelLSTM/app.py) untuk mengintegrasikan sistem Kasir/POS Eatstedi dengan fitur prediksi penjualan dan sistem rekomendasi berbasis data historis.

* **Base URL**: `http://localhost:5000` (atau IP server Python Anda)
* **Format Request/Response**: `application/json`

---

## Daftar Endpoint

1. [Cek Status API (`GET /api/status`)](#1-cek-status-api-get-apistatus)
2. [Prediksi Penjualan Harian (`POST /api/predict/daily`)](#2-prediksi-penjualan-harian-post-apipredictdaily)
3. [Prediksi Penjualan Mingguan (`POST /api/predict/weekly`)](#3-prediksi-penjualan-mingguan-post-apipredictweekly)
4. [Prediksi Penjualan Bulanan (`POST /api/predict/monthly`)](#4-prediksi-penjualan-bulanan-post-apipredictmonthly)
5. [Rekomendasi Stok Kategori & Produk (`POST /api/recommendations/stock`)](#5-rekomendasi-stok-kategori--produk-post-apirecommendationsstock)
6. [Rekomendasi Target Omzet (`POST /api/recommendations/target`)](#6-rekomendasi-target-omzet-post-apirecommendationstarget)
7. [Pencatatan Penjualan Baru (`POST /api/sales/record`)](#7-pencatatan-penjualan-baru-post-apisalesrecord)

---

### 1. Cek Status API (`GET /api/status`)
Mengecek kesehatan sistem API dan status database/CSV lokal di server Python.

* **Method**: `GET`
* **URL Path**: `/api/status`
* **Headers**: `Accept: application/json`

#### Contoh Response (200 OK)
```json
{
  "database_status": {
    "daily_records_count": 313,
    "daily_sales_exists": true,
    "last_record_date": "2026-06-20",
    "monthly_sales_exists": true,
    "raw_transactions_exists": true,
    "weekly_sales_exists": true
  },
  "status": "online",
  "timestamp": "2026-06-24T03:30:34.333003"
}
```

---

### 2. Prediksi Penjualan Harian (`POST /api/predict/daily`)
Memperkirakan omzet penjualan harian untuk beberapa hari aktif bisnis ke depan. Sistem otomatis melewati akhir pekan (Sabtu/Minggu) dan bulan libur semester (Januari dan Juli).

* **Method**: `POST` (Mendukung `GET` tanpa payload)
* **URL Path**: `/api/predict/daily`
* **Request Body (JSON)**:
  * `n_days` (integer, opsional): Jumlah hari aktif ke depan yang akan diprediksi. Default: `7`.
  * `k` (integer, opsional): Jumlah periode historis hari-yang-sama untuk rata-rata Seasonal-Naive. Default: `4`.
  * `history` (array of objects, opsional): Data transaksi eksternal tambahan.

#### Contoh Request
```json
{
  "n_days": 5,
  "k": 4
}
```

#### Contoh Response (200 OK)
```json
{
  "metadata": {
    "baseline_model": "Naive (Kemarin)",
    "historical_last_date": "2026-06-20",
    "historical_last_revenue": 68500.0,
    "k_seasons": 4,
    "model_used": "Seasonal-Naive (k=4) - Paling Stabil",
    "n_days_forecasted": 5
  },
  "predictions": [
    {
      "date": "2026-06-22",
      "day": "Monday",
      "predicted_revenue_naive": 68500,
      "predicted_revenue_seasonal_naive": 2051500
    },
    {
      "date": "2026-06-23",
      "day": "Tuesday",
      "predicted_revenue_naive": 68500,
      "predicted_revenue_seasonal_naive": 1941250
    },
    {
      "date": "2026-06-24",
      "day": "Wednesday",
      "predicted_revenue_naive": 68500,
      "predicted_revenue_seasonal_naive": 2038750
    },
    {
      "date": "2026-06-25",
      "day": "Thursday",
      "predicted_revenue_naive": 68500,
      "predicted_revenue_seasonal_naive": 1974750
    },
    {
      "date": "2026-06-26",
      "day": "Friday",
      "predicted_revenue_naive": 68500,
      "predicted_revenue_seasonal_naive": 1980875
    }
  ],
  "summary": {
    "average_predicted_revenue_naive": 68500,
    "average_predicted_revenue_seasonal_naive": 1997425,
    "total_predicted_revenue_naive": 342500,
    "total_predicted_revenue_seasonal_naive": 9987125
  }
}
```

---

### 3. Prediksi Penjualan Mingguan (`POST /api/predict/weekly`)
Memperkirakan penjualan mingguan untuk $N$ minggu ke depan berdasarkan rata-rata pergerakan mingguan (Mean-4).

* **Method**: `POST` (Mendukung `GET` tanpa payload)
* **URL Path**: `/api/predict/weekly`
* **Request Body (JSON)**:
  * `n_weeks` (integer, opsional): Jumlah minggu ke depan. Default: `4`.

#### Contoh Request
```json
{
  "n_weeks": 2
}
```

#### Contoh Response (200 OK)
```json
{
  "metadata": {
    "baseline_model": "Naive Weekly",
    "model_used": "Mean-4 Weekly Average"
  },
  "predictions": [
    {
      "predicted_revenue_mean4": 7248125,
      "predicted_revenue_naive": 5781000,
      "week_index": 1
    },
    {
      "predicted_revenue_mean4": 7248125,
      "predicted_revenue_naive": 5781000,
      "week_index": 2
    }
  ],
  "summary": {
    "total_predicted_revenue_mean4": 14496250,
    "total_predicted_revenue_naive": 11562000
  }
}
```

---

### 4. Prediksi Penjualan Bulanan (`POST /api/predict/monthly`)
Memperkirakan total penjualan bulanan untuk beberapa bulan ke depan.

* **Method**: `POST` (Mendukung `GET` tanpa payload)
* **URL Path**: `/api/predict/monthly`
* **Request Body (JSON)**:
  * `n_months` (integer, opsional): Jumlah bulan ke depan. Default: `3`.

#### Contoh Request
```json
{
  "n_months": 2
}
```

#### Contoh Response (200 OK)
```json
{
  "metadata": {
    "baseline_model": "Naive Monthly",
    "model_used": "Mean-3 Monthly Average"
  },
  "predictions": [
    {
      "month_index": 1,
      "predicted_revenue_mean3": 35487833,
      "predicted_revenue_naive": 24621500
    },
    {
      "month_index": 2,
      "predicted_revenue_mean3": 35487833,
      "predicted_revenue_naive": 24621500
    }
  ],
  "summary": {
    "total_predicted_revenue_mean3": 70975667,
    "total_predicted_revenue_naive": 49243000
  }
}
```

---

### 5. Rekomendasi Stok Kategori & Produk (`POST /api/recommendations/stock`)
Mengalokasikan anggaran/omzet harian ke setiap kategori berdasarkan performa historis, serta menyajikan rekomendasi kuantitas stok produk terlaris (termasuk *safety stock buffer* 15%).

* **Method**: `POST` (Mendukung `GET` tanpa payload)
* **URL Path**: `/api/recommendations/stock`
* **Request Body (JSON)**:
  * `predicted_revenue` (float, opsional): Target/prediksi omzet harian. Jika dikosongkan, API akan memprediksi omzet hari kerja berikutnya secara otomatis menggunakan Seasonal-Naive.
  * `target_date` (string, format `YYYY-MM-DD`, opsional): Tanggal yang ingin diproyeksikan (jika `predicted_revenue` kosong).

#### Contoh Request
```json
{
  "predicted_revenue": 2051500
}
```

#### Contoh Response (200 OK)
```json
{
  "target_date": "2026-06-22",
  "predicted_revenue_base": 2051500,
  "category_allocations": [
    {
      "category": "MAKANAN BASAH",
      "revenue_share": 0.4842,
      "allocated_budget_rupiah": 993336
    },
    {
      "category": "MINUMAN",
      "revenue_share": 0.1638,
      "allocated_budget_rupiah": 336036
    },
    {
      "category": "MAKANAN KERING",
      "revenue_share": 0.0837,
      "allocated_budget_rupiah": 171711
    },
    {
      "category": "ICE CREAM",
      "revenue_share": 0.0674,
      "allocated_budget_rupiah": 138271
    },
    {
      "category": "SNACKS",
      "revenue_share": 0.0532,
      "allocated_budget_rupiah": 109140
    },
    {
      "category": "MERCH",
      "revenue_share": 0.001,
      "allocated_budget_rupiah": 2052
    }
  ],
  "product_recommendations": [
    {
      "product_name": "TAHU BAKSO",
      "category": "MAKANAN BASAH",
      "unit_price_rupiah": 2500,
      "qty_share": 0.153,
      "recommended_stock_qty": 115,
      "recommended_stock_with_safety_buffer": 132,
      "handling_instruction": "PERISHABLE! Segera habiskan/jangan simpan melebihi 24 jam"
    },
    {
      "product_name": "TAHU BAKSO BALADO",
      "category": "MAKANAN BASAH",
      "unit_price_rupiah": 2518,
      "qty_share": 0.111,
      "recommended_stock_qty": 83,
      "recommended_stock_with_safety_buffer": 95,
      "handling_instruction": "PERISHABLE! Segera habiskan/jangan simpan melebihi 24 jam"
    },
    {
      "product_name": "RISOL MAYO & SOSIS",
      "category": "MAKANAN BASAH",
      "unit_price_rupiah": 2558,
      "qty_share": 0.056,
      "recommended_stock_qty": 42,
      "recommended_stock_with_safety_buffer": 48,
      "handling_instruction": "PERISHABLE! Segera habiskan/jangan simpan melebihi 24 jam"
    }
  ]
}
```

---

### 6. Rekomendasi Target Omzet (`POST /api/recommendations/target`)
Memberikan target target omzet harian berbasis statistik historis 15 hari kerja aktif terakhir.

* **Method**: `POST` (Mendukung `GET` tanpa payload)
* **URL Path**: `/api/recommendations/target`
* **Request Body (JSON)**:
  * `factor` (float, opsional): Faktor pertumbuhan untuk target moderat. Default: `1.10` (+10%).

#### Contoh Request
```json
{
  "factor": 1.15
}
```

#### Contoh Response (200 OK)
```json
{
  "historical_basis_15_active_days": {
    "average_revenue": 1658867,
    "max_revenue": 3254000,
    "std_deviation": 869260
  },
  "status_pesan": {
    "agresif_pesan": "Cocok untuk masa-masa ramai (seperti awal masuk semester/welcome week).",
    "konservatif_pesan": "Cocok untuk hari biasa/sepi (misal: masa-masa ujian awal).",
    "moderat_pesan": "Target pertumbuhan standar (+14%). Direkomendasikan untuk target operasional harian."
  },
  "targets": {
    "agresif_target": 2962757,
    "konservatif_target": 1658867,
    "moderat_target": 1907697
  }
}
```

---

### 7. Pencatatan Penjualan Baru (`POST /api/sales/record`)
Digunakan oleh aplikasi POS Eatstedi untuk melaporkan rekap harian setelah kasir tutup buku (omzet harian, transaksi harian, dan kuantitas terjual). Data disimpan ke database CSV agar prediksi hari berikutnya selalu aktual.

* **Method**: `POST`
* **URL Path**: `/api/sales/record`
* **Headers**: `Content-Type: application/json`
* **Request Body (JSON)**:
  * `date` (string, format `YYYY-MM-DD`, **Wajib**): Tanggal rekap penjualan.
  * `revenue` (float, **Wajib**): Total omzet harian.
  * `transactions` (integer, **Wajib**): Total invoice sukses/lunas.
  * `qty_sold` (integer, **Wajib**): Total unit barang terjual.

#### Contoh Request
```json
{
  "date": "2026-06-21",
  "revenue": 1650000.0,
  "transactions": 210,
  "qty_sold": 490
}
```

#### Contoh Response (200 OK)
```json
{
  "message": "Data tanggal 2026-06-21 berhasil ditambahkan.",
  "recorded_data": {
    "date": "2026-06-21",
    "qty_sold": 490,
    "revenue": 1650000.0,
    "transactions": 210
  }
}
```

---

## Panduan Integrasi (Sisi POS Laravel)

Jika aplikasi POS Eatstedi Anda dibangun dengan **Laravel**, berikut adalah contoh kode *Controller* untuk mengambil data rekomendasi persediaan barang (*stock*) dari API Python:

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class StockRecommendationController extends Controller
{
    /**
     * Menampilkan rekomendasi stok ke Dashboard POS Owner.
     */
    public function index()
    {
        $apiUrl = 'http://localhost:5000/api/recommendations/stock';

        try {
            // Memanggil API Python menggunakan HTTP Client Laravel
            $response = Http::timeout(5)->post($apiUrl, [
                // Opsional: kirim parameter jika diperlukan
            ]);

            if ($response->successful()) {
                $data = $response->json();
                
                return view('owner.stock_recommendations', [
                    'targetDate' => $data['target_date'],
                    'baseRevenue' => $data['predicted_revenue_base'],
                    'categories' => $data['category_allocations'],
                    'products' => $data['product_recommendations']
                ]);
            }

            return back()->with('error', 'API Prediksi gagal mengembalikan data.');

        } catch (\Exception $e) {
            return back()->with('error', 'Koneksi ke API Prediksi gagal: ' . $e->getMessage());
        }
    }
}
```
