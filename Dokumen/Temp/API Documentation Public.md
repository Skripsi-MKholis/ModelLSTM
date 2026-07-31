# API Documentation - POS Public Integration (Generic & Stateless)

Dokumen ini menjelaskan spesifikasi antarmuka pemrograman (API) publik yang disediakan oleh [app_public.py](file:///e:/SKRIPSI/Kholis/ModelLSTM/app_public.py). API ini didesain secara ***stateless* (tanpa status)** agar dapat diintegrasikan dengan database eksternal seperti **Supabase** secara aman dan fleksibel melalui payload request HTTP (Cara 1).

* **Base URL**: `http://localhost:5000` (atau IP server Python Anda)
* **Format Request/Response**: `application/json`

---

## Fitur Unggulan API Publik
1. **Stateless Parameter-Driven**: Semua data historis transaksi dapat dikirimkan langsung sebagai parameter request (`history`, `category_shares`, `top_products`). Server Python tidak perlu terhubung langsung ke database Supabase Anda.
2. **Dynamic Calendar Settings**: Konfigurasi operasional akhir pekan (`open_on_weekends`) dan bulan libur (`closed_months`) dapat ditentukan dinamis per request.

---

## Daftar Endpoint

1. [Cek Status API (`GET /api/status`)](#1-cek-status-api-get-apistatus)
2. [Prediksi Penjualan Harian Dinamis (`POST /api/predict/daily`)](#2-prediksi-penjualan-harian-dinamis-post-apipredictdaily)
3. [Rekomendasi Stok Produk Dinamis (`POST /api/recommendations/stock`)](#3-rekomendasi-stok-produk-dinamis-post-apirecommendationsstock)
4. [Rekomendasi Target Omzet Dinamis (`POST /api/recommendations/target`)](#4-rekomendasi-target-omzet-dinamis-post-apirecommendationstarget)

---

### 1. Cek Status API (`GET /api/status`)
Mendapatkan konfigurasi default server dan status file database lokal.

* **Method**: `GET`
* **URL Path**: `/api/status`

#### Contoh Response (200 OK)
```json
{
  "database_status": {
    "daily_records_count": 314,
    "daily_sales_exists": true,
    "last_record_date": "2026-06-21",
    "monthly_sales_exists": true,
    "raw_transactions_exists": true,
    "weekly_sales_exists": true
  },
  "default_config": {
    "closed_months": [],
    "open_on_weekends": true
  },
  "scope": "public_generic",
  "status": "online",
  "timestamp": "2026-06-24T04:27:55.019590"
}
```

---

### 2. Prediksi Penjualan Harian Dinamis (`POST /api/predict/daily`)
Memprediksi omzet penjualan harian berdasarkan riwayat transaksi harian dari Supabase yang dikirimkan via parameter payload.

* **Method**: `POST`
* **URL Path**: `/api/predict/daily`
* **Request Body (JSON)**:
  * `n_days` (integer, opsional): Jumlah hari aktif ke depan yang akan diprediksi. Default: `7`.
  * `k` (integer, opsional): Jumlah periode historis untuk rata-rata Seasonal-Naive. Default: `4`.
  * `open_on_weekends` (boolean, opsional): `true` jika toko buka di akhir pekan (Sabtu/Minggu). Default: `true`.
  * `closed_months` (array of integers, opsional): Indeks bulan toko tutup (misal `[6, 12]` untuk libur sekolah). Default: `[]`.
  * `history` (array of objects, **Sangat Direkomendasikan untuk Supabase**): List transaksi harian. Setiap objek wajib memiliki kolom:
    * `date` (string, format `YYYY-MM-DD`): Tanggal penjualan.
    * `revenue` (float): Total omzet pada tanggal tersebut.

#### Contoh Request (Integrasi Supabase)
```json
{
  "n_days": 5,
  "k": 4,
  "open_on_weekends": true,
  "closed_months": [],
  "history": [
    {"date": "2026-06-01", "revenue": 1500000},
    {"date": "2026-06-02", "revenue": 1750000},
    {"date": "2026-06-03", "revenue": 1200000},
    {"date": "2026-06-04", "revenue": 1400000},
    {"date": "2026-06-05", "revenue": 1650000},
    {"date": "2026-06-06", "revenue": 1900000},
    {"date": "2026-06-07", "revenue": 950000},
    {"date": "2026-06-08", "revenue": 1300000},
    {"date": "2026-06-09", "revenue": 1450000},
    {"date": "2026-06-10", "revenue": 1550000},
    {"date": "2026-06-11", "revenue": 1700000},
    {"date": "2026-06-12", "revenue": 1800000},
    {"date": "2026-06-13", "revenue": 2100000},
    {"date": "2026-06-14", "revenue": 1100000},
    {"date": "2026-06-15", "revenue": 1600000},
    {"date": "2026-06-16", "revenue": 1550000},
    {"date": "2026-06-17", "revenue": 1480000},
    {"date": "2026-06-18", "revenue": 1720000},
    {"date": "2026-06-19", "revenue": 1950000},
    {"date": "2026-06-20", "revenue": 2200000}
  ]
}
```

#### Contoh Response (200 OK)
```json
{
  "metadata": {
    "baseline_model": "Naive",
    "config_applied": {
      "closed_months": [],
      "open_on_weekends": true
    },
    "historical_last_date": "2026-06-20",
    "model_used": "Seasonal-Naive (k=4) - Stabil",
    "n_days_forecasted": 5
  },
  "predictions": [
    {
      "date": "2026-06-21",
      "day": "Sunday",
      "predicted_revenue_naive": 2200000,
      "predicted_revenue_seasonal_naive": 1025000
    },
    {
      "date": "2026-06-22",
      "day": "Monday",
      "predicted_revenue_naive": 2200000,
      "predicted_revenue_seasonal_naive": 1462500
    },
    {
      "date": "2026-06-23",
      "day": "Tuesday",
      "predicted_revenue_naive": 2200000,
      "predicted_revenue_seasonal_naive": 1587500
    },
    {
      "date": "2026-06-24",
      "day": "Wednesday",
      "predicted_revenue_naive": 2200000,
      "predicted_revenue_seasonal_naive": 1407500
    },
    {
      "date": "2026-06-25",
      "day": "Thursday",
      "predicted_revenue_naive": 2200000,
      "predicted_revenue_seasonal_naive": 1590000
    }
  ],
  "summary": {
    "average_predicted_revenue_naive": 2200000,
    "average_predicted_revenue_seasonal_naive": 1414500,
    "total_predicted_revenue_naive": 11000000,
    "total_predicted_revenue_seasonal_naive": 7072500
  }
}
```

---

### 3. Rekomendasi Stok Produk Dinamis (`POST /api/recommendations/stock`)
Membagi target/prediksi omzet harian ke setiap kategori dan produk secara proporsional berdasarkan data statistik produk yang dikirim langsung dari Supabase.

* **Method**: `POST`
* **URL Path**: `/api/recommendations/stock`
* **Request Body (JSON)**:
  * `predicted_revenue` (float, opsional): Nilai proyeksi omzet harian. Jika kosong, sistem memprediksi otomatis dari `history`.
  * `category_shares` (object, **Wajib untuk Supabase**): Objek berisi key nama kategori dan value persentase kontribusi omzet (nilai 0 s.d 1).
  * `top_products` (array of objects, **Wajib untuk Supabase**): 10 produk terlaris. Setiap objek berisi:
    * `name` (string): Nama produk.
    * `category` (string): Kategori produk.
    * `price` (integer): Harga jual satuan (Rupiah).
    * `qty_share` (float): Persentase kontribusi kuantitas unit produk terhadap total kuantitas penjualan harian (nilai 0 s.d 1).
  * `history` (array of objects, opsional): Digunakan untuk mencari prediksi otomatis jika `predicted_revenue` dikosongkan.

#### Contoh Request (Integrasi Supabase)
```json
{
  "predicted_revenue": 2000000,
  "category_shares": {
    "Makanan Berat": 0.50,
    "Minuman Dingin": 0.30,
    "Snacks": 0.20
  },
  "top_products": [
    {
      "name": "Nasi Ayam Bakar",
      "category": "Makanan Berat",
      "price": 20000,
      "qty_share": 0.25
    },
    {
      "name": "Es Teh Manis",
      "category": "Minuman Dingin",
      "price": 5000,
      "qty_share": 0.40
    },
    {
      "name": "Kentang Goreng",
      "category": "Snacks",
      "price": 10000,
      "qty_share": 0.15
    }
  ]
}
```

#### Contoh Response (200 OK)
```json
{
  "predicted_revenue_base": 2000000,
  "target_date": "2026-06-25",
  "category_allocations": [
    {
      "allocated_budget_rupiah": 1000000,
      "category": "Makanan Berat",
      "revenue_share": 0.5
    },
    {
      "allocated_budget_rupiah": 600000,
      "category": "Minuman Dingin",
      "revenue_share": 0.3
    },
    {
      "allocated_budget_rupiah": 400000,
      "category": "Snacks",
      "revenue_share": 0.2
    }
  ],
  "product_recommendations": [
    {
      "category": "Makanan Berat",
      "product_name": "Nasi Ayam Bakar",
      "qty_share": 0.25,
      "recommended_stock_qty": 43,
      "recommended_stock_with_safety_buffer": 49,
      "unit_price_rupiah": 20000
    },
    {
      "category": "Minuman Dingin",
      "product_name": "Es Teh Manis",
      "qty_share": 0.4,
      "recommended_stock_qty": 69,
      "recommended_stock_with_safety_buffer": 79,
      "unit_price_rupiah": 5000
    },
    {
      "category": "Snacks",
      "product_name": "Kentang Goreng",
      "qty_share": 0.15,
      "recommended_stock_qty": 26,
      "recommended_stock_with_safety_buffer": 30,
      "unit_price_rupiah": 10000
    }
  ]
}
```

---

### 4. Rekomendasi Target Omzet Dinamis (`POST /api/recommendations/target`)
Menghasilkan target omzet berbasis data historis harian dari Supabase.

* **Method**: `POST`
* **URL Path**: `/api/recommendations/target`
* **Request Body (JSON)**:
  * `factor` (float, opsional): Faktor pertumbuhan target moderat. Default: `1.10` (+10%).
  * `history` (array of objects, **Wajib untuk Supabase**): Deret penjualan harian terakhir (disarankan mengirimkan minimal 15 data hari aktif terakhir).

#### Contoh Request
```json
{
  "factor": 1.15,
  "history": [
    {"date": "2026-06-01", "revenue": 1500000},
    {"date": "2026-06-02", "revenue": 1750000},
    {"date": "2026-06-03", "revenue": 1200000},
    {"date": "2026-06-04", "revenue": 1400000},
    {"date": "2026-06-05", "revenue": 1650000},
    {"date": "2026-06-06", "revenue": 1900000},
    {"date": "2026-06-07", "revenue": 950000},
    {"date": "2026-06-08", "revenue": 1300000},
    {"date": "2026-06-09", "revenue": 1450000},
    {"date": "2026-06-10", "revenue": 1550000},
    {"date": "2026-06-11", "revenue": 1700000},
    {"date": "2026-06-12", "revenue": 1800000},
    {"date": "2026-06-13", "revenue": 2100000},
    {"date": "2026-06-14", "revenue": 1100000},
    {"date": "2026-06-15", "revenue": 1600000}
  ]
}
```

#### Contoh Response (200 OK)
```json
{
  "historical_basis_15_active_days": {
    "average_revenue": 1523333,
    "max_revenue": 2100000,
    "std_deviation": 313931
  },
  "targets": {
    "agresif_target": 1994230,
    "konservatif_target": 1523333,
    "moderat_target": 1751833
  }
}
```

---

## Contoh Integrasi PHP Laravel dengan Supabase (Query + API Call)

Berikut adalah contoh lengkap fungsi pada Laravel Controller untuk memanggil endpoint rekomendasi stok menggunakan data dinamis yang ditarik secara real-time dari Supabase:

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;

class DashboardPredictionController extends Controller
{
    public function getRecommendations()
    {
        // 1. Ambil data historis omzet 30 hari aktif terakhir dari Supabase
        $history = DB::table('invoices')
            ->select(DB::raw("DATE(succeeded_at) as date"), DB::raw("SUM(total_price) as revenue"))
            ->where('is_paid', 1)
            ->groupBy('date')
            ->orderBy('date', 'desc')
            ->limit(30)
            ->get()
            ->reverse() // Urutkan kronologis (terlama ke terbaru)
            ->values();

        // 2. Hitung statistik kontribusi kategori dari Supabase
        $totalRevenue = DB::table('product_sold')
            ->join('invoices', 'product_sold.invoice_id', '=', 'invoices.id')
            ->where('invoices.is_paid', 1)
            ->sum(DB::raw('quantity * price'));

        $categoryShares = DB::table('product_sold')
            ->join('invoices', 'product_sold.invoice_id', '=', 'invoices.id')
            ->join('products', 'product_sold.product_id', '=', 'products.id')
            ->join('categories', 'products.category_id', '=', 'categories.id')
            ->where('invoices.is_paid', 1)
            ->select('categories.name as category', DB::raw('SUM(product_sold.quantity * product_sold.price) as revenue'))
            ->groupBy('categories.name')
            ->get()
            ->pluck('revenue', 'category')
            ->map(fn($rev) => $totalRevenue > 0 ? $rev / $totalRevenue : 0);

        // 3. Hitung 10 produk terlaris dari Supabase
        $totalQty = DB::table('product_sold')
            ->join('invoices', 'product_sold.invoice_id', '=', 'invoices.id')
            ->where('invoices.is_paid', 1)
            ->sum('quantity');

        $topProducts = DB::table('product_sold')
            ->join('invoices', 'product_sold.invoice_id', '=', 'invoices.id')
            ->join('products', 'product_sold.product_id', '=', 'products.id')
            ->join('categories', 'products.category_id', '=', 'categories.id')
            ->where('invoices.is_paid', 1)
            ->select(
                'products.name as name',
                'categories.name as category',
                DB::raw('AVG(product_sold.price) as price'),
                DB::raw('SUM(product_sold.quantity) as total_qty')
            )
            ->groupBy('products.name', 'categories.name')
            ->orderBy('total_qty', 'desc')
            ->limit(10)
            ->get()
            ->map(fn($prod) => [
                'name' => $prod->name,
                'category' => $prod->category,
                'price' => (int) round($prod->price),
                'qty_share' => $totalQty > 0 ? $prod->total_qty / $totalQty : 0
            ]);

        // 4. Kirim data ke API Python secara stateless
        $apiUrl = 'http://localhost:5000/api/recommendations/stock';
        
        $response = Http::timeout(10)->post($apiUrl, [
            'open_on_weekends' => true,
            'closed_months' => [],
            'category_shares' => $category_shares,
            'top_products' => $topProducts,
            'history' => $history
        ]);

        if ($response->successful()) {
            return view('owner.predictions_dashboard', $response->json());
        }

        return back()->with('error', 'Gagal mendapatkan data prediksi.');
    }
}
```
