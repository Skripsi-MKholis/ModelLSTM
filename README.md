# Model LSTM Prediksi Penjualan Eatstedi

Proyek skripsi: peramalan omzet penjualan menggunakan **LSTM (Long Short-Term Memory)** untuk **Eatstedi**, kantin di DTEDI UGM. Rentang data 24 Agustus 2024 – 20 Juni 2026 (~316 hari aktif bisnis).

Proyek memprediksi omzet pada tiga granularitas: **harian, mingguan, dan bulanan**, lengkap dengan REST API untuk integrasi ke sistem kasir (POS) dan sistem rekomendasi stok/target.

---

## Dua Track yang Terpisah

Proyek ini terdiri dari dua bagian yang **tidak saling terhubung secara arsitektur** — penting untuk tidak mencampuradukkannya:

1. **Pipeline riset ([main.ipynb](main.ipynb))** — melatih model LSTM sesungguhnya. Menghasilkan `Models/*.keras` dan `Models/scalers/*.pkl`.
2. **API produksi ([app.py](app.py) / [app_public.py](app_public.py))** — melayani prediksi ke POS, tetapi **tidak me-load model LSTM sama sekali**. Endpoint mengembalikan **baseline statistik** yang dihitung on-the-fly (Seasonal-Naive, Mean-4, Mean-3).

Alasannya ada di `Models/TimeGAN/comparison_results.csv`: pada dataset kecil ini, baseline Seasonal-Naive **mengungguli** LSTM (test MAE ~Rp 755rb vs LSTM baseline ~Rp 1,75jt). Jadi API sengaja menyajikan baseline yang stabil dan murah-hitung.

---

## Pipeline Riset LSTM (`main.ipynb`)

Alur: `eatstedi-*.sql` → `raw_transactions.csv` → agregasi harian/mingguan/bulanan → sequence → 3 model LSTM → evaluasi → forecasting. Notebook terbagi menjadi 8 section (0–7).

### Ekstraksi data (Section 1)
- **Parse dump MariaDB ~32MB langsung di Python** (`_parse_sql_tuples`, parser karakter-level buatan sendiri) — tanpa server MySQL.
- Hanya 4 tabel diekstrak: `invoices` (difilter `is_paid=1`), `product_sold`, `products`, `categories`.
- **Setiap tahap cache-guarded**: jika CSV output sudah ada, sel di-skip. Untuk membangun ulang, hapus CSV target.

### Konsep domain inti: filter hari aktif bisnis
Sekitar separuh hari kalender beromzet `revenue=0` (akhir pekan, libur semester Jan/Juli, lesu Ramadan). Pipeline **membuang semua baris `revenue==0`** dan melatih model pada urutan *hari aktif saja* — model belajar "hari aktif → hari aktif berikutnya", bukan hari kalender. Saat forecasting, tanggal masa depan dipetakan kembali ke kalender dengan melewati Sabtu/Minggu dan bulan 1 & 7.

### Konfigurasi model (sumber kebenaran = `main.ipynb`)

| Granularitas | look_back | horizon | split | catatan |
|---|---|---|---|---|
| Harian   | 14 hari aktif | 7 | 70/15/15 kronologis | 10 fitur, termasuk cyclical sin/cos + `is_ramadan` |
| Mingguan | 5 minggu | 4 | 70/15/15 | dikurangi dari 8 karena ~75 minggu aktif |
| Bulanan  | 3 bulan | 3 | 65/35, **tanpa val set** | hanya ~20 bulan; EarlyStopping memantau `loss` |

Arsitektur `build_lstm_model`: stacked 2-layer LSTM (units, units//2) + Dropout(0.2) + L2(1e-4) + Dense(32, relu) + Dense(horizon). Split **kronologis, tidak diacak** (hindari kebocoran data time series). `MinMaxScaler` di-fit hanya pada train. Metrik: **RMSE, MAE, MAPE** — dihitung dalam Rupiah setelah `inverse_transform`.

---

## API POS (`app.py` vs `app_public.py`)

Keduanya adalah REST API Flask (port 5000) dengan **7 endpoint yang sama**. Perbedaannya di target pengguna.

### [app.py](app.py) — versi khusus Eatstedi
API produksi untuk kantin Eatstedi. Semua di-hardcode sesuai profilnya:
- **Kalender akademik baked-in**: Sabtu/Minggu tutup, Januari/Juli (libur semester) tutup.
- **Data produk asli Eatstedi** sebagai fallback (MAKANAN BASAH 48%, MINUMAN 16%, dst.; TAHU BAKSO, RISOL, dll.).
- `is_ramadan` khusus (Maret 2025/2026); rekomendasi stok pakai konstanta Eatstedi (Rp 3.150/unit) + peringatan `PERISHABLE`.

### [app_public.py](app_public.py) — versi generik/multi-tenant
Versi "produk jadi" untuk toko lain mana pun. Yang di app.py di-hardcode, di sini jadi konfigurasi dinamis:

| Aspek | app.py (Eatstedi) | app_public.py (Generik) |
|---|---|---|
| Akhir pekan | Selalu tutup | `open_on_weekends` (default buka), per-request |
| Bulan libur | Jan & Juli (paten) | `closed_months` (default kosong), per-request |
| Data produk | Nilai Eatstedi | Placeholder, atau di-inject via payload |
| Sumber kategori/produk | `raw_transactions.csv` / default | Bisa dari payload (mis. dari Supabase) |
| K (seasonal-naive) | Tetap | Auto-turun kalau data toko masih sedikit |

Referensi endpoint lengkap: `Dokumen/API Documentation.md` dan `Dokumen/API Documentation Public.md`.

### Daftar Endpoint
1. `GET  /api/status` — cek kesehatan API & ketersediaan data
2. `POST /api/predict/daily` — prediksi omzet harian
3. `POST /api/predict/weekly` — prediksi omzet mingguan
4. `POST /api/predict/monthly` — prediksi omzet bulanan
5. `POST /api/recommendations/stock` — alokasi anggaran per kategori + kuantitas stok produk (safety buffer 15%)
6. `POST /api/recommendations/target` — target omzet (konservatif/moderat/agresif) dari 15 hari aktif terakhir
7. `POST /api/sales/record` — catat rekap penjualan harian (upsert ke `daily_sales.csv`)

---

## Baseline Statistik yang Dipakai API

API **tidak memakai LSTM**. Prediksinya berbasis metode "naive" — mengambil rata-rata data historis terakhir.

### Seasonal-Naive (harian)
Omzet suatu hari diprediksi = **rata-rata K hari-yang-sama terakhir** (default `k=4`). "Hari yang sama" = hari dalam seminggu (Senin, Selasa, …), untuk menangkap **pola mingguan**.

> Contoh: 4 hari Senin terakhir beromzet 2,0/2,1/1,9/2,2 juta → prediksi tiap Senin ke depan = rata-ratanya = **Rp 2,05jt**. Selasa dihitung terpisah dari 4 Selasa terakhir, dst.

### Mean-4 (mingguan)
Prediksi minggu depan = **rata-rata 4 minggu terakhir**. Nilai yang sama (datar) dipakai untuk semua minggu ke depan.

### Mean-3 (bulanan)
Sama seperti Mean-4, tapi **rata-rata 3 bulan terakhir** (pakai 3 karena data bulanan sangat sedikit, ~20 bulan).

### Naive (pembanding)
Setiap endpoint juga menyertakan **Naive** = "prediksi = nilai terakhir yang tercatat" — pembanding paling minimal.

| Metode | Rumus inti | Pola mingguan? | Respons tren? |
|---|---|---|---|
| Naive | nilai terakhir | ❌ | ❌ |
| Seasonal-Naive | rata-rata K hari-yang-sama | ✅ | sebagian |
| Mean-4 / Mean-3 | rata-rata N periode terakhir | — (sudah agregat) | sebagian |

**Kelemahan:** karena berbasis rata-rata, prediksinya cenderung **datar** dan tidak menangkap lonjakan (welcome week) atau penurunan tajam (Ramadan) sampai kejadian itu masuk ke jendela rata-rata.

---

## Augmentasi Data TimeGAN

Implementasi PyTorch mandiri ([Data/TimeGAN/TimeGAN.py](Data/TimeGAN/TimeGAN.py), mengacu Yoon et al. 2019, NeurIPS) untuk mensintesis data harian tambahan guna mengatasi dataset kecil. Melatih 5 jaringan (embedder, recovery, generator, supervisor, discriminator) dalam 3 fase dan menulis `synthetic_daily.csv` (200 sequence × 24 hari). Perbandingan LSTM augmented-vs-baseline dijalankan dari `Eksperimen/lstm_timegan_augmented.ipynb`, hasilnya di `Models/TimeGAN/`.

---

## Struktur Proyek

```
ModelLSTM/
├── main.ipynb                    # Pipeline LSTM kanonik (Section 0–7)
├── app.py                        # API POS khusus Eatstedi
├── app_public.py                 # API POS generik/multi-tenant
├── requirements.txt
├── Data/
│   ├── Ekstrak/                  # daily/weekly/monthly_sales.csv, raw_transactions.csv, dump .sql
│   └── TimeGAN/                  # TimeGAN.py, model .pt, synthetic_daily.csv
├── Models/
│   ├── lstm_daily/weekly/monthly.keras
│   ├── scalers/                  # scaler .pkl
│   └── TimeGAN/                  # model & CSV eksperimen perbandingan
├── Eksperimen/                   # notebook eksplorasi (bukan pipeline kanonik)
├── Scripts/
│   └── generate_notebook.py      # regenerasi main.ipynb dari nol
└── Dokumen/                      # dokumen perencanaan & referensi API
```

Catatan: `Eksperimen/` berisi eksperimen (LSTM disederhanakan, hybrid-residual, forecast rekursif, evaluasi multi-seed, augmentasi TimeGAN). **`main.ipynb` adalah yang kanonik.** Bila `Dokumen/Planning.md` berbeda dengan notebook, **notebook yang benar**.

---

## Cara Menjalankan

```bash
pip install -r requirements.txt          # torch dari CPU wheel index (lihat requirements.txt)

python app.py                            # API POS Eatstedi, port 5000
python app_public.py                     # API POS generik

python Scripts/generate_notebook.py      # regenerasi main.ipynb (jalankan dari root)

# Augmentasi TimeGAN (PyTorch)
python Data/TimeGAN/TimeGAN.py --mode all       # latih + generate (default)
python Data/TimeGAN/TimeGAN.py --mode train     # latih + simpan model
python Data/TimeGAN/TimeGAN.py --mode generate  # generate dari model tersimpan
```

Pipeline riset LSTM dijalankan sel-per-sel di Jupyter melalui `main.ipynb`, bukan dari CLI.
