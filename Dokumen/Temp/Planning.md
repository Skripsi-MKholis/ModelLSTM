# Planning: Model LSTM Prediksi Penjualan Eatstedi

**Fokus Skripsi:** Prediksi Penjualan Harian, Mingguan, dan Bulanan menggunakan LSTM  
**Sumber Data:** `Data/eatstedi-20260621-010820.sql` (MariaDB dump, ~32MB)  
**Rentang Data:** 24 Agustus 2024 – 20 Juni 2026 (~22 bulan, 316 hari aktif bisnis)

---

## Ringkasan Skema Database Relevan

```
invoices          → id, user_id, is_paid, total_price, total_quantity, succeeded_at
product_sold      → id, invoice_id, product_id, quantity, price, purchased_at
products          → id, category_id, name, price
categories        → id, name  (6 kategori: MAKANAN BASAH, MINUMAN, MAKANAN KERING, ICE CREAM, SNACKS, MERCH)
```

**Join utama:**
```sql
SELECT ps.purchased_at, ps.quantity, ps.price,
       p.name AS product_name, c.name AS category_name,
       i.total_price, i.total_quantity, i.succeeded_at
FROM invoices i
JOIN product_sold ps ON ps.invoice_id = i.id
JOIN products p ON p.id = ps.product_id
JOIN categories c ON c.id = p.category_id
WHERE i.is_paid = 1
```

---

## Struktur Notebook (`main.ipynb`)

Notebook dibagi menjadi **7 section** dengan sel terpisah per langkah:

```
[SECTION 0] Setup & Import Libraries
[SECTION 1] Ekstraksi Data: SQL → CSV
[SECTION 2] Exploratory Data Analysis (EDA)
[SECTION 3] Preprocessing & Feature Engineering
[SECTION 4] Pembentukan Sequence LSTM
[SECTION 5] Arsitektur & Training Model
[SECTION 6] Evaluasi & Visualisasi Hasil
[SECTION 7] Prediksi ke Depan (Forecasting)
```

---

## SECTION 0 — Setup & Import Libraries

```python
# Wajib diinstall:
# pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras sqlparse

import re, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings; warnings.filterwarnings('ignore')
```

---

## SECTION 1 — Ekstraksi Data: SQL → CSV

> **Strategi:** Parse SQL dump secara langsung dengan Python (tanpa instalasi MySQL server) menggunakan `re` + `pandas`. Hanya ekstrak tabel yang dibutuhkan.

### 1.1 Fungsi Parser SQL

```python
def parse_sql_inserts(sql_path: str, table_name: str) -> pd.DataFrame:
    """
    Baca INSERT statements dari SQL dump, return DataFrame.
    Hanya memproses tabel yang diminta untuk efisiensi memori.
    """
    ...
```

**Tabel yang diekstrak:** `invoices`, `product_sold`, `products`, `categories`

### 1.2 Join & Filter

- Filter `invoices` hanya `is_paid = 1`
- Join `product_sold` → `products` → `categories`
- Kolom waktu utama: `invoices.succeeded_at` (waktu transaksi lunas)

### 1.3 Output CSV

| File | Isi | Digunakan untuk |
|------|-----|-----------------|
| `Data/raw_transactions.csv` | Semua transaksi per item produk + join | Base data, input agregasi |
| `Data/daily_sales.csv` | Agregasi harian | Prediksi harian |
| `Data/weekly_sales.csv` | Agregasi mingguan (Senin–Jumat) | Prediksi mingguan |
| `Data/monthly_sales.csv` | Agregasi bulanan | Prediksi bulanan |

#### Skema `daily_sales.csv`

```
date          | YYYY-MM-DD  | Tanggal transaksi (dari succeeded_at)
revenue       | float       | Total omzet harian (sum total_price)
transactions  | int         | Jumlah invoice lunas
qty_sold      | int         | Total unit produk terjual
day_of_week   | int         | 0=Senin, 4=Jumat, 5=Sabtu, 6=Minggu
week_of_year  | int         | Nomor minggu dalam setahun (ISO)
month         | int         | Nomor bulan (1–12)
is_weekend    | int         | 1 jika Sabtu/Minggu, 0 lainnya
is_holiday    | int         | 1 jika Januari/Juli (libur semester), 0 lainnya
is_ramadan    | int         | 1 jika Maret 2025 / Maret 2026, 0 lainnya
```

#### Skema `weekly_sales.csv`

```
year_week     | YYYY-WW     | Tahun-Minggu ISO (misal: 2025-03)
week_start    | YYYY-MM-DD  | Tanggal Senin minggu tersebut
revenue       | float       | Total omzet mingguan
transactions  | int         | Total transaksi
qty_sold      | int         | Total unit terjual
active_days   | int         | Jumlah hari aktif dalam minggu itu
```

#### Skema `monthly_sales.csv`

```
year_month    | YYYY-MM     | Periode bulan
revenue       | float       | Total omzet bulanan
transactions  | int         | Total transaksi
qty_sold      | int         | Total unit terjual
active_days   | int         | Jumlah hari aktif dalam bulan itu
```

---

## SECTION 2 — Exploratory Data Analysis (EDA)

> Tujuan: Memahami distribusi, tren, dan pola musiman sebelum masuk ke preprocessing.

### Visualisasi yang dibuat:

1. **Time series plot** — Revenue harian seluruh periode
2. **Heatmap pola mingguan** — Omzet rata-rata per hari dalam seminggu × bulan
3. **Bar chart bulanan** — Total revenue & jumlah hari aktif per bulan
4. **Box plot harian** — Distribusi omzet per hari dalam seminggu
5. **Rolling average** — Moving average 7 hari untuk melihat tren
6. **Correlation matrix** — Korelasi antar fitur (`revenue`, `transactions`, `qty_sold`)
7. **Distribusi revenue** — Histogram + KDE untuk memahami outlier

### Temuan kritis yang harus dicatat:

- **316 hari aktif** dari 666 hari total (~47% hari aktif)
- **Zero days**: 350 hari tanpa transaksi (Sabtu/Minggu + libur)
- **Januari & Juli** = libur semester total (0 transaksi)
- **Maret 2025 & 2026** = Ramadan, omzet turun >90%
- Revenue harian rentang: ~Rp100.000 – Rp2.500.000+

---

## SECTION 3 — Preprocessing & Feature Engineering

### 3.1 Strategi Penanganan Zero Days (Hari Tanpa Transaksi)

**Pendekatan yang dipilih: Filtering + Business Day Index**

- Buang semua baris dengan `revenue = 0` (hari Sabtu, Minggu, libur semester)
- Latih model menggunakan urutan **hari kerja aktif** saja (bukan kalender penuh)
- Implikasi: model belajar pola "hari aktif ke hari aktif berikutnya"
- Saat prediksi ke depan, output dikembalikan ke kalender dengan skip weekend otomatis

> Alternatif: Interpolasi linear untuk gap pendek (<3 hari), tapi untuk gap Januari/Juli lebih baik filter karena distorsi tinggi.

### 3.2 Outlier Handling

- Identifikasi outlier revenue dengan IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
- **Tidak dihapus**, hanya dicatat — hari ramai (welcome week mahasiswa) adalah pola valid
- Winsorize jika nilai > 3× std dari mean harian bulan yang sama

### 3.3 Normalisasi (Scaling)

```python
scaler = MinMaxScaler(feature_range=(0, 1))

# Scaling per kolom target
# revenue, transactions, qty_sold → masing-masing scaler terpisah
# Simpan scaler untuk inverse_transform saat evaluasi
```

> **Penting:** Fit scaler HANYA pada training set, lalu transform validation & test set — cegah data leakage.

### 3.4 Feature Engineering

**Fitur kontekstual yang ditambahkan ke input LSTM (Multivariate):**

| Fitur | Tipe | Keterangan |
|-------|------|------------|
| `revenue_scaled` | float [0,1] | Target utama, juga sebagai input |
| `transactions_scaled` | float [0,1] | Korelasi kuat dengan revenue |
| `qty_scaled` | float [0,1] | Total unit terjual |
| `day_of_week` | int / one-hot | Enkoding hari kerja (0–4) |
| `week_of_year_sin/cos` | float | Siklus mingguan (sin-cos encoding) |
| `month_sin/cos` | float | Siklus bulanan |
| `is_ramadan` | int 0/1 | Faktor penurunan Ramadan |

> **Cyclical encoding** untuk `week_of_year` dan `month`:
> ```python
> df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
> df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
> ```

---

## SECTION 4 — Pembentukan Sequence LSTM

### 4.1 Parameter Sequence

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `LOOK_BACK` | 14 | Panjang window input (14 hari aktif terakhir ≈ 3 minggu) |
| `FORECAST_HORIZON` | 7 | Prediksi 7 hari aktif ke depan (daily) |
| `BATCH_SIZE` | 32 | Ukuran batch training |

### 4.2 Fungsi Pembuat Sequence

```python
def create_sequences(data: np.ndarray, look_back: int, horizon: int):
    """
    data: array shape (n_samples, n_features)
    Returns X shape (samples, look_back, n_features)
            y shape (samples, horizon)  ← hanya kolom revenue
    """
    X, y = [], []
    for i in range(len(data) - look_back - horizon + 1):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back : i + look_back + horizon, 0])  # kolom 0 = revenue
    return np.array(X), np.array(y)
```

### 4.3 Train / Validation / Test Split

**Pembagian berdasarkan waktu (chronological split), BUKAN random:**

```
[=== TRAIN: 70% ===][= VAL: 15% =][= TEST: 15% =]
 Agt 2024–Agt 2025   Sep–Des 2025   Jan–Jun 2026
```

| Set | Proporsi | Tujuan |
|-----|----------|--------|
| Train | 70% | Latih bobot model |
| Validation | 15% | Tuning hyperparameter & early stopping |
| Test | 15% | Evaluasi akhir (tidak disentuh saat training) |

> **Catatan:** Setelah split, index `look_back` harus dijaga agar tidak ada overlap antar set.

---

## SECTION 5 — Arsitektur & Training Model

### 5.1 Model untuk Prediksi Harian (Multivariate Many-to-Many)

```python
def build_lstm_model(look_back, n_features, forecast_horizon, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back, n_features)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(forecast_horizon)          # output: 7 nilai revenue ke depan
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### 5.2 Model untuk Prediksi Mingguan

> **Catatan (diperbarui, sumber kebenaran = `main.ipynb`):** angka di bawah ini menggantikan draf awal
> (look_back=8) yang sudah tidak dipakai — direduksi karena data hanya ~75 minggu aktif.

- Input: sequence 5 minggu terakhir (`LOOK_BACK_W=5`)
- Output: 4 minggu ke depan (`HORIZON_W=4`)
- Fitur (`FEATURES_W`): `revenue`, `transactions`, `qty_sold`, `active_days`
- Split: 70/15/15 kronologis

### 5.3 Model untuk Prediksi Bulanan

> **Catatan (diperbarui, sumber kebenaran = `main.ipynb`):** angka di bawah ini menggantikan draf awal
> (look_back=6) yang sudah tidak dipakai — data hanya ~20 bulan, sehingga tidak ada val set terpisah.

- Input: sequence 3 bulan terakhir (`LOOK_BACK_M=3`)
- Output: 3 bulan ke depan (`HORIZON_M=3`)
- Fitur (`FEATURES_M`): `revenue`, `transactions`, `qty_sold`, `active_days`
- Split: 65/35, **tanpa val set** — EarlyStopping memantau `loss`, bukan `val_loss`

### 5.4 Hyperparameter Default

| Hyperparameter | Nilai |
|----------------|-------|
| `EPOCHS` | 100 (dengan early stopping) |
| `BATCH_SIZE` | 32 |
| `LSTM_UNITS` | 64 (layer 1), 32 (layer 2) |
| `DROPOUT` | 0.2 |
| `LEARNING_RATE` | 0.001 (Adam) |
| `PATIENCE` | 15 (early stopping) |

### 5.5 Callbacks

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
]
```

### 5.6 Training

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
```

---

## SECTION 6 — Evaluasi & Visualisasi

### 6.1 Metrik Evaluasi

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **RMSE** | √(mean((y_pred - y_true)²)) | Dalam satuan Rupiah, penalti outlier tinggi |
| **MAE** | mean(\|y_pred - y_true\|) | Rata-rata error absolut |
| **MAPE** | mean(\|y_pred - y_true\| / y_true) × 100 | Error dalam %, mudah diinterpretasi |

> Semua metrik dihitung setelah `inverse_transform` scaler ke satuan asli (Rupiah).

### 6.2 Visualisasi

1. **Training history** — Loss & MAE curve (train vs val per epoch)
2. **Actual vs Predicted** — Plot garis revenue aktual vs prediksi pada test set
3. **Scatter plot** — Actual vs Predicted (diagonal ideal = garis sempurna)
4. **Error distribution** — Histogram residual prediksi
5. **Tabel metrik** — RMSE, MAE, MAPE untuk train/val/test

---

## SECTION 7 — Forecasting (Prediksi ke Depan)

### 7.1 Prediksi Harian (7 Hari Aktif ke Depan)

```python
def forecast_next_n_days(model, last_sequence, n_days, scaler):
    """
    last_sequence: array (look_back, n_features) — data terakhir dari dataset
    Mengembalikan prediksi revenue untuk n hari kerja ke depan
    """
    ...
```

- Input: 14 hari aktif terakhir
- Output: 7 hari aktif ke depan
- Mapping ke kalender: skip Sabtu, Minggu, dan bulan libur

### 7.2 Prediksi Mingguan (4 Minggu ke Depan)

- Input: 8 minggu terakhir
- Output: total revenue 4 minggu ke depan

### 7.3 Prediksi Bulanan (3 Bulan ke Depan)

- Input: 6 bulan terakhir
- Output: total revenue 3 bulan ke depan

### 7.4 Visualisasi Forecast

- Plot gabungan: data historis 30 hari + prediksi 7 hari ke depan (dengan confidence band)
- Tabel prediksi dengan kolom: `tanggal`, `hari`, `prediksi_revenue`, `prediksi_transaksi`

---

## Struktur File Proyek

```
ModelLSTM/
├── Data/
│   ├── eatstedi-20260621-010820.sql    ← Raw dump (jangan diubah)
│   ├── raw_transactions.csv            ← Output Section 1 (per item)
│   ├── daily_sales.csv                 ← Output Section 1 (agregasi harian)
│   ├── weekly_sales.csv                ← Output Section 1 (agregasi mingguan)
│   └── monthly_sales.csv              ← Output Section 1 (agregasi bulanan)
├── Models/
│   ├── lstm_daily.keras               ← Model harian tersimpan
│   ├── lstm_weekly.keras              ← Model mingguan
│   ├── lstm_monthly.keras             ← Model bulanan
│   └── scalers/
│       ├── scaler_revenue.pkl
│       ├── scaler_transactions.pkl
│       └── scaler_qty.pkl
├── Dokumen/
│   ├── Planning.md                    ← File ini
│   └── Summary.md                    ← Analisis dataset
└── main.ipynb                        ← Notebook utama
```

---

## Checklist Pengerjaan

### Phase 1: Persiapan Data
- [ ] Section 0 — Setup library & konstanta
- [ ] Section 1.1 — Fungsi parser SQL dump
- [ ] Section 1.2 — Ekstrak & join 4 tabel utama
- [ ] Section 1.3 — Export `raw_transactions.csv`
- [ ] Section 1.4 — Agregasi & export `daily_sales.csv`
- [ ] Section 1.5 — Agregasi & export `weekly_sales.csv`
- [ ] Section 1.6 — Agregasi & export `monthly_sales.csv`

### Phase 2: EDA
- [ ] Section 2.1 — Time series plot harian
- [ ] Section 2.2 — Pola mingguan & bulanan
- [ ] Section 2.3 — Distribusi & outlier
- [ ] Section 2.4 — Correlation matrix

### Phase 3–4: Preprocessing & Sequence
- [ ] Section 3.1 — Filter zero days + business day index
- [ ] Section 3.2 — Feature engineering (cyclical encoding, binary flags)
- [ ] Section 3.3 — MinMaxScaler + simpan scaler
- [ ] Section 4.1 — Fungsi `create_sequences`
- [ ] Section 4.2 — Train/Val/Test split (70/15/15 chronological)

### Phase 5: Modeling
- [ ] Section 5.1 — Build & compile model harian
- [ ] Section 5.2 — Training model harian
- [ ] Section 5.3 — Build & train model mingguan
- [ ] Section 5.4 — Build & train model bulanan
- [ ] Section 5.5 — Simpan model (`.keras`) & scaler (`.pkl`)

### Phase 6: Evaluasi
- [ ] Section 6.1 — Hitung RMSE, MAE, MAPE (train/val/test)
- [ ] Section 6.2 — Plot actual vs predicted
- [ ] Section 6.3 — Tabel ringkasan metrik

### Phase 7: Forecasting
- [ ] Section 7.1 — Fungsi forecast 7 hari ke depan
- [ ] Section 7.2 — Fungsi forecast 4 minggu ke depan
- [ ] Section 7.3 — Fungsi forecast 3 bulan ke depan
- [ ] Section 7.4 — Visualisasi & tabel output prediksi

---

## Catatan Penting & Risiko

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Parse SQL 32MB lambat | Section 1 berjalan lama | Baca sekali, cache ke CSV, skip jika CSV sudah ada |
| Data Ramadan maret distorsi | MAPE membengkak | Tambah fitur `is_ramadan` + evaluasi terpisah per periode |
| Overfitting pada data kecil (316 hari) | Model tidak generalisasi | Dropout 0.2, early stopping, regularisasi L2 jika perlu |
| Look-back terlalu panjang | Kurang data train | Mulai dengan look_back=14, tuning ke 7 atau 21 jika MAPE >20% |
| Prediksi melampaui batas libur | Output tidak realistis | Post-processing: set prediksi = 0 untuk Sabtu/Minggu/Jan/Jul |
