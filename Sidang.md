# Naskah Presentasi — Model LSTM Prediksi Penjualan Eatstedi

> Naskah ini untuk dibacakan/dipresentasikan saat sidang. Tiap bagian ≈ 1 slide.

---

### Slide 1 — Pembuka
📄 `main.ipynb` (di-generate dari [Scripts/generate_notebook.py](Scripts/generate_notebook.py)) — Section 0

"Pada bagian ini, saya akan menjelaskan model prediksi yang saya bangun untuk memprediksi pendapatan kantin Eatstedi. Model menggunakan pendekatan **LSTM atau Long Short-Term Memory**, yaitu jenis jaringan saraf tiruan yang dirancang khusus untuk data deret waktu atau time-series. Datanya sendiri mencakup transaksi dari 24 Agustus 2024 sampai 20 Juni 2026, dan saya membangun **tiga model terpisah** untuk tiga skala waktu: harian, mingguan, dan bulanan."

---

### Slide 2 — Alur Ekstraksi Data
📄 `main.ipynb` Section 1 (fungsi `_parse_sql_tuples`) → [Scripts/generate_notebook.py:85-263](Scripts/generate_notebook.py#L85-L263) · sumber: `Data/Ekstrak/eatstedi-*.sql` → output: `Data/Ekstrak/raw_transactions.csv`, `daily_sales.csv`, `weekly_sales.csv`, `monthly_sales.csv`

"Data mentah saya peroleh dari dump SQL database MariaDB milik sistem POS Eatstedi, berukuran sekitar 32 megabyte. Dump ini saya parsing langsung menggunakan Python, tanpa perlu menjalankan server MySQL. Dari dump tersebut, saya hanya mengambil empat tabel yang relevan: invoice, produk terjual, produk, dan kategori. Hasilnya kemudian saya agregasi menjadi tiga file: data harian, mingguan, dan bulanan."

---

### Slide 3 — Exploratory Data Analysis
📄 `main.ipynb` Section 2 → [Scripts/generate_notebook.py:368-500](Scripts/generate_notebook.py#L368-L500) (2.1 tren revenue, 2.2 pola bulanan, 2.3 heatmap, 2.4 distribusi/korelasi/outlier IQR)

"Sebelum masuk ke pemodelan, saya lakukan eksplorasi data terlebih dahulu — melihat tren pendapatan, pola bulanan, heatmap pola mingguan, distribusi data, korelasi antar variabel, dan deteksi outlier. Temuan pentingnya: **sekitar separuh hari kalender memiliki pendapatan nol** — ini terjadi karena kantin tutup di akhir pekan, libur akademik bulan Januari dan Juli, serta bulan Ramadan."

---

### Slide 4 — Preprocessing: Bagian Paling Penting
📄 `main.ipynb` Section 3 → [Scripts/generate_notebook.py:507-565](Scripts/generate_notebook.py#L507-L565)
- Filter hari aktif: baris 511 (`df_active = df[df['revenue'] > 0]`)
- Cyclical encoding sin/cos: baris 515-520
- Daftar `FEATURES` (10 fitur): baris 523-527
- Split kronologis 70/15/15: baris 533-544
- MinMaxScaler fit-train-only: baris 549-561 → simpan `Models/scalers/scaler_daily.pkl`, `scaler_revenue.pkl`

"Ini adalah tahap paling krusial dalam penelitian ini. Ada empat keputusan desain utama:

Pertama, saya **membuang seluruh hari dengan pendapatan nol**. Model hanya belajar dari urutan hari aktif berturut-turut — jadi polanya adalah 'hari buka ke hari buka berikutnya', bukan hari kalender biasa. Ini mencegah model salah belajar dari hari-hari tutup.

Kedua, saya terapkan **cyclical encoding** menggunakan fungsi sinus dan cosinus untuk fitur waktu yang sifatnya berulang, seperti hari dalam minggu dan bulan. Ini membantu model memahami bahwa, misalnya, Jumat dan Senin sebenarnya 'berdekatan' secara siklus mingguan.

Ketiga, saya membagi data secara **kronologis** — 70 persen data awal untuk training, 15 persen berikutnya untuk validasi, dan 15 persen terakhir untuk testing. Saya sengaja tidak mengacak data, karena ini data time-series — mengacak akan menyebabkan kebocoran informasi masa depan ke masa lalu.

Keempat, saya lakukan normalisasi dengan MinMaxScaler, dan scaler ini hanya di-fit pada data training saja, untuk mencegah data leakage."

---

### Slide 5 — Pembentukan Sequence
📄 `main.ipynb` Section 4 → fungsi `create_sequences()` di [Scripts/generate_notebook.py:573-586](Scripts/generate_notebook.py#L573-L586); dipanggil untuk daily (baris 590-592), weekly (Section 5.3), monthly (Section 5.4)

"Data yang sudah bersih kemudian saya ubah menjadi pasangan input-output menggunakan teknik sliding window. Untuk model harian, saya gunakan window 14 hari aktif terakhir sebagai input, untuk memprediksi 7 hari ke depan. Untuk model mingguan, window 5 minggu untuk memprediksi 4 minggu ke depan. Untuk model bulanan, window 3 bulan untuk memprediksi 3 bulan ke depan — model ini tidak memakai data validasi karena datanya sangat terbatas, hanya sekitar 20 bulan."

---

### Slide 6 — Arsitektur Model
📄 `main.ipynb` Section 5.1 → fungsi `build_lstm_model()` di [Scripts/generate_notebook.py:609-623](Scripts/generate_notebook.py#L609-L623); callbacks `EarlyStopping`/`ReduceLROnPlateau` baris 626-631. Training tiap model: Section 5.2 (harian), 5.3 (mingguan), 5.4 (bulanan) → output `Models/lstm_daily.keras`, `lstm_weekly.keras`, `lstm_monthly.keras`

"Ketiga model memakai arsitektur yang sama, yaitu **LSTM bertumpuk dua lapis**. Lapis pertama memiliki 64 unit, lapis kedua 32 unit, masing-masing diikuti Dropout sebesar 0.2 dan regularisasi L2 untuk mencegah overfitting — ini penting karena ukuran dataset kami relatif kecil. Setelah itu ada dua lapis Dense, dengan lapis terakhir menghasilkan output prediksi sesuai horizonnya.

Untuk training, saya gunakan optimizer Adam dengan loss function Mean Squared Error. Saya juga menerapkan dua callback: EarlyStopping, yang menghentikan training saat performa validasi berhenti membaik, dan ReduceLROnPlateau, yang menurunkan learning rate saat training mulai stagnan."

---

### Slide 7 — Evaluasi Model
📄 `main.ipynb` Section 6 → [Scripts/generate_notebook.py:770-899](Scripts/generate_notebook.py#L770-L899) (6.1 fungsi metrik RMSE/MAE/MAPE, 6.2 visualisasi actual vs predicted, 6.3 ringkasan tabel). Perbandingan lengkap dengan baseline: `Models/TimeGAN/comparison_results.csv`

"Setiap model saya evaluasi menggunakan tiga metrik dalam satuan Rupiah, yaitu RMSE, MAE, dan MAPE. Temuan penting dari evaluasi ini: pada dataset yang masih relatif kecil ini, baseline sederhana seperti **Seasonal-Naive** — yaitu menggunakan nilai minggu lalu di hari yang sama — ternyata seringkali **menyamai atau bahkan mengungguli** performa LSTM. Ini karena data historis yang tersedia belum cukup panjang bagi LSTM untuk menemukan pola yang lebih kompleks dari sekadar pola musiman mingguan."

---

### Slide 8 — Forecasting
📄 `main.ipynb` Section 7 → [Scripts/generate_notebook.py:901-1050+](Scripts/generate_notebook.py#L901) — `forecast_daily()` (7.1, baris 906), `forecast_weekly_fn()` (7.2, baris 979), `forecast_monthly_fn()` (7.3, baris 1012), visualisasi gabungan (7.4)

"Setelah model dilatih dan dievaluasi, saya gunakan untuk memprediksi masa depan: 7 hari aktif ke depan untuk model harian, 4 minggu ke depan untuk model mingguan, dan 3 bulan ke depan untuk model bulanan. Karena model hanya mengenal hari aktif, tanggal hasil prediksi saya petakan kembali ke kalender asli dengan melompati akhir pekan dan bulan libur, supaya hasilnya tetap masuk akal sebagai hari kantin buka."

---

### Slide 9 — Kaitan dengan API Produksi
📄 [app.py](app.py) / [app_public.py](app_public.py) (endpoint statistik, tidak memuat `.keras`) · gate lolos/tidak: `v2/forecast_service.py` fungsi `lstm_cleared_production_bar()` · verdict backtest: `Dokumen/M3 - Hasil Backtest.md` · adapter siap pakai: `v2/lstm_adapter.py`

"Perlu saya tegaskan bahwa model LSTM ini **belum digunakan pada API produksi** yang melayani aplikasi POS. API produksi saat ini masih memakai baseline statistik seperti Seasonal-Naive. Ini bukan kekurangan implementasi, melainkan **keputusan berbasis data** — hasil backtest menunjukkan LSTM belum secara konsisten mengungguli Seasonal-Naive pada dataset satu toko yang masih terbatas ini. Saya sudah menyiapkan mekanisme di sistem versi 2 yang akan otomatis beralih memakai LSTM begitu data bertambah banyak dan backtest dinyatakan lolos."

---

### Slide 10 — Upaya Mengatasi Keterbatasan Data
📄 [Data/TimeGAN/TimeGAN.py](Data/TimeGAN/TimeGAN.py) (5 jaringan: embedder, recovery, generator, supervisor, discriminator) → output `Data/TimeGAN/synthetic_daily.csv` · eksperimen perbandingan: [Eksperimen/lstm_timegan_augmented.ipynb](Eksperimen/lstm_timegan_augmented.ipynb) · hasil: `Models/TimeGAN/`

"Untuk mengatasi keterbatasan jumlah data ini, saya juga bereksperimen dengan augmentasi data sintetis menggunakan metode **TimeGAN**, yaitu model generative adversarial network yang dirancang khusus untuk menghasilkan data time-series sintetis. Saya melatih lima jaringan sekaligus — embedder, recovery, generator, supervisor, dan discriminator — untuk menghasilkan sequence harian tambahan, lalu membandingkan performa LSTM dengan dan tanpa augmentasi data ini."

---

### Slide 11 — Penutup
📄 Ringkasan seluruh pipeline: `main.ipynb` (Section 0–7) · dokumentasi keseluruhan: [CLAUDE.md](CLAUDE.md)

"Sebagai kesimpulan, penelitian ini membangun pipeline lengkap dari ekstraksi data mentah hingga forecasting menggunakan LSTM, dengan perhatian khusus pada karakteristik unik data kantin — yaitu hari-hari tutup yang signifikan. Meskipun performa LSTM saat ini belum secara konsisten mengungguli baseline sederhana karena keterbatasan volume data historis, arsitektur dan sistem evaluasi yang saya bangun sudah siap untuk secara otomatis mengadopsi LSTM begitu data historis Eatstedi terus bertambah dari waktu ke waktu. Terima kasih."
