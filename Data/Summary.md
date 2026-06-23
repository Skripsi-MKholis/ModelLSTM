# Rangkuman Analisis Data: `eatstedi-20260621-010820.sql`
**Fokus Penelitian:** Rancang Bangun Aplikasi *Point of Sale* (POS) dengan Fitur Prediksi Penjualan Harian Menggunakan Metode *Long Short-Term Memory* (LSTM)

Dokumen ini menyajikan rangkuman dan analisis isi dari dump database `u556939220_eatstedi` yang bersumber dari data transaksi riil UMKM mitra **Eatstedi**. Analisis ini dirancang khusus untuk memetakan bagaimana data tersebut dapat digunakan sebagai **data latih (training data) dan data uji (testing data)** untuk model prediksi LSTM pada skripsi.

---

## Overview Sumber Data: EatsTEDI

EatsTEDI adalah platform kantin dan katalog menu digital resmi yang dikelola oleh mahasiswa di Departemen Teknik Elektro dan Informatika (DTEDI/TRE), bagian dari Universitas Gadjah Mada.

Platform ini dikelola oleh Divisi Kewirausahaan KM TEDI dan memungkinkan mahasiswa serta staf untuk memantau menu harian kantin secara digital. Penting untuk dicatat bahwa **tidak ada fitur untuk melakukan pemesanan secara online langsung dari website ini**. Website ini murni berfungsi sebagai katalog menu digital interaktif. Jika ingin membeli makanan yang ada di daftar tersebut, pembeli tetap harus datang langsung ke kantin DTEDI dan memesannya secara langsung di kasir fisik.

### Fitur Utama Platform [eatstedi.trpl.space](https://eatstedi.trpl.space/):
* 🍽️ **Katalog Menu Digital Interaktif**: Menampilkan daftar makanan/minuman harian terbaru (seperti Rice Bowl, Risol, hingga camilan).
* 🔍 **Cek Ketersediaan Menu**: Mengecek status ketersediaan menu secara real-time (apakah sedang *ready* atau habis).
* 💰 **Pantau Harga Terbaru**: Menampilkan informasi harga makanan dan minuman terkini secara transparan.
* 📦 **Indikator Sisa Stok**: Melihat sisa porsi makanan yang tersedia melalui angka indikator kotak di pojok kiri atas gambar menu (misalnya sisa 1 atau 3).
* 💳 **Pembayaran Non-Tunai**: Transaksi pembayaran di kasir fisik dapat menggunakan metode non-tunai (QRIS).
* 📢 **Informasi & Operasional**: Akun Instagram EatsTEDI menyediakan pengumuman jam operasional, tenant baru, dan berbagai kegiatan kantin mahasiswa.

### Ringkasan Singkat:
EatsTEDI ([eatstedi.trpl.space](https://eatstedi.trpl.space/)) adalah platform katalog menu digital interaktif resmi kantin DTEDI/TRE UGM yang memudahkan mahasiswa dan staf memantau harga, ketersediaan, dan sisa stok makanan secara real-time, sebelum melakukan pemesanan dan pembayaran langsung di kasir fisik kantin.

---

## 1. Ikhtisar Umum Dataset (General Dataset Overview)

* **Nama File:** `eatstedi-20260621-010820.sql`
* **Format File:** MariaDB/MySQL Database Dump (SQL)
* **Ukuran File:** 32,15 MB (~395.090 baris data)
* **Rentang Waktu Transaksi:** **24 Agustus 2024 s.d. 20 Juni 2026** (Rentang 666 hari / ~22 bulan)
* **Total Tabel:** 24 tabel (termasuk tabel sistem Laravel seperti `migrations`, `personal_access_tokens`, dan tabel Spatie Role-Permission).

### Tabel Utama yang Relevan untuk Penelitian:
1. **`invoices`**: Menyimpan data ringkasan transaksi utama (tanggal sukses, total harga, total kuantitas produk, status pembayaran, kasir yang memproses).
2. **`product_sold`**: Menghubungkan setiap item produk yang terjual dengan invoice, mencakup kuantitas, harga satuan, dan timestamp pembelian.
3. **`products`**: Menyimpan master data produk (nama, kategori, harga, supplier).
4. **`categories`**: Menyimpan data kategori produk (total 6 kategori aktif).
5. **`users`**: Menyimpan data akun pengguna (Owner, Admin, Employee/Kasir).

---

## 2. Statistik Transaksi Utama (Paid Invoices Only)

Statistik di bawah ini dihitung dari transaksi yang berstatus **lunas (`is_paid` = 1)** pada tabel `invoices`:

* **Total Nilai Penjualan (Revenue):** **Rp 501.440.492,00** (Setengah Miliar Rupiah)
* **Total Transaksi (Invoices):** **61.749 transaksi**
* **Total Kuantitas Produk Terjual:** **150.709 unit**
* **Rata-rata Transaksi per Hari Aktif:** **195,41 transaksi/hari**
* **Rata-rata Pendapatan per Hari Aktif:** **Rp 1.586.837,00/hari**
* **Rata-rata Produk Terjual per Hari Aktif:** **476,93 unit/hari**

---

## 3. Analisis Karakteristik Waktu & Musiman (Time-Series & Seasonality)
Karakteristik ini sangat krusial untuk pemodelan deret waktu dengan **LSTM**:

### A. Pola Mingguan (Weekly Profile)
Analisis jumlah hari aktif (hari dengan penjualan > 0) berdasarkan hari dalam seminggu menunjukkan pola yang sangat ekstrem:
* **Senin:** 61 hari aktif
* **Selasa:** 63 hari aktif
* **Rabu:** 64 hari aktif
* **Kamis:** 61 hari aktif
* **Jumat:** 59 hari aktif
* **Sabtu:** 4 hari aktif
* **Minggu:** 4 hari aktif

> [!IMPORTANT]
> **Temuan Karakteristik Bisnis:** Mitra UMKM Eatstedi hampir **100% beroperasi hanya pada hari kerja (Senin - Jumat)**. Transaksi pada hari Sabtu dan Minggu hampir tidak ada (hanya terjadi 4 kali sepanjang 2 tahun). Ini menunjukkan profil usaha yang berada di lingkungan akademis/kampus (kantin/koperasi kampus) yang tutup saat akhir pekan.

### B. Pola Bulanan & Libur Akademis (Monthly & Holiday Profile)
Pengelompokan pendapatan dan hari aktif per bulan menunjukkan adanya siklus libur berkala:
* **Bulan Sangat Aktif (Omzet Rp 25M - Rp 43M, 16-23 hari aktif):** September, Oktober, November, Februari, April, Mei.
* **Bulan Transisi/Ujian (Omzet Menurun, 8-15 hari aktif):** Desember, Juni (akhir semester).
* **Bulan Libur Total (Omzet Rp 0, 0 hari aktif):** **Januari dan Juli** (Siklus libur semester mahasiswa).
* **Bulan Ramadan (Omzet Merosot Tajam):** 
  * **Maret 2025:** Rp 1.427.000,00 (15 hari aktif, rata-rata omzet harian sangat kecil)
  * **Maret 2026:** Rp 3.531.000,00 (12 hari aktif)
  *(Kantin tutup atau membatasi jam operasional karena bulan puasa).*

---

## 4. Analisis Produk dan Kategori Terlaris

### A. Penjualan Berdasarkan Kategori Produk
Data ini penting untuk fitur restock pada POS karena membantu memprioritaskan kategori barang dengan perputaran tercepat (*fast-moving inventory*):

| No | Kategori | Total Penjualan (IDR) | Total Unit Terjual | Kontribusi (%) |
|----|----------|----------------------|--------------------|----------------|
| 1 | **MAKANAN BASAH** | Rp 242.770.500,00 | 82.134 unit | 48,42% |
| 2 | **MINUMAN** | Rp 82.122.500,00 | 16.861 unit | 16,38% |
| 3 | **MAKANAN KERING**| Rp 41.969.500,00 | 10.361 unit | 8,37% |
| 4 | **ICE CREAM** | Rp 33.802.500,00 | 10.942 unit | 6,74% |
| 5 | **SNACKS** | Rp 26.671.500,00 | 10.847 unit | 5,32% |
| 6 | **MERCH** | Rp 496.000,00 | 110 unit | 0,10% |

### B. Top 10 Produk Terlaris (Berdasarkan Kuantitas)
1. **TAHU BAKSO:** 23.097 unit (Total Omzet: Rp 57.742.500,00)
2. **TAHU BAKSO BALADO:** 16.688 unit (Total Omzet: Rp 42.020.500,00)
3. **RISOL MAYO & SOSIS:** 8.489 unit (Total Omzet: Rp 21.717.000,00)
4. **CHEESE/BANANA ROLL:** 3.417 unit (Total Omzet: Rp 3.417,000,00)
5. **LARIST AIR MINERAL:** 3.020 unit (Total Omzet: Rp 12.080.000,00)
6. **SANDWICH CRISPY:** 2.796 unit (Total Omzet: Rp 11.184.000,00)
7. **PISCOK:** 2.524 unit (Total Omzet: Rp 7.572,000,00)
8. **AYAM KRISPI:** 2.495 unit (Total Omzet: Rp 6,237,500,00)
9. **NASI RAMES/UDUK:** 2.382 unit (Total Omzet: Rp 11.910.000,00)
10. **DONAT:** 2.233 unit (Total Omzet: Rp 5.582.500,00)

---

## 5. Rekomendasi Implementasi untuk Model LSTM Skripsi

Berdasarkan temuan di atas, berikut adalah langkah-langkah konkret yang wajib diterapkan saat mengembangkan model prediksi penjualan harian dengan LSTM:

### 1. Penanganan Gap/Nilai Kosong (Data Cleaning & Imputation)
* **Masalah:** Ada 350 hari dengan penjualan Rp 0 (karena libur akhir pekan dan libur semester). Jika data ini dibiarkan apa adanya, model LSTM akan mengalami kebingungan karena lonjakan data yang tidak kontinu.
* **Solusi:** 
  * **Pendekatan 1 (Filtering):** Buang hari Sabtu, Minggu, serta bulan Januari dan Juli dari dataset. Latih model LSTM hanya menggunakan deret waktu *business days* (hari kerja aktif).
  * **Pendekatan 2 (Interpolasi Linier):** Sesuai rancangan skripsi (Bab III), gunakan interpolasi untuk meratakan gap kecil, tetapi untuk gap libur panjang (Januari/Juli), disarankan untuk memisahkan sesi pelatihan (*training session*) per semester untuk menghindari distorsi nilai prediksi.

### 2. Penggunaan Fitur Kontekstual (Multivariate LSTM)
* **Masalah:** Model LSTM *univariate* (hanya berdasarkan data historis omzet saja) akan kesulitan memprediksi penurunan tajam pada hari Sabtu/Minggu atau saat memasuki libur semester.
* **Solusi:** Gunakan arsitektur **Multivariate LSTM** dengan menambahkan fitur kontekstual (variabel masukan eksternal):
  * **Hari dalam seminggu (Day of Week):** Representasi integer `0` (Senin) s.d. `4` (Jumat) dan `5-6` (Sabtu-Minggu).
  * **Status Libur Akademis (Is_Holiday):** Variabel biner (`1` jika Januari/Juli/Ramadan, `0` jika hari kuliah aktif).
  * **Jumlah Transaksi Harian:** Sebagaimana riset terdahulu, korelasi antara jumlah transaksi dan omzet harian sangat erat sehingga fitur ini dapat menjadi prediktor kuat.

### 3. Normalisasi Data (Scaling)
* Mengingat omzet harian berfluktuasi cukup lebar (antara Rp 100.000 hingga Rp 2.500.000+ pada hari sibuk), penggunaan **MinMaxScaler** (skala 0 s.d. 1) wajib diimplementasikan sebelum data dimasukkan ke jaringan saraf LSTM untuk mempercepat konvergensi dan mencegah bias akibat nominal angka yang besar.

### 4. Relevansi Fitur Manajemen Stok POS
* Kategori **Makanan Basah** menyumbang hampir **50% total pendapatan** dan memiliki shelf life yang sangat singkat (cepat basi). 
* Oleh karena itu, fitur prediksi penjualan harian pada aplikasi POS yang Anda bangun akan sangat bermanfaat bagi Owner jika difokuskan untuk memprediksi kebutuhan stok bahan makanan basah (seperti bahan untuk Tahu Bakso, Risol, dsb.) guna menghindari *overstock* (bahan basi dan rugi) serta *stockout* (kehilangan potensi penjualan).
