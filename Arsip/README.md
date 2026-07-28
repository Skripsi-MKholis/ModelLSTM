# Arsip

Berkas di sini **tidak dipakai** oleh pipeline LSTM (`main.ipynb`), API produksi
(`app.py`, `app_public.py`, `v2/`), atau backtest (`evaluation/backtest.py`).
Dipindahkan ke sini (bukan dihapus) supaya riwayatnya tetap ada tapi tidak lagi
membingungkan struktur folder utama.

## `Migrasi-Supabase/`

Skrip migrasi satu-kali (`extract_products.py`, `csv_to_products_sql.py`,
`gen_transactions_sql.py`) yang mengonversi dump SQL lama Eatstedi menjadi
`INSERT`/`UPDATE` untuk skema Supabase baru, beserta output-nya
(`products.csv`, `*_from_csv.sql`, `products_rows.sql`, `product_map*.json`,
`tx_batches/`). Tugas migrasinya sudah selesai (lihat commit "Ekstrak & Import
Data ke Supabase") — tidak ada kode lain di repo yang mereferensikan file-file
ini. `Data/Ekstrak/` sekarang hanya berisi input/output yang benar-benar dibaca
`main.ipynb` (dump `.sql` sumber + `raw_transactions.csv` + 3 CSV agregat).

## `featdump.txt`

Dump teks mentah beberapa sel `main.ipynb` (fitur harian), dibuat sebagai
catatan sementara — tidak direferensikan oleh kode apa pun.
