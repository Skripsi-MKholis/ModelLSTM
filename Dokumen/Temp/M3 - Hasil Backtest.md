# M3 — Hasil Backtest & Keputusan Kriteria Produksi

> Dihasilkan otomatis oleh `evaluation/backtest.py`. Total fold walk-forward: 13.

## Ringkasan metrik (rata-rata lintas fold)

| model            | horizon   |              mae |             rmse |     mape |    smape |   n_samples |
|:-----------------|:----------|-----------------:|-----------------:|---------:|---------:|------------:|
| lstm_finetuned   | H+1       |      1.11284e+06 |      1.11284e+06 | 161.945  |  96.6825 |          13 |
| lstm_finetuned   | H+3       |      1.09575e+06 |      1.14585e+06 | 228.124  | 102.043  |          13 |
| lstm_finetuned   | H+7       |      1.15249e+06 |      1.23898e+06 | 242.333  | 114.501  |          13 |
| lstm_global      | H+1       |      1.11284e+06 |      1.11284e+06 | 161.945  |  96.6825 |          13 |
| lstm_global      | H+3       |      1.09575e+06 |      1.14585e+06 | 228.124  | 102.043  |          13 |
| lstm_global      | H+7       |      1.15249e+06 |      1.23898e+06 | 242.333  | 114.501  |          13 |
| moving_average_7 | H+1       | 473907           | 473907           |  78.5568 |  45.7111 |          13 |
| moving_average_7 | H+3       | 639278           | 693124           | 155.412  |  58.4637 |          13 |
| moving_average_7 | H+7       | 668575           | 763533           | 160.63   |  65.5851 |          13 |
| naive            | H+1       | 278808           | 278808           |  17.7545 |  17.0177 |          13 |
| naive            | H+3       | 382821           | 423722           |  57.3426 |  32.1251 |          13 |
| naive            | H+7       | 507330           | 599600           | 120.033  |  44.4756 |          13 |
| seasonal_naive   | H+1       | 729731           | 729731           | 179.021  |  58.455  |          13 |
| seasonal_naive   | H+3       | 825321           | 886224           | 243.198  |  70.6588 |          13 |
| seasonal_naive   | H+7       | 776989           | 912424           | 185.442  |  68.6145 |          13 |

## Kriteria lulus (§6 dokumen M3)

LSTM harus mengalahkan seasonal_naive pada MAPE H+1 **dan** H+7 di >= 60% unit uji.

**Keterbatasan yang diketahui**: dataset ini hanya berisi satu toko (Eatstedi) dengan seri harian panjang, jadi 'unit uji' di sini adalah *fold* walk-forward, bukan toko seperti diasumsikan kriteria asli (§4.2 mencatat mengapa model lintas-toko tidak diandalkan dengan komposisi data yang ada). `lstm_finetuned` adalah duplikat `lstm_global` karena tidak ada model fine-tuned terpisah yang bisa dilatih dari satu toko saja — dicatat di sini, bukan disembunyikan.

- Fold LSTM menang MAPE H+1: 6/13
- Fold LSTM menang MAPE H+7: 7/13
- Fold LSTM menang **keduanya** (H+1 dan H+7): 5/13 = 38.5%

## Keputusan: TIDAK LULUS

LSTM TIDAK mengalahkan seasonal_naive secara konsisten (>= 60% fold) pada MAPE H+1 dan H+7. Sesuai §6 dokumen M3: **baseline (seasonal_naive) tetap dipakai di produksi**, dan server tidak boleh mengembalikan `model_used = "lstm"` sampai kriteria ini terpenuhi — meski `/api/v2/forecast` tetap mengaktifkan jalur LSTM bila data mencukupi, hasil negatif ini adalah temuan yang sah dan harus dilaporkan apa adanya di skripsi, bukan disembunyikan atau dipaksakan lulus.