Cara pakai:


# Latih + generate sekaligus (default)
python Data/TimeGAN.py --mode all

# Hanya latih (simpan model)
python Data/TimeGAN.py --mode train

# Hanya generate dari model tersimpan
python Data/TimeGAN.py --mode generate
Output:

Data/TimeGAN/ — model tersimpan (embedder, recovery, generator, dll.)
Data/synthetic_daily.csv — 200 synthetic sequences × 24 hari = 4.800 baris sintetis, format sama persis dengan daily_sales.csv
