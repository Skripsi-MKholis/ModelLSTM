import os
import json
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

from v2.routes import register_v2_routes
register_v2_routes(app)

# ── KONFIGURASI PATH ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
EKSTRAK_DIR = os.path.join(DATA_DIR, "Ekstrak")
DAILY_CSV = os.path.join(EKSTRAK_DIR, "daily_sales.csv")
WEEKLY_CSV = os.path.join(EKSTRAK_DIR, "weekly_sales.csv")
MONTHLY_CSV = os.path.join(EKSTRAK_DIR, "monthly_sales.csv")
TRANSACTIONS_CSV = os.path.join(EKSTRAK_DIR, "raw_transactions.csv")

# ── INSIALISASI DAN FALLBACK DATA ───────────────────────────────────────────
# Proporsi historis default jika file raw_transactions.csv tidak ada/error
DEFAULT_CATEGORY_SHARES = {
    "MAKANAN BASAH": 0.4842,
    "MINUMAN": 0.1638,
    "MAKANAN KERING": 0.0837,
    "ICE CREAM": 0.0674,
    "SNACKS": 0.0532,
    "MERCH": 0.0010
}

DEFAULT_TOP_PRODUCTS = [
    {"name": "TAHU BAKSO", "category": "MAKANAN BASAH", "price": 2500, "qty_share": 0.153},
    {"name": "TAHU BAKSO BALADO", "category": "MAKANAN BASAH", "price": 2518, "qty_share": 0.111},
    {"name": "RISOL MAYO & SOSIS", "category": "MAKANAN BASAH", "price": 2558, "qty_share": 0.056},
    {"name": "CHEESE/BANANA ROLL", "category": "MAKANAN BASAH", "price": 1000, "qty_share": 0.023},
    {"name": "LARIST AIR MINERAL", "category": "MINUMAN", "price": 4000, "qty_share": 0.020},
    {"name": "SANDWICH CRISPY", "category": "MAKANAN BASAH", "price": 4000, "qty_share": 0.019},
    {"name": "PISCOK", "category": "MAKANAN BASAH", "price": 3000, "qty_share": 0.017},
    {"name": "AYAM KRISPI", "category": "MAKANAN BASAH", "price": 2500, "qty_share": 0.016},
    {"name": "NASI RAMES/UDUK", "category": "MAKANAN BASAH", "price": 5000, "qty_share": 0.016},
    {"name": "DONAT", "category": "MAKANAN BASAH", "price": 2500, "qty_share": 0.015}
]

# Cache in-memory untuk statistik produk & kategori
category_shares = DEFAULT_CATEGORY_SHARES.copy()
top_products = DEFAULT_TOP_PRODUCTS.copy()

def load_product_statistics():
    """Memuat statistik porsi produk dari raw_transactions.csv jika tersedia."""
    global category_shares, top_products
    if os.path.exists(TRANSACTIONS_CSV):
        try:
            print("[INFO] Memuat statistik produk dari raw_transactions.csv...")
            df_tx = pd.read_csv(TRANSACTIONS_CSV)
            
            # Cek kolom yang dibutuhkan (name adalah nama kolom produk di CSV ini)
            required_cols = {"category_name", "name", "quantity", "price"}
            if not required_cols.issubset(df_tx.columns):
                print("[WARNING] Kolom transaksi tidak lengkap. Menggunakan nilai default.")
                return
                
            # Hitung share revenue per kategori
            df_tx["total_item_revenue"] = df_tx["quantity"] * df_tx["price"]
            total_rev = df_tx["total_item_revenue"].sum()
            if total_rev > 0:
                cat_rev = df_tx.groupby("category_name")["total_item_revenue"].sum()
                category_shares = (cat_rev / total_rev).to_dict()
                
            # Hitung 10 produk terlaris berdasarkan kuantitas
            total_qty = df_tx["quantity"].sum()
            if total_qty > 0:
                prod_stats = df_tx.groupby(["name", "category_name"]).agg(
                    total_qty=("quantity", "sum"),
                    avg_price=("price", "mean")
                ).reset_index()
                
                prod_stats["qty_share"] = prod_stats["total_qty"] / total_qty
                top_10 = prod_stats.sort_values(by="total_qty", ascending=False).head(10)
                
                top_products = []
                for _, row in top_10.iterrows():
                    top_products.append({
                        "name": row["name"],
                        "category": row["category_name"],
                        "price": int(round(row["avg_price"])),
                        "qty_share": float(row["qty_share"])
                    })
            print("[INFO] Statistik produk berhasil dimuat.")
        except Exception as e:
            print(f"[ERROR] Gagal memproses raw_transactions.csv: {e}. Menggunakan default.")

# Load statistik saat startup
load_product_statistics()


# ── FUNGSI PEMBANTU FORECASTING ──────────────────────────────────────────────
def get_daily_dataframe(history_data=None):
    """Mengambil dataframe harian, memfilter data aktif (>0) dan mengurutkannya."""
    if history_data:
        df = pd.DataFrame(history_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        if not os.path.exists(DAILY_CSV):
            raise FileNotFoundError(f"File database harian tidak ditemukan di {DAILY_CSV}")
        df = pd.read_csv(DAILY_CSV)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # Filter hanya hari aktif bisnis (revenue > 0)
    df_active = df[df['revenue'] > 0].copy()
    df_active['day_of_week'] = df_active['date'].dt.weekday
    return df_active

def is_business_day(date):
    """Menentukan apakah tanggal adalah hari aktif bisnis (bukan weekend & bukan Jan/Jul)."""
    # Weekend (Sabtu = 5, Minggu = 6)
    if date.weekday() >= 5:
        return False
    # Libur Semester Akademik (Januari = 1, Juli = 7)
    if date.month in [1, 7]:
        return False
    return True

def generate_future_business_days(start_date, n_steps):
    """Menghasilkan deret tanggal kerja aktif ke depan."""
    future_days = []
    curr_date = pd.Timestamp(start_date)
    while len(future_days) < n_steps:
        curr_date += pd.Timedelta(days=1)
        if is_business_day(curr_date):
            future_days.append(curr_date)
    return future_days


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Halaman indeks — mencegah 404 saat URL dasar diakses langsung."""
    return jsonify({
        "message": "ModelLSTM API (Eatstedi) aktif.",
        "scope": "eatstedi",
        "endpoints": {
            "status": "GET /api/status",
            "predict_daily": "GET/POST /api/predict/daily",
            "predict_weekly": "GET/POST /api/predict/weekly",
            "predict_monthly": "GET/POST /api/predict/monthly",
            "recommend_stock": "GET/POST /api/recommendations/stock",
            "recommend_target": "GET/POST /api/recommendations/target",
            "sales_record": "POST /api/sales/record"
        },
        "docs": "Dokumen/API Documentation.md"
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint cek status & kesehatan API."""
    status = {
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "database_status": {
            "daily_sales_exists": os.path.exists(DAILY_CSV),
            "weekly_sales_exists": os.path.exists(WEEKLY_CSV),
            "monthly_sales_exists": os.path.exists(MONTHLY_CSV),
            "raw_transactions_exists": os.path.exists(TRANSACTIONS_CSV)
        }
    }
    
    # Tambahkan info jumlah record jika ada
    if os.path.exists(DAILY_CSV):
        try:
            df = pd.read_csv(DAILY_CSV)
            status["database_status"]["daily_records_count"] = len(df)
            status["database_status"]["last_record_date"] = str(df['date'].max())
        except Exception:
            pass
            
    return jsonify(status), 200


@app.route('/api/predict/daily', methods=['GET', 'POST'])
def predict_daily():
    """
    Endpoint Prediksi Penjualan Harian.
    Metode: Seasonal-Naive (k=4) & Naive.
    Menerima JSON parameter:
      - n_days: jumlah hari prediksi ke depan (default 7)
      - k: jumlah riwayat kemunculan hari-yang-sama untuk dirata-rata (default 4)
      - history: list opsional berisi data transaksi terbaru dari POS
    """
    data = request.get_json(silent=True) or {}
    n_days = int(data.get("n_days", 7))
    k = int(data.get("k", 4))
    history_payload = data.get("history", None)
    
    try:
        dfa = get_daily_dataframe(history_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
    if len(dfa) < k * 5:
        return jsonify({"error": f"Data historis terlalu sedikit. Minimal diperlukan {k*5} hari aktif."}), 400
        
    last_date = dfa['date'].max()
    
    # ── METODE 1: Seasonal-Naive (Rata-rata K hari yang sama terakhir) ──
    # Menghitung rata-rata pendapatan berdasarkan Day of Week (0=Senin, 4=Jumat)
    by_dow = {}
    for dow in range(5):
        dow_data = dfa[dfa['day_of_week'] == dow]['revenue']
        if len(dow_data) >= k:
            by_dow[dow] = dow_data.tail(k).mean()
        elif len(dow_data) > 0:
            by_dow[dow] = dow_data.mean()
        else:
            by_dow[dow] = dfa['revenue'].tail(10).mean() # Fallback ke rata-rata umum
            
    # ── METODE 2: Naive (Prediksi = omzet hari kerja terakhir) ──
    naive_value = float(dfa['revenue'].iloc[-1])
    
    # Generate tanggal prediksi masa depan
    future_dates = generate_future_business_days(last_date, n_days)
    
    predictions = []
    for dt in future_dates:
        dow = dt.weekday()
        pred_sn = int(round(by_dow.get(dow, naive_value)))
        predictions.append({
            "date": dt.strftime('%Y-%m-%d'),
            "day": dt.strftime('%A'),
            "predicted_revenue_seasonal_naive": pred_sn,
            "predicted_revenue_naive": int(round(naive_value))
        })
        
    # Ringkasan total
    total_sn = sum(p["predicted_revenue_seasonal_naive"] for p in predictions)
    total_naive = sum(p["predicted_revenue_naive"] for p in predictions)
    
    response = {
        "metadata": {
            "model_used": "Seasonal-Naive (k=4) - Paling Stabil",
            "baseline_model": "Naive (Kemarin)",
            "historical_last_date": last_date.strftime('%Y-%m-%d'),
            "historical_last_revenue": naive_value,
            "n_days_forecasted": n_days,
            "k_seasons": k
        },
        "predictions": predictions,
        "summary": {
            "total_predicted_revenue_seasonal_naive": total_sn,
            "average_predicted_revenue_seasonal_naive": int(round(total_sn / n_days)),
            "total_predicted_revenue_naive": total_naive,
            "average_predicted_revenue_naive": int(round(total_naive / n_days))
        }
    }

    print(f"[PREDICT] /api/predict/daily -> {json.dumps(response, indent=2, default=str)}")
    return jsonify(response), 200


@app.route('/api/predict/weekly', methods=['GET', 'POST'])
def predict_weekly():
    """
    Endpoint Prediksi Penjualan Mingguan.
    Menerima JSON parameter:
      - n_weeks: jumlah minggu prediksi ke depan (default 4)
      - history: list opsional berisi data mingguan
    """
    data = request.get_json(silent=True) or {}
    n_weeks = int(data.get("n_weeks", 4))
    history_payload = data.get("history", None)
    
    try:
        if history_payload:
            df = pd.DataFrame(history_payload)
            df['revenue'] = df['revenue'].astype(float)
        else:
            if not os.path.exists(WEEKLY_CSV):
                return jsonify({"error": "Data mingguan tidak ditemukan"}), 400
            df = pd.read_csv(WEEKLY_CSV)
            
        if len(df) < 4:
            return jsonify({"error": "Data historis mingguan kurang untuk peramalan."}), 400
            
        # Metode: Rata-rata 4 minggu terakhir (Mean-4) & Naive
        mean_4 = float(df['revenue'].tail(4).mean())
        naive_val = float(df['revenue'].iloc[-1])
        
        predictions = []
        for i in range(1, n_weeks + 1):
            predictions.append({
                "week_index": i,
                "predicted_revenue_mean4": int(round(mean_4)),
                "predicted_revenue_naive": int(round(naive_val))
            })
            
        response = {
            "metadata": {
                "model_used": "Mean-4 Weekly Average",
                "baseline_model": "Naive Weekly"
            },
            "predictions": predictions,
            "summary": {
                "total_predicted_revenue_mean4": int(round(mean_4 * n_weeks)),
                "total_predicted_revenue_naive": int(round(naive_val * n_weeks))
            }
        }
        print(f"[PREDICT] /api/predict/weekly -> {json.dumps(response, indent=2, default=str)}")
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/predict/monthly', methods=['GET', 'POST'])
def predict_monthly():
    """
    Endpoint Prediksi Penjualan Bulanan.
    Menerima JSON parameter:
      - n_months: jumlah bulan prediksi ke depan (default 3)
      - history: list opsional berisi data bulanan
    """
    data = request.get_json(silent=True) or {}
    n_months = int(data.get("n_months", 3))
    history_payload = data.get("history", None)
    
    try:
        if history_payload:
            df = pd.DataFrame(history_payload)
            df['revenue'] = df['revenue'].astype(float)
        else:
            if not os.path.exists(MONTHLY_CSV):
                return jsonify({"error": "Data bulanan tidak ditemukan"}), 400
            df = pd.read_csv(MONTHLY_CSV)
            
        if len(df) < 3:
            return jsonify({"error": "Data historis bulanan kurang untuk peramalan."}), 400
            
        # Metode: Rata-rata 3 bulan terakhir (Mean-3) & Naive
        mean_3 = float(df['revenue'].tail(3).mean())
        naive_val = float(df['revenue'].iloc[-1])
        
        predictions = []
        for i in range(1, n_months + 1):
            predictions.append({
                "month_index": i,
                "predicted_revenue_mean3": int(round(mean_3)),
                "predicted_revenue_naive": int(round(naive_val))
            })
            
        response = {
            "metadata": {
                "model_used": "Mean-3 Monthly Average",
                "baseline_model": "Naive Monthly"
            },
            "predictions": predictions,
            "summary": {
                "total_predicted_revenue_mean3": int(round(mean_3 * n_months)),
                "total_predicted_revenue_naive": int(round(naive_val * n_months))
            }
        }
        print(f"[PREDICT] /api/predict/monthly -> {json.dumps(response, indent=2, default=str)}")
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/recommendations/stock', methods=['GET', 'POST'])
def recommend_stock():
    """
    Endpoint Rekomendasi Stok/Persediaan.
    Menghitung alokasi porsi belanja/stok per kategori dan rekomendasi kuantitas
    top produk berdasarkan target prediksi omzet harian.
    Menerima JSON parameter:
      - predicted_revenue: omzet yang diproyeksikan untuk hari tersebut (opsional)
      - target_date: tanggal target rekomendasi (default besok, digunakan jika predicted_revenue kosong)
    """
    data = request.get_json(silent=True) or {}
    pred_rev = data.get("predicted_revenue", None)
    target_date_str = data.get("target_date", None)
    
    # Jika tidak disediakan predicted_revenue, hitung otomatis menggunakan model Seasonal-Naive
    if pred_rev is None:
        try:
            dfa = get_daily_dataframe()
            last_date = dfa['date'].max()
            
            if target_date_str:
                target_date = pd.to_datetime(target_date_str)
            else:
                # Default besok (hari kerja berikutnya)
                target_date = generate_future_business_days(last_date, 1)[0]
                
            dow = target_date.weekday()
            
            # Hitung Seasonal-Naive untuk hari tersebut
            dow_data = dfa[dfa['day_of_week'] == dow]['revenue']
            if len(dow_data) >= 4:
                pred_rev = float(dow_data.tail(4).mean())
            else:
                pred_rev = float(dfa['revenue'].tail(10).mean())
        except Exception as e:
            return jsonify({"error": f"Gagal menghitung prediksi otomatis: {str(e)}"}), 400
    else:
        pred_rev = float(pred_rev)
        target_date = datetime.date.today() + datetime.timedelta(days=1)
        
    # ── ALOKASI REVENUE PER KATEGORI ──────────────────────────────────────────
    category_allocations = []
    for cat, share in category_shares.items():
        allocated_value = pred_rev * share
        category_allocations.append({
            "category": cat,
            "revenue_share": float(round(share, 4)),
            "allocated_budget_rupiah": int(round(allocated_value))
        })
        
    # Urutkan kategori berdasarkan anggaran terbesar (MAKANAN BASAH paling atas)
    category_allocations = sorted(category_allocations, key=lambda x: x["allocated_budget_rupiah"], reverse=True)
    
    # ── REKOMENDASI KUANTITAS PRODUK ──────────────────────────────────────────
    # Berdasarkan porsi unit penjualan historis
    product_recommendations = []
    for prod in top_products:
        # Perkiraan kontribusi penjualan produk (Rupiah) berdasarkan persentase
        # Kita estimasikan kuantitas unit yang perlu distok:
        # total unit terjual harian rata-rata dari revenue: 
        # Rata-rata transaksi harian: Rp 1,5M dengan kuantitas ~476 unit.
        # Artinya Rasio Rp/Unit = Rp 1.500.000 / 476 unit ≈ Rp 3,150 per unit barang.
        total_estimated_units = pred_rev / 3150.0
        
        # Rekomendasi kuantitas = estimasi unit total * porsi produk
        recommended_qty = total_estimated_units * prod["qty_share"]
        # Berikan safety stock buffer (mis. +15% untuk menghindari stockout pada makanan terlaris)
        safety_stock_qty = recommended_qty * 1.15
        
        # Makanan basah sangat cepat basi, jadi berikan warning khusus
        is_fast_moving_perishable = prod["category"] == "MAKANAN BASAH"
        
        product_recommendations.append({
            "product_name": prod["name"],
            "category": prod["category"],
            "unit_price_rupiah": prod["price"],
            "qty_share": prod["qty_share"],
            "recommended_stock_qty": int(max(1, round(recommended_qty))),
            "recommended_stock_with_safety_buffer": int(max(1, round(safety_stock_qty))),
            "handling_instruction": "PERISHABLE! Segera habiskan/jangan simpan melebihi 24 jam" if is_fast_moving_perishable else "Aman disimpan"
        })
        
    response = {
        "target_date": target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date),
        "predicted_revenue_base": int(round(pred_rev)),
        "category_allocations": category_allocations,
        "product_recommendations": product_recommendations
    }
    print(f"[PREDICT] /api/recommendations/stock -> {json.dumps(response, indent=2, default=str)}")
    return jsonify(response), 200


@app.route('/api/recommendations/target', methods=['GET', 'POST'])
def recommend_target():
    """
    Endpoint Rekomendasi Target Omzet.
    Memberikan target pencapaian penjualan yang menantang namun realistis (data-driven).
    Menerima JSON parameter:
      - factor: faktor kenaikan target, misal 1.10 untuk +10% target (default 1.10)
    """
    data = request.get_json(silent=True) or {}
    factor = float(data.get("factor", 1.10))
    
    try:
        dfa = get_daily_dataframe()
        last_15_days = dfa.tail(15)
        
        mean_15 = float(last_15_days['revenue'].mean())
        std_15 = float(last_15_days['revenue'].std())
        max_15 = float(last_15_days['revenue'].max())
        
        # Rumus target:
        # Target Konservatif = Rata-rata 15 hari aktif
        # Target Moderat = Rata-rata * factor
        # Target Agresif = Rata-rata + (1.5 * standar deviasi)
        target_konservatif = mean_15
        target_moderat = mean_15 * factor
        target_agresif = mean_15 + (1.5 * std_15)
        
        # Pastikan target agresif tidak melebihi omzet maks historis secara ekstrem
        if target_agresif > max_15 * 1.3:
            target_agresif = max_15 * 1.1
            
        response = {
            "historical_basis_15_active_days": {
                "average_revenue": int(round(mean_15)),
                "std_deviation": int(round(std_15)),
                "max_revenue": int(round(max_15))
            },
            "targets": {
                "konservatif_target": int(round(target_konservatif)),
                "moderat_target": int(round(target_moderat)),
                "agresif_target": int(round(target_agresif))
            },
            "status_pesan": {
                "konservatif_pesan": "Cocok untuk hari biasa/sepi (misal: masa-masa ujian awal).",
                "moderat_pesan": f"Target pertumbuhan standar (+{int((factor-1)*100)}%). Direkomendasikan untuk target operasional harian.",
                "agresif_pesan": "Cocok untuk masa-masa ramai (seperti awal masuk semester/welcome week)."
            }
        }
        print(f"[PREDICT] /api/recommendations/target -> {json.dumps(response, indent=2, default=str)}")
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/sales/record', methods=['POST'])
def record_sales():
    """
    Endpoint Pencatatan Penjualan Baru.
    Digunakan oleh POS untuk mengirim data transaksi harian lunas terbaru secara real-time.
    Menerima JSON parameter:
      - date: YYYY-MM-DD (Wajib)
      - revenue: float (Wajib)
      - transactions: int (Wajib)
      - qty_sold: int (Wajib)
    """
    data = request.get_json(silent=True) or {}
    req_fields = ["date", "revenue", "transactions", "qty_sold"]
    for field in req_fields:
        if field not in data:
            return jsonify({"error": f"Field '{field}' wajib diisi."}), 400
            
    # Validasi format tanggal
    try:
        target_date = datetime.datetime.strptime(data["date"], "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Format tanggal salah. Harus YYYY-MM-DD."}), 400
        
    rev = float(data["revenue"])
    tx = int(data["transactions"])
    qty = int(data["qty_sold"])
    
    # Cek apakah file daily_sales.csv ada
    if not os.path.exists(DAILY_CSV):
        return jsonify({"error": "Database daily_sales.csv tidak ditemukan."}), 500
        
    try:
        df = pd.read_csv(DAILY_CSV)
        # Konversi kolom date ke datetime untuk perbandingan
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
        
        # Cek apakah tanggal sudah pernah dicatat
        if target_date in df['date_parsed'].values:
            # Overwrite baris yang sudah ada
            idx = df[df['date_parsed'] == target_date].index[0]
            df.loc[idx, 'revenue'] = rev
            df.loc[idx, 'transactions'] = tx
            df.loc[idx, 'qty_sold'] = qty
            # Tentukan kolom ekstra
            df.loc[idx, 'day_of_week'] = target_date.weekday()
            df.loc[idx, 'week_of_year'] = target_date.isocalendar()[1]
            df.loc[idx, 'month'] = target_date.month
            df.loc[idx, 'is_weekend'] = 1 if target_date.weekday() >= 5 else 0
            df.loc[idx, 'is_holiday'] = 1 if target_date.month in [1, 7] else 0
            df.loc[idx, 'is_ramadan'] = 1 if (target_date.year == 2025 and target_date.month == 3) or (target_date.year == 2026 and target_date.month == 3) else 0
            
            message = f"Data tanggal {data['date']} berhasil diperbarui."
        else:
            # Append baris baru
            new_row = {
                "date": data["date"],
                "revenue": rev,
                "transactions": tx,
                "qty_sold": qty,
                "day_of_week": target_date.weekday(),
                "week_of_year": target_date.isocalendar()[1],
                "month": target_date.month,
                "is_weekend": 1 if target_date.weekday() >= 5 else 0,
                "is_holiday": 1 if target_date.month in [1, 7] else 0,
                "is_ramadan": 1 if (target_date.year == 2025 and target_date.month == 3) or (target_date.year == 2026 and target_date.month == 3) else 0
            }
            # Tambahkan kolom lain jika ada di csv asli
            for col in df.columns:
                if col not in new_row and col != 'date_parsed':
                    new_row[col] = 0
            
            # Buang date_parsed sementara untuk append
            df = df.drop(columns=['date_parsed'])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            message = f"Data tanggal {data['date']} berhasil ditambahkan."
            
        if 'date_parsed' in df.columns:
            df = df.drop(columns=['date_parsed'])
            
        # Simpan kembali ke CSV
        df.to_csv(DAILY_CSV, index=False)
        return jsonify({"message": message, "recorded_data": data}), 200
        
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan data: {str(e)}"}), 500


# ── RUN SERVER ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Jalankan server lokal di port 5000
    print("[START] Menjalankan server API POS Eatstedi...")
    app.run(host='0.0.0.0', port=5000, debug=True)
