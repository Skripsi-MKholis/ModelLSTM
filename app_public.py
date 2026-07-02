import os
import json
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── CONFIGURATION & CONFIG PATHS ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
EKSTRAK_DIR = os.path.join(DATA_DIR, "Ekstrak")
DAILY_CSV = os.path.join(EKSTRAK_DIR, "daily_sales.csv")
WEEKLY_CSV = os.path.join(EKSTRAK_DIR, "weekly_sales.csv")
MONTHLY_CSV = os.path.join(EKSTRAK_DIR, "monthly_sales.csv")
TRANSACTIONS_CSV = os.path.join(EKSTRAK_DIR, "raw_transactions.csv")

# ── GLOBAL CONFIG OPERASIONAL DEFAULT ────────────────────────────────────────
# Pengaturan operasional default (bisa di-overwrite dinamis via request payload)
OPEN_ON_WEEKENDS_DEFAULT = True   # Untuk toko umum, biasanya buka di akhir pekan
CLOSED_MONTHS_DEFAULT = []        # Untuk toko umum, biasanya buka sepanjang tahun (tidak ada libur akademik)

# ── INSIALISASI DAN FALLBACK DATA PRODUK ─────────────────────────────────────
# Kategori & produk default (hanya digunakan jika raw_transactions.csv tidak ditemukan)
DEFAULT_CATEGORY_SHARES = {
    "Makanan": 0.50,
    "Minuman": 0.30,
    "Lain-lain": 0.20
}

DEFAULT_TOP_PRODUCTS = [
    {"name": "Produk A", "category": "Makanan", "price": 10000, "qty_share": 0.20},
    {"name": "Produk B", "category": "Minuman", "price": 5000, "qty_share": 0.15},
    {"name": "Produk C", "category": "Makanan", "price": 15000, "qty_share": 0.10}
]

category_shares = DEFAULT_CATEGORY_SHARES.copy()
top_products = DEFAULT_TOP_PRODUCTS.copy()

def load_product_statistics():
    """Membuat statistik produk dari raw_transactions.csv milik toko baru secara otomatis."""
    global category_shares, top_products
    if os.path.exists(TRANSACTIONS_CSV):
        try:
            print("[INFO] Memuat statistik produk baru dari raw_transactions.csv...")
            df_tx = pd.read_csv(TRANSACTIONS_CSV)
            
            # Cek kolom wajib (name = nama produk, category_name = kategori)
            required_cols = {"category_name", "name", "quantity", "price"}
            if not required_cols.issubset(df_tx.columns):
                print("[WARNING] Kolom transaksi tidak sesuai spesifikasi. Menggunakan default.")
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
            print("[INFO] Statistik produk toko berhasil dimuat secara dinamis.")
        except Exception as e:
            print(f"[ERROR] Gagal memproses data produk: {e}. Menggunakan default.")

# Load statistik saat server dijalankan
load_product_statistics()


# ── FUNGSI PEMBANTU FORECASTING (DENGAN OPERASIONAL DINAMIS) ─────────────────
def get_daily_dataframe(history_data=None):
    """Mengambil dataframe harian dari file CSV atau request payload."""
    if history_data:
        df = pd.DataFrame(history_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        if not os.path.exists(DAILY_CSV):
            raise FileNotFoundError(f"Database daily_sales.csv tidak ditemukan di {DAILY_CSV}")
        df = pd.read_csv(DAILY_CSV)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # Filter hanya hari dengan penjualan > 0
    df_active = df[df['revenue'] > 0].copy()
    df_active['day_of_week'] = df_active['date'].dt.weekday
    return df_active

def is_business_day(date, open_on_weekends=OPEN_ON_WEEKENDS_DEFAULT, closed_months=CLOSED_MONTHS_DEFAULT):
    """Menentukan apakah tanggal tersebut adalah hari kerja aktif berdasarkan kalender toko."""
    # Periksa operasional akhir pekan
    if not open_on_weekends and date.weekday() >= 5:
        return False
    # Periksa daftar bulan libur/tutup
    if date.month in closed_months:
        return False
    return True

def generate_future_business_days(start_date, n_steps, open_on_weekends=OPEN_ON_WEEKENDS_DEFAULT, closed_months=CLOSED_MONTHS_DEFAULT):
    """Membentuk tanggal masa depan yang aktif sesuai profil operasional toko."""
    future_days = []
    curr_date = pd.Timestamp(start_date)
    while len(future_days) < n_steps:
        curr_date += pd.Timedelta(days=1)
        if is_business_day(curr_date, open_on_weekends, closed_months):
            future_days.append(curr_date)
    return future_days


# ── ENDPOINTS API ────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Halaman indeks — mencegah 404 saat URL dasar diakses langsung."""
    return jsonify({
        "message": "ModelLSTM API (Public/Generic) aktif.",
        "scope": "public_generic",
        "endpoints": {
            "status": "GET /api/status",
            "predict_daily": "GET/POST /api/predict/daily",
            "predict_weekly": "GET/POST /api/predict/weekly",
            "predict_monthly": "GET/POST /api/predict/monthly",
            "recommend_stock": "GET/POST /api/recommendations/stock",
            "recommend_target": "GET/POST /api/recommendations/target",
            "sales_record": "POST /api/sales/record"
        },
        "docs": "Dokumen/API Documentation Public.md"
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Mengecek status kesiapan server dan ketersediaan data toko."""
    status = {
        "status": "online",
        "scope": "public_generic",
        "timestamp": datetime.datetime.now().isoformat(),
        "database_status": {
            "daily_sales_exists": os.path.exists(DAILY_CSV),
            "weekly_sales_exists": os.path.exists(WEEKLY_CSV),
            "monthly_sales_exists": os.path.exists(MONTHLY_CSV),
            "raw_transactions_exists": os.path.exists(TRANSACTIONS_CSV)
        },
        "default_config": {
            "open_on_weekends": OPEN_ON_WEEKENDS_DEFAULT,
            "closed_months": CLOSED_MONTHS_DEFAULT
        }
    }
    
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
    Endpoint Prediksi Penjualan Harian (Generic).
    Menerima JSON parameter:
      - n_days: Jumlah hari prediksi ke depan (default 7)
      - k: Jumlah periode historis hari-yang-sama untuk rata-rata (default 4)
      - open_on_weekends: Buka di hari Sabtu/Minggu? (default True)
      - closed_months: List indeks bulan toko libur/tutup (default [])
      - history: Data transaksi eksternal opsional
    """
    data = request.get_json(silent=True) or {}
    n_days = int(data.get("n_days", 7))
    k = int(data.get("k", 4))
    
    # Pengaturan operasional dinamis dari request
    open_on_weekends = data.get("open_on_weekends", OPEN_ON_WEEKENDS_DEFAULT)
    if isinstance(open_on_weekends, str):
        open_on_weekends = open_on_weekends.lower() == 'true'
        
    raw_closed_months = data.get("closed_months", CLOSED_MONTHS_DEFAULT)
    closed_months = [int(m) for m in raw_closed_months] if raw_closed_months else []
    
    history_payload = data.get("history", None)
    
    try:
        dfa = get_daily_dataframe(history_payload)
    except Exception as e:
        return jsonify({"error": f"Gagal memuat data penjualan: {str(e)}"}), 400
        
    if len(dfa) < k * 5:
        # Jika toko baru memiliki data sedikit, kurangi nilai K secara otomatis agar tidak error
        k = max(1, len(dfa) // 5)
        if k == 0:
            return jsonify({"error": "Data historis harian terlalu sedikit untuk memulai peramalan."}), 400
            
    last_date = dfa['date'].max()
    
    # Perhitungan Seasonal-Naive
    by_dow = {}
    active_days_in_week = 7 if open_on_weekends else 5
    
    for dow in range(7):
        dow_data = dfa[dfa['day_of_week'] == dow]['revenue']
        if len(dow_data) >= k:
            by_dow[dow] = dow_data.tail(k).mean()
        elif len(dow_data) > 0:
            by_dow[dow] = dow_data.mean()
        else:
            by_dow[dow] = dfa['revenue'].tail(10).mean() if len(dfa) > 0 else 0
            
    # Perhitungan Naive (kemarin)
    naive_value = float(dfa['revenue'].iloc[-1]) if len(dfa) > 0 else 0.0
    
    # Buat tanggal prediksi ke depan sesuai setelan operasional
    future_dates = generate_future_business_days(last_date, n_days, open_on_weekends, closed_months)
    
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
        
    total_sn = sum(p["predicted_revenue_seasonal_naive"] for p in predictions)
    total_naive = sum(p["predicted_revenue_naive"] for p in predictions)
    
    response = {
        "metadata": {
            "model_used": f"Seasonal-Naive (k={k}) - Stabil",
            "baseline_model": "Naive",
            "historical_last_date": last_date.strftime('%Y-%m-%d'),
            "n_days_forecasted": n_days,
            "config_applied": {
                "open_on_weekends": open_on_weekends,
                "closed_months": closed_months
            }
        },
        "predictions": predictions,
        "summary": {
            "total_predicted_revenue_seasonal_naive": total_sn,
            "average_predicted_revenue_seasonal_naive": int(round(total_sn / n_days)) if n_days > 0 else 0,
            "total_predicted_revenue_naive": total_naive,
            "average_predicted_revenue_naive": int(round(total_naive / n_days)) if n_days > 0 else 0
        }
    }

    print(f"[PREDICT] /api/predict/daily -> {json.dumps(response, indent=2, default=str)}")
    return jsonify(response), 200


@app.route('/api/predict/weekly', methods=['GET', 'POST'])
def predict_weekly():
    """Prediksi mingguan untuk toko baru."""
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
    """Prediksi bulanan untuk toko baru."""
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
    Mendukung operasional dinamis toko baru.
    """
    data = request.get_json(silent=True) or {}
    pred_rev = data.get("predicted_revenue", None)
    target_date_str = data.get("target_date", None)
    open_on_weekends = data.get("open_on_weekends", OPEN_ON_WEEKENDS_DEFAULT)
    raw_closed_months = data.get("closed_months", CLOSED_MONTHS_DEFAULT)
    closed_months = [int(m) for m in raw_closed_months] if raw_closed_months else []
    
    if pred_rev is None:
        try:
            history_payload = data.get("history", None)
            dfa = get_daily_dataframe(history_payload)
            last_date = dfa['date'].max()
            
            if target_date_str:
                target_date = pd.to_datetime(target_date_str)
            else:
                target_date = generate_future_business_days(last_date, 1, open_on_weekends, closed_months)[0]
                
            dow = target_date.weekday()
            dow_data = dfa[dfa['day_of_week'] == dow]['revenue']
            if len(dow_data) >= 4:
                pred_rev = float(dow_data.tail(4).mean())
            else:
                pred_rev = float(dfa['revenue'].tail(10).mean())
        except Exception as e:
            return jsonify({"error": f"Gagal memprediksi target omzet otomatis: {str(e)}"}), 400
    else:
        pred_rev = float(pred_rev)
        target_date = datetime.date.today() + datetime.timedelta(days=1)
        
    # Alokasi kategori produk toko baru secara dinamis
    # Memeriksa apakah ada custom category shares di request payload dari database (Supabase)
    custom_category_shares = data.get("category_shares", None)
    shares_to_use = custom_category_shares if custom_category_shares else category_shares
    
    category_allocations = []
    for cat, share in shares_to_use.items():
        allocated_value = pred_rev * share
        category_allocations.append({
            "category": cat,
            "revenue_share": float(round(share, 4)),
            "allocated_budget_rupiah": int(round(allocated_value))
        })
    category_allocations = sorted(category_allocations, key=lambda x: x["allocated_budget_rupiah"], reverse=True)
    
    # Rekomendasi kuantitas produk berdasarkan porsi unit penjualan
    # Memeriksa apakah ada custom top products di request payload dari database (Supabase)
    custom_top_products = data.get("top_products", None)
    products_to_use = custom_top_products if custom_top_products else top_products
    
    product_recommendations = []
    avg_item_price = 10000.0
    if len(products_to_use) > 0:
        prices = [p["price"] for p in products_to_use]
        avg_item_price = sum(prices) / len(prices)
        
    total_estimated_units = pred_rev / avg_item_price if avg_item_price > 0 else 0
    
    for prod in products_to_use:
        recommended_qty = total_estimated_units * prod["qty_share"]
        safety_stock_qty = recommended_qty * 1.15  # +15% safety stock buffer
        
        product_recommendations.append({
            "product_name": prod["name"],
            "category": prod["category"],
            "unit_price_rupiah": prod["price"],
            "qty_share": prod["qty_share"],
            "recommended_stock_qty": int(max(1, round(recommended_qty))),
            "recommended_stock_with_safety_buffer": int(max(1, round(safety_stock_qty)))
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
    """Memberikan target omzet dinamis dari riwayat transaksi toko baru."""
    data = request.get_json(silent=True) or {}
    factor = float(data.get("factor", 1.10))
    history_payload = data.get("history", None)
    
    try:
        dfa = get_daily_dataframe(history_payload)
        last_15_days = dfa.tail(15)
        
        mean_15 = float(last_15_days['revenue'].mean())
        std_15 = float(last_15_days['revenue'].std()) if len(last_15_days) > 1 else 0.0
        max_15 = float(last_15_days['revenue'].max())
        
        target_konservatif = mean_15
        target_moderat = mean_15 * factor
        target_agresif = mean_15 + (1.5 * std_15) if std_15 > 0 else mean_15 * 1.25
        
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
            }
        }
        print(f"[PREDICT] /api/recommendations/target -> {json.dumps(response, indent=2, default=str)}")
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/sales/record', methods=['POST'])
def record_sales():
    """Menambahkan data rekap penjualan kasir terbaru."""
    data = request.get_json(silent=True) or {}
    req_fields = ["date", "revenue", "transactions", "qty_sold"]
    for field in req_fields:
        if field not in data:
            return jsonify({"error": f"Field '{field}' wajib diisi."}), 400
            
    try:
        target_date = datetime.datetime.strptime(data["date"], "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Format tanggal salah. Harus YYYY-MM-DD."}), 400
        
    rev = float(data["revenue"])
    tx = int(data["transactions"])
    qty = int(data["qty_sold"])
    
    if not os.path.exists(DAILY_CSV):
        return jsonify({"error": "Database harian tidak ditemukan."}), 500
        
    try:
        df = pd.read_csv(DAILY_CSV)
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
        
        # Cek jika tanggal sudah ada untuk ditimpa (overwrite)
        if target_date in df['date_parsed'].values:
            idx = df[df['date_parsed'] == target_date].index[0]
            df.loc[idx, 'revenue'] = rev
            df.loc[idx, 'transactions'] = tx
            df.loc[idx, 'qty_sold'] = qty
            df.loc[idx, 'day_of_week'] = target_date.weekday()
            df.loc[idx, 'week_of_year'] = target_date.isocalendar()[1]
            df.loc[idx, 'month'] = target_date.month
            df.loc[idx, 'is_weekend'] = 1 if target_date.weekday() >= 5 else 0
            # Penentuan libur disesuaikan dinamis
            df.loc[idx, 'is_holiday'] = 1 if target_date.month in CLOSED_MONTHS_DEFAULT else 0
            df.loc[idx, 'is_ramadan'] = 0
            
            message = f"Data tanggal {data['date']} berhasil diperbarui."
        else:
            # Tambahkan baris baru (append)
            new_row = {
                "date": data["date"],
                "revenue": rev,
                "transactions": tx,
                "qty_sold": qty,
                "day_of_week": target_date.weekday(),
                "week_of_year": target_date.isocalendar()[1],
                "month": target_date.month,
                "is_weekend": 1 if target_date.weekday() >= 5 else 0,
                "is_holiday": 1 if target_date.month in CLOSED_MONTHS_DEFAULT else 0,
                "is_ramadan": 0
            }
            
            for col in df.columns:
                if col not in new_row and col != 'date_parsed':
                    new_row[col] = 0
            
            df = df.drop(columns=['date_parsed'])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            message = f"Data tanggal {data['date']} berhasil disimpan."
            
        if 'date_parsed' in df.columns:
            df = df.drop(columns=['date_parsed'])
            
        df.to_csv(DAILY_CSV, index=False)
        return jsonify({"message": message, "recorded_data": data}), 200
        
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan transaksi: {str(e)}"}), 500


# ── RUN SERVER ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[START] Menjalankan server API Publik POS (Kompatibel Toko Lain)...")
    app.run(host='0.0.0.0', port=5000, debug=True)
