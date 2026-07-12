"""
Konversi Data/Ekstrak/products.csv menjadi INSERT SQL untuk tabel `categories`
dan `products` (Supabase/Postgres), mengikuti skema & gaya file
Data/Ekstrak/categories_rows.sql & products_rows.sql.

6 kategori Eatstedi (hardcoded, sama seperti main.ipynb) masing-masing
diberi UUID baru untuk store_id ini, lalu setiap produk di-link ke
category_id UUID tersebut lewat category_id integer lama di CSV.

Output: Data/Ekstrak/categories_from_csv.sql, Data/Ekstrak/products_from_csv.sql
"""
import csv
import os
import uuid
from datetime import datetime, timezone

STORE_ID = '49e30ecd-48a0-441d-864d-0251323adc7b'
IN_CSV        = os.path.join('Data', 'Ekstrak', 'products.csv')
OUT_SQL_PROD  = os.path.join('Data', 'Ekstrak', 'products_from_csv.sql')
OUT_SQL_CAT   = os.path.join('Data', 'Ekstrak', 'categories_from_csv.sql')

CATEGORIES = {
    1: 'MAKANAN KERING',
    2: 'MINUMAN',
    3: 'SNACKS',
    4: 'ICE CREAM',
    5: 'MAKANAN BASAH',
    6: 'MERCH',
}

COLUMNS = [
    'id', 'store_id', 'name', 'price', 'modal_price', 'image_url', 'category',
    'stock_quantity', 'is_infinite_stock', 'barcode', 'variants',
    'discount_applied', 'created_at', 'description', 'category_id',
    'min_stock_level', 'sku', 'updated_at', 'cost_price', 'discount_id',
    'preparation_area',
]

CAT_COLUMNS = ['id', 'name', 'store_id', 'created_at']


def sql_str(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_num(value):
    return f"{float(value):.2f}"


def main():
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f+00')

    # ── Kategori: 1 UUID baru per kategori untuk store ini ───────────────────
    category_ids = {cat_id: str(uuid.uuid4()) for cat_id in CATEGORIES}

    cat_rows_sql = []
    for cat_id, cat_name in CATEGORIES.items():
        values = [
            sql_str(category_ids[cat_id]),  # id
            sql_str(cat_name),              # name
            sql_str(STORE_ID),              # store_id
            sql_str(now),                   # created_at
        ]
        cat_rows_sql.append('(' + ', '.join(values) + ')')

    cat_cols_quoted = ', '.join(f'"{c}"' for c in CAT_COLUMNS)
    cat_header = f'INSERT INTO "public"."categories" ({cat_cols_quoted}) VALUES '
    cat_body = ',\n'.join(cat_rows_sql) + ';'

    with open(OUT_SQL_CAT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(cat_header + '\n' + cat_body + '\n')
    print(f'Saved {OUT_SQL_CAT}: {len(cat_rows_sql):,} baris INSERT')

    # ── Produk ────────────────────────────────────────────────────────────────
    # Produk sudah ter-import lebih dulu (dengan id UUID acak), jadi di sini
    # kita UPDATE baris yang sudah ada, di-match lewat (store_id, sku) —
    # bukan INSERT baru — agar category_id yang tadinya NULL ikut ter-isi.
    update_stmts = []
    with open(IN_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('id'):
                continue
            name = sql_str(row['name'])
            price = sql_num(row['price'])
            category = sql_str(row['category_name']) if row['category_name'] else 'null'
            stock_qty = int(row['current_stock'])
            sku = sql_str(row['slug']) if row['slug'] else 'null'
            cat_uuid = category_ids.get(int(row['category_id'])) if row['category_id'] else None
            category_id_sql = sql_str(cat_uuid) if cat_uuid else 'null'

            set_clause = ', '.join([
                f'"name" = {name}',
                f'"price" = {price}',
                f'"category" = {category}',
                f'"stock_quantity" = {stock_qty}',
                f'"category_id" = {category_id_sql}',
                f'"updated_at" = {sql_str(now)}',
            ])
            where_clause = f'"store_id" = {sql_str(STORE_ID)} AND "sku" = {sku}'
            update_stmts.append(f'UPDATE "public"."products" SET {set_clause} WHERE {where_clause};')

    with open(OUT_SQL_PROD, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(update_stmts) + '\n')

    print(f'Saved {OUT_SQL_PROD}: {len(update_stmts):,} baris UPDATE')


if __name__ == '__main__':
    main()
