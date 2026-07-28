"""
Ekstrak data produk & stok dari SQL dump Eatstedi.

Parsing dilakukan langsung dari file .sql (tanpa server MySQL), memakai
parser char-level yang sama dengan Section 1 main.ipynb / generate_notebook.py.

Tabel yang diambil:
  - products      : id, supplier_id, category_id, name, slug, price
  - daily_stocks  : product_id, sold (terjual, snapshot), quantity (stok saat ini)
  - categories    : id -> name (hardcode, sama seperti main.ipynb)

Catatan: tabel `stocks` (riwayat stok per transaksi) kosong di dump ini,
jadi tidak ikut diekstrak.

Output: Data/Ekstrak/products.csv
"""
import os
import pandas as pd

SQL_PATH = os.path.join('Data', 'Ekstrak', 'eatstedi-20260621-010820.sql')
OUT_CSV  = os.path.join('Data', 'Ekstrak', 'products.csv')

CATEGORIES = {
    1: 'MAKANAN KERING',
    2: 'MINUMAN',
    3: 'SNACKS',
    4: 'ICE CREAM',
    5: 'MAKANAN BASAH',
    6: 'MERCH',
}


def _parse_sql_tuples(text):
    """Character-level parser untuk VALUES dalam INSERT SQL."""
    Q, BS, NL, TB, CR = chr(39), chr(92), chr(10), chr(9), chr(13)

    rows = []
    i = 0
    n = len(text)

    while i < n:
        while i < n and text[i] != '(':
            i += 1
        if i >= n:
            break
        i += 1

        vals = []
        while i < n and text[i] != ')':
            while i < n and text[i] in (' ', NL, TB, CR):
                i += 1
            if i >= n or text[i] == ')':
                break

            if text[i] == Q:
                i += 1
                buf = []
                while i < n:
                    c = text[i]
                    if c == BS and i + 1 < n:
                        nc = text[i + 1]
                        i += 2
                        if   nc == Q:   buf.append(Q)
                        elif nc == 'n': buf.append(NL)
                        elif nc == 't': buf.append(TB)
                        elif nc == 'r': buf.append(CR)
                        elif nc == BS:  buf.append(BS)
                        else:           buf.append(nc)
                    elif c == Q:
                        i += 1
                        break
                    else:
                        buf.append(c)
                        i += 1
                vals.append(''.join(buf))
            elif text[i:i+4] == 'NULL':
                vals.append(None)
                i += 4
            else:
                buf = []
                while i < n and text[i] not in (',', ')', ' ', NL, TB, CR):
                    buf.append(text[i])
                    i += 1
                vals.append(''.join(buf) if buf else None)

            while i < n and text[i] in (' ', NL, TB, CR):
                i += 1
            if i < n and text[i] == ',':
                i += 1

        if i < n and text[i] == ')':
            i += 1
        if vals:
            rows.append(tuple(vals))

    return rows


def parse_sql_table(sql_path, table_name):
    """Stream-baca SQL dump, ekstrak INSERT block untuk table_name."""
    lines = []
    capturing = False

    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.rstrip()
            if f'LOCK TABLES `{table_name}` WRITE' in s:
                capturing = True
                continue
            if capturing and 'UNLOCK TABLES' in s:
                break
            if capturing:
                skip = (
                    not s
                    or s.startswith('--')
                    or s.startswith('/*!')
                    or s.startswith('SET ')
                    or s.startswith('INSERT INTO')
                )
                if not skip:
                    lines.append(s)

    return _parse_sql_tuples(' '.join(lines))


def main():
    print('Parsing products ...')
    prod_cols = ['id', 'supplier_id', 'category_id', 'picture', 'name', 'slug',
                 'price', 'created_at', 'updated_at']
    df_prod = pd.DataFrame(parse_sql_table(SQL_PATH, 'products'), columns=prod_cols)
    df_prod['id']            = df_prod['id'].astype(int)
    df_prod['category_id']   = df_prod['category_id'].astype(int)
    df_prod['category_name'] = df_prod['category_id'].map(CATEGORIES)
    df_prod['price']         = pd.to_numeric(df_prod['price'], errors='coerce')
    print(f'  -> {len(df_prod):,} produk')

    print('Parsing daily_stocks ...')
    ds_cols = ['id', 'product_id', 'sold', 'quantity', 'created_at', 'updated_at']
    df_stock = pd.DataFrame(parse_sql_table(SQL_PATH, 'daily_stocks'), columns=ds_cols)
    df_stock['product_id'] = df_stock['product_id'].astype(int)
    df_stock['sold']       = pd.to_numeric(df_stock['sold'], errors='coerce').fillna(0).astype(int)
    df_stock['quantity']   = pd.to_numeric(df_stock['quantity'], errors='coerce').fillna(0).astype(int)
    print(f'  -> {len(df_stock):,} baris stok')

    df_out = df_prod.merge(
        df_stock[['product_id', 'sold', 'quantity']],
        left_on='id', right_on='product_id', how='left', suffixes=('', '_stock')
    ).drop(columns=['product_id'])
    df_out = df_out.rename(columns={'sold': 'total_sold', 'quantity': 'current_stock'})
    df_out[['total_sold', 'current_stock']] = df_out[['total_sold', 'current_stock']].fillna(0).astype(int)

    df_out = df_out[['id', 'name', 'slug', 'category_id', 'category_name',
                      'supplier_id', 'price', 'current_stock', 'total_sold']]

    df_out.to_csv(OUT_CSV, index=False)
    print(f'Saved {OUT_CSV}: {len(df_out):,} produk')
    print(df_out.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
