"""
Bangun batch SQL INSERT untuk public.transactions & public.transaction_items
dari Data/Ekstrak/raw_transactions.csv, untuk store_id di Supabase.

Mapping id produk lama (integer, dari SQL dump) -> UUID produk baru dilakukan
lewat file JSON product_map.json (slug -> uuid), yang diambil dari query:
    SELECT id, sku FROM public.products WHERE store_id = '<STORE_ID>';

Transaksi lama tidak punya kolom payment_method/cashier eksplisit yang cocok
skema baru, jadi:
  - payment_method = 'Tunai' (default, sesuai keputusan)
  - cashier_id     = NULL
  - local_id       = invoice_id lama (untuk traceability / idempotency)

Output: Data/Ekstrak/tx_batches/tx_<n>.sql dan items_<n>.sql, batch_size baris
per file agar tiap batch bisa dieksekusi satu-per-satu lewat MCP execute_sql.
"""
import csv
import json
import os
import uuid

STORE_ID   = '49e30ecd-48a0-441d-864d-0251323adc7b'
RAW_CSV    = os.path.join('Data', 'Ekstrak', 'raw_transactions.csv')
MAP_JSON   = os.path.join('Data', 'Ekstrak', 'product_map.json')
OUT_DIR    = os.path.join('Data', 'Ekstrak', 'tx_batches')
BATCH_SIZE = 1000

os.makedirs(OUT_DIR, exist_ok=True)


def sql_str(v):
    return "'" + str(v).replace("'", "''") + "'"


def main():
    with open(MAP_JSON, encoding='utf-8') as f:
        slug_to_uuid = json.load(f)

    with open(os.path.join('Data', 'Ekstrak', 'products.csv'), newline='', encoding='utf-8') as f:
        prod_id_to_slug = {int(r['id']): r['slug'] for r in csv.DictReader(f)}

    invoices = {}   # invoice_id -> dict(succeeded_at, total_price, new_uuid)
    items_by_invoice = {}  # invoice_id -> list of item rows

    with open(RAW_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv_id = row['invoice_id']
            if inv_id not in invoices:
                invoices[inv_id] = {
                    'succeeded_at': row['succeeded_at'],
                    'total_price': row['total_price'],
                    'uuid': str(uuid.uuid4()),
                }
                items_by_invoice[inv_id] = []

            slug = prod_id_to_slug.get(int(row['product_id']))
            product_uuid = slug_to_uuid.get(slug)
            qty = int(float(row['quantity']))
            price = float(row['price'])
            items_by_invoice[inv_id].append({
                'product_id': product_uuid,
                'product_name': row['name'],
                'product_sku': slug,
                'unit_price': price,
                'quantity': qty,
                'subtotal': price * qty,
            })

    inv_ids = list(invoices.keys())
    print(f'Total invoices: {len(inv_ids):,}')

    # ── Transactions batches ──────────────────────────────────────────────────
    tx_cols = ['id', 'local_id', 'store_id', 'total_amount', 'payment_method',
               'status', 'created_at']
    n_tx_batches = 0
    for start in range(0, len(inv_ids), BATCH_SIZE):
        chunk = inv_ids[start:start + BATCH_SIZE]
        rows = []
        for inv_id in chunk:
            inv = invoices[inv_id]
            values = [
                sql_str(inv['uuid']),
                sql_str(inv_id),
                sql_str(STORE_ID),
                f"{float(inv['total_price']):.2f}",
                "'Tunai'",
                "'Berhasil'",
                sql_str(inv['succeeded_at']),
            ]
            rows.append('(' + ','.join(values) + ')')
        cols_q = ','.join(f'"{c}"' for c in tx_cols)
        sql = f'INSERT INTO "public"."transactions" ({cols_q}) VALUES ' + ','.join(rows) + ';'
        with open(os.path.join(OUT_DIR, f'tx_{n_tx_batches:03d}.sql'), 'w', encoding='utf-8') as f:
            f.write(sql)
        n_tx_batches += 1
    print(f'Transactions: {n_tx_batches} batch file(s)')

    # ── Transaction items batches ─────────────────────────────────────────────
    item_cols = ['id', 'transaction_id', 'product_id', 'product_name',
                 'product_sku', 'unit_price', 'quantity', 'subtotal']
    all_items = []
    for inv_id in inv_ids:
        tx_uuid = invoices[inv_id]['uuid']
        for it in items_by_invoice[inv_id]:
            all_items.append((tx_uuid, it))

    print(f'Total items: {len(all_items):,}')
    n_item_batches = 0
    for start in range(0, len(all_items), BATCH_SIZE):
        chunk = all_items[start:start + BATCH_SIZE]
        rows = []
        for tx_uuid, it in chunk:
            pid_sql = sql_str(it['product_id']) if it['product_id'] else 'null'
            values = [
                sql_str(str(uuid.uuid4())),
                sql_str(tx_uuid),
                pid_sql,
                sql_str(it['product_name']),
                sql_str(it['product_sku']) if it['product_sku'] else 'null',
                f"{it['unit_price']:.2f}",
                str(it['quantity']),
                f"{it['subtotal']:.2f}",
            ]
            rows.append('(' + ','.join(values) + ')')
        cols_q = ','.join(f'"{c}"' for c in item_cols)
        sql = f'INSERT INTO "public"."transaction_items" ({cols_q}) VALUES ' + ','.join(rows) + ';'
        with open(os.path.join(OUT_DIR, f'items_{n_item_batches:03d}.sql'), 'w', encoding='utf-8') as f:
            f.write(sql)
        n_item_batches += 1
    print(f'Transaction items: {n_item_batches} batch file(s)')


if __name__ == '__main__':
    main()
