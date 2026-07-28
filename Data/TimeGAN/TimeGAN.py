"""
TimeGAN Pipeline untuk augmentasi daily_sales.csv
Referensi: Yoon et al., "Time-series Generative Adversarial Networks" (NeurIPS 2019)

Alur:
  daily_sales.csv  →  normalisasi  →  sliding windows
  →  latih TimeGAN (3 fase)
  →  generate synthetic sequences
  →  inverse transform  →  synthetic_daily.csv

STATUS: cabang eksperimen, TIDAK dipakai di jalur produksi (main.ipynb, app.py,
app_public.py, v2/*, evaluation/backtest.py sama sekali tidak mereferensikan
modul atau output modul ini). Hasil perbandingan di
Models/TimeGAN/comparison_results.csv menunjukkan LSTM+TimeGAN mengungguli
LSTM baseline murni, tapi masih kalah dari baseline Seasonal-Naive-5 pada
MASE (2.51 vs 2.14) — sama seperti model LSTM utama yang juga TIDAK LULUS
kriteria produksi vs seasonal_naive (lihat Dokumen/M3 - Hasil Backtest.md).
Dipertahankan di repo sebagai eksperimen yang didokumentasikan gagal/inconclusive,
bukan sebagai bagian dari pipeline yang direkomendasikan.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ─── Config ───────────────────────────────────────────────────────────────────

TIMEGAN_DIR  = os.path.dirname(os.path.abspath(__file__))   # Data/TimeGAN/
DATA_DIR     = os.path.dirname(TIMEGAN_DIR)                  # Data/
DAILY_CSV    = os.path.join(DATA_DIR, 'Ekstrak', 'daily_sales.csv')
OUTPUT_CSV   = os.path.join(TIMEGAN_DIR, 'synthetic_daily.csv')
MODEL_DIR    = TIMEGAN_DIR

SEQ_LEN      = 24       # panjang window (hari); ~1 bulan
NUM_FEAT     = 3        # revenue, transactions, qty_sold
HIDDEN_DIM   = 24
NUM_LAYERS   = 3
BATCH_SIZE   = 32
EPOCHS_EMB   = 600      # fase 1: autoencoder
EPOCHS_SUP   = 600      # fase 2: supervisor
EPOCHS_JNT   = 600      # fase 3: joint training
LR           = 1e-3
GAMMA        = 1.0      # bobot supervised loss pada joint training
N_SYNTHETIC  = 200      # jumlah synthetic sequences yang di-generate
SEED         = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kolom numerik yang akan disintesis
NUM_COLS     = ['revenue', 'transactions', 'qty_sold']
# Kolom kalender yang akan di-derive ulang dari tanggal sintetis
CAL_COLS     = ['day_of_week', 'week_of_year', 'month', 'year',
                'is_weekend', 'is_holiday', 'is_ramadan']

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Data Loading & Preprocessing ─────────────────────────────────────────

def load_and_scale(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[NUM_COLS].values.astype(np.float32))
    return df, data_scaled, scaler


def make_windows(data_scaled, seq_len):
    """Buat sliding windows dari data time series."""
    windows = []
    n = len(data_scaled)
    for i in range(n - seq_len + 1):
        windows.append(data_scaled[i : i + seq_len])
    return np.array(windows, dtype=np.float32)   # (N, seq_len, num_feat)


def random_noise(batch_size, seq_len, num_feat):
    return torch.rand(batch_size, seq_len, num_feat, device=DEVICE)


# ─── 2. Arsitektur Jaringan ───────────────────────────────────────────────────

class EmbedderNet(nn.Module):
    """Memetakan data nyata X → representasi laten H."""
    def __init__(self):
        super().__init__()
        self.rnn  = nn.GRU(NUM_FEAT, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Sigmoid())

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.proj(h)


class RecoveryNet(nn.Module):
    """Merekonstruksi data X̂ dari laten H."""
    def __init__(self):
        super().__init__()
        self.rnn  = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(HIDDEN_DIM, NUM_FEAT), nn.Sigmoid())

    def forward(self, h):
        out, _ = self.rnn(h)
        return self.proj(out)


class GeneratorNet(nn.Module):
    """Memetakan noise Z → laten sintetis Ê."""
    def __init__(self):
        super().__init__()
        self.rnn  = nn.GRU(NUM_FEAT, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Sigmoid())

    def forward(self, z):
        h, _ = self.rnn(z)
        return self.proj(h)


class SupervisorNet(nn.Module):
    """Mengawasi dinamika temporal pada laten H → Ĥ."""
    def __init__(self):
        super().__init__()
        self.rnn  = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS - 1, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Sigmoid())

    def forward(self, h):
        out, _ = self.rnn(h)
        return self.proj(out)


class DiscriminatorNet(nn.Module):
    """Membedakan laten nyata vs sintetis."""
    def __init__(self):
        super().__init__()
        self.rnn  = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS,
                           batch_first=True, bidirectional=True)
        self.proj = nn.Linear(HIDDEN_DIM * 2, 1)

    def forward(self, h):
        out, _ = self.rnn(h)
        return self.proj(out)


# ─── 3. Fungsi Loss ───────────────────────────────────────────────────────────

bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()


def discriminator_loss(d_real, d_fake, d_fake_sup):
    loss_real     = bce_loss(d_real,     torch.ones_like(d_real))
    loss_fake     = bce_loss(d_fake,     torch.zeros_like(d_fake))
    loss_fake_sup = bce_loss(d_fake_sup, torch.zeros_like(d_fake_sup))
    return loss_real + loss_fake + GAMMA * loss_fake_sup


def generator_loss(d_fake, d_fake_sup, h_hat_sup, h, x_hat, x):
    # Adversarial: fool discriminator
    loss_unsup    = bce_loss(d_fake,     torch.ones_like(d_fake))
    loss_unsup_e  = bce_loss(d_fake_sup, torch.ones_like(d_fake_sup))
    # Supervised: matching temporal dynamics
    loss_sup      = mse_loss(h_hat_sup[:, :-1, :], h[:, 1:, :])
    # Moment: matching mean & variance per feature
    loss_mean     = mse_loss(x_hat.mean(dim=0), x.mean(dim=0))
    loss_var      = mse_loss(x_hat.var(dim=0),  x.var(dim=0))
    return loss_unsup + loss_unsup_e + GAMMA * loss_sup + 100 * (loss_mean + loss_var)


# ─── 4. Training ──────────────────────────────────────────────────────────────

def train_embedder(E, R, loader, epochs):
    """Fase 1: Latih Embedder + Recovery (autoencoder)."""
    opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=LR)
    print(f"\n[Fase 1] Autoencoder pretraining — {epochs} epoch")
    for ep in range(1, epochs + 1):
        total = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            h    = E(x)
            x_hat = R(h)
            loss = 10 * torch.sqrt(mse_loss(x, x_hat))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 100 == 0:
            print(f"  Epoch {ep:4d}/{epochs} | Recon Loss: {total/len(loader):.4f}")


def train_supervisor(E, S, loader, epochs):
    """Fase 2: Latih Supervisor pada laten nyata."""
    opt = torch.optim.Adam(list(E.parameters()) + list(S.parameters()), lr=LR)
    print(f"\n[Fase 2] Supervisor pretraining — {epochs} epoch")
    for ep in range(1, epochs + 1):
        total = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            h        = E(x)
            h_hat    = S(h)
            loss_sup = mse_loss(h_hat[:, :-1, :], h[:, 1:, :])
            opt.zero_grad(); loss_sup.backward(); opt.step()
            total += loss_sup.item()
        if ep % 100 == 0:
            print(f"  Epoch {ep:4d}/{epochs} | Supervisor Loss: {total/len(loader):.4f}")


def train_joint(E, R, G, S, D, loader, epochs):
    """Fase 3: Joint training semua komponen."""
    opt_G = torch.optim.Adam(list(G.parameters()) + list(S.parameters()), lr=LR)
    opt_E = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)

    print(f"\n[Fase 3] Joint training — {epochs} epoch")
    for ep in range(1, epochs + 1):
        g_total = d_total = 0.0

        for (x,) in loader:
            x = x.to(DEVICE)
            bs = x.size(0)

            # ── Generator step (×2 per discriminator step) ──
            for _ in range(2):
                z         = random_noise(bs, SEQ_LEN, NUM_FEAT)
                h         = E(x)
                h_hat     = G(z)
                h_hat_sup = S(h_hat)
                x_hat     = R(h_hat)

                d_fake     = D(h_hat)
                d_fake_sup = D(h_hat_sup)
                h_hat_s    = S(h)

                loss_g = generator_loss(d_fake, d_fake_sup, h_hat_sup, h, x_hat, x)
                opt_G.zero_grad(); loss_g.backward(); opt_G.step()

                # Embedder update dengan supervised loss
                h         = E(x)
                h_hat_s   = S(h)
                x_hat     = R(h)
                loss_e    = (10 * torch.sqrt(mse_loss(x, x_hat))
                             + 0.1 * mse_loss(h_hat_s[:, :-1, :], h[:, 1:, :]))
                opt_E.zero_grad(); loss_e.backward(); opt_E.step()

                g_total += loss_g.item()

            # ── Discriminator step ──
            z         = random_noise(bs, SEQ_LEN, NUM_FEAT)
            h         = E(x).detach()
            h_hat     = G(z).detach()
            h_hat_sup = S(G(z)).detach()

            d_real     = D(h)
            d_fake     = D(h_hat)
            d_fake_sup = D(h_hat_sup)
            loss_d     = discriminator_loss(d_real, d_fake, d_fake_sup)

            # Hanya update D jika loss masih signifikan (hindari over-training D)
            if loss_d.item() > 0.15:
                opt_D.zero_grad(); loss_d.backward(); opt_D.step()

            d_total += loss_d.item()

        if ep % 100 == 0:
            print(f"  Epoch {ep:4d}/{epochs} | G Loss: {g_total/len(loader)/2:.4f} "
                  f"| D Loss: {d_total/len(loader):.4f}")


# ─── 5. Generate & Post-Processing ───────────────────────────────────────────

def generate_sequences(G, R, n):
    """Generate n synthetic sequences, return numpy (n, seq_len, num_feat)."""
    G.eval(); R.eval()
    seqs = []
    with torch.no_grad():
        for _ in range(0, n, BATCH_SIZE):
            bs = min(BATCH_SIZE, n - len(seqs))
            z  = random_noise(bs, SEQ_LEN, NUM_FEAT)
            h  = G(z)
            x  = R(h)
            seqs.append(x.cpu().numpy())
    return np.concatenate(seqs, axis=0)   # (n, seq_len, num_feat)


def derive_calendar(date_series):
    """Derive kembali fitur kalender dari kolom date."""
    df = pd.DataFrame({'date': pd.to_datetime(date_series)})
    df['day_of_week']  = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month']        = df['date'].dt.month
    df['year']         = df['date'].dt.year
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday']   = df['month'].isin([1, 7]).astype(int)
    df['is_ramadan']   = (
        ((df['year'] == 2025) & (df['month'] == 3)) |
        ((df['year'] == 2026) & (df['month'] == 3))
    ).astype(int)
    return df.drop(columns=['date'])


def sequences_to_dataframe(sequences, scaler, ref_df):
    """
    Konversi synthetic sequences → DataFrame dengan format daily_sales.csv.
    Tanggal sintetis diambil secara acak dari rentang data nyata (with replacement).
    """
    rows = []
    all_dates = ref_df['date'].values

    for seq in sequences:
        # Pilih start date acak dari data nyata
        start_idx = np.random.randint(0, len(all_dates) - SEQ_LEN + 1)
        dates     = pd.date_range(
            start=all_dates[start_idx],
            periods=SEQ_LEN,
            freq='D'
        )
        inv       = scaler.inverse_transform(seq)           # (seq_len, 3)
        # Pastikan nilai non-negatif & integer wajar
        inv[:, 0] = np.clip(inv[:, 0], 0, None)            # revenue
        inv[:, 1] = np.clip(np.round(inv[:, 1]), 1, None)  # transactions
        inv[:, 2] = np.clip(np.round(inv[:, 2]), 1, None)  # qty_sold

        cal = derive_calendar(dates)
        for i, d in enumerate(dates):
            row = {'date': d.date()}
            row.update({c: inv[i, j] for j, c in enumerate(NUM_COLS)})
            row.update(cal.iloc[i].to_dict())
            rows.append(row)

    df_syn = pd.DataFrame(rows)
    # Urutkan kolom sama persis dengan daily_sales.csv
    col_order = ['date'] + NUM_COLS + CAL_COLS
    return df_syn[col_order]


def save_models(E, R, G, S, D):
    torch.save(E.state_dict(), os.path.join(MODEL_DIR, 'embedder.pt'))
    torch.save(R.state_dict(), os.path.join(MODEL_DIR, 'recovery.pt'))
    torch.save(G.state_dict(), os.path.join(MODEL_DIR, 'generator.pt'))
    torch.save(S.state_dict(), os.path.join(MODEL_DIR, 'supervisor.pt'))
    torch.save(D.state_dict(), os.path.join(MODEL_DIR, 'discriminator.pt'))
    print(f"\nModel disimpan di: {MODEL_DIR}")


def load_models():
    E = EmbedderNet().to(DEVICE)
    R = RecoveryNet().to(DEVICE)
    G = GeneratorNet().to(DEVICE)
    S = SupervisorNet().to(DEVICE)
    D = DiscriminatorNet().to(DEVICE)
    E.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'embedder.pt'),      map_location=DEVICE))
    R.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'recovery.pt'),      map_location=DEVICE))
    G.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'generator.pt'),     map_location=DEVICE))
    S.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'supervisor.pt'),    map_location=DEVICE))
    D.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'discriminator.pt'), map_location=DEVICE))
    return E, R, G, S, D


# ─── 6. Main ──────────────────────────────────────────────────────────────────

def run_training():
    print(f"Device: {DEVICE}")
    print(f"Data: {DAILY_CSV}")

    # Load & preprocess
    df, data_scaled, scaler = load_and_scale(DAILY_CSV)
    print(f"Baris data nyata : {len(df)}")
    print(f"Rentang           : {df['date'].min().date()} – {df['date'].max().date()}")

    windows = make_windows(data_scaled, SEQ_LEN)
    print(f"Jumlah windows    : {len(windows)}  (seq_len={SEQ_LEN})")

    dataset = TensorDataset(torch.tensor(windows))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Inisialisasi model
    E = EmbedderNet().to(DEVICE)
    R = RecoveryNet().to(DEVICE)
    G = GeneratorNet().to(DEVICE)
    S = SupervisorNet().to(DEVICE)
    D = DiscriminatorNet().to(DEVICE)

    # Tiga fase training
    train_embedder(E, R, loader, EPOCHS_EMB)
    train_supervisor(E, S, loader, EPOCHS_SUP)
    train_joint(E, R, G, S, D, loader, EPOCHS_JNT)

    save_models(E, R, G, S, D)
    return E, R, G, S, D, scaler, df


def run_generation(G=None, R=None, scaler=None, df=None):
    """Generate synthetic data. Jika model belum ada di memori, load dari disk."""
    if G is None or R is None:
        E, R, G, S, D = load_models()
    if scaler is None or df is None:
        df, data_scaled, scaler = load_and_scale(DAILY_CSV)

    print(f"\nMen-generate {N_SYNTHETIC} synthetic sequences …")
    sequences = generate_sequences(G, R, N_SYNTHETIC)

    df_syn = sequences_to_dataframe(sequences, scaler, df)
    df_syn.to_csv(OUTPUT_CSV, index=False)
    print(f"Synthetic data disimpan : {OUTPUT_CSV}")
    print(f"Total baris sintetis    : {len(df_syn)}")
    print(f"\nSampel:")
    print(df_syn.head(5).to_string(index=False))
    return df_syn


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TimeGAN untuk daily_sales.csv')
    parser.add_argument('--mode', choices=['train', 'generate', 'all'],
                        default='all',
                        help='train: hanya latih | generate: hanya generate | all: keduanya')
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        E, R, G, S, D, scaler, df = run_training()
        if args.mode == 'all':
            run_generation(G=G, R=R, scaler=scaler, df=df)
    elif args.mode == 'generate':
        run_generation()
