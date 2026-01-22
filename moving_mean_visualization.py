import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ═══════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════

SAMPLING_FREQ = 38400     # Hz
MOVING_MEAN_WINDOW = 100  # taille de la fenêtre pour la moyenne mobile

CSV_FILE = "acquisition_1769101642.csv"

# ═══════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════

def load_csv_voltage(filepath):
    df = pd.read_csv(filepath, comment='#', header=0)
    signal_data = df.iloc[:, 0].astype(float).values
    print(f"📂 {len(signal_data)} samples chargés")
    return signal_data

# ═══════════════════════════════════════
# MOYENNE MOBILE
# ═══════════════════════════════════════


def moving_mean(signal_data, window_size):
    
    # Moyenne mobile via convolution : chaque point devient la moyenne de ses N voisins
    # np.ones(N)/N crée le noyau [1/N, 1/N, ..., 1/N], mode='same' conserve la longueur
    return np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')

# ═══════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════

def plot_results(raw, filtered, fs):
    # Axe des temps basé sur la fréquence d'échantillonnage
    t = np.arange(len(raw)) / fs

    fig, (ax_raw, ax_filtered) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── Signal brut ─────────────────────
    ax_raw.plot(t, raw, linewidth=0.5, label="Signal brut")
    ax_raw.set_title("Signal brut")
    ax_raw.set_ylabel("Tension (V)")
    ax_raw.grid(alpha=0.3)
    ax_raw.legend()

    # ── Signal filtré par moyenne mobile ──────
    ax_filtered.plot(t, filtered, linewidth=0.8, label="Signal filtré (moyenne mobile)", color='orange')
    ax_filtered.set_title("Signal filtré par moyenne mobile")
    ax_filtered.set_ylabel("Tension (V)")
    ax_filtered.set_xlabel("Temps (s)")
    ax_filtered.grid(alpha=0.3)
    ax_filtered.legend()

    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

if __name__ == "__main__":

    raw_signal = load_csv_voltage(CSV_FILE)

    filtered_signal = moving_mean(raw_signal, MOVING_MEAN_WINDOW)

    print(f" Filtrage par moyenne mobile appliqué (fenêtre = {MOVING_MEAN_WINDOW})")

    plot_results(raw_signal, filtered_signal, SAMPLING_FREQ)
