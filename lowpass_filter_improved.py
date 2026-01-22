import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# ═══════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════

SAMPLING_FREQ = 38400     # Hz (fréquence réelle Arduino prescaler=32)

# Paramètres du filtre passe-bas
LOWPASS_CUTOFF = 60     # Hz - fréquence de coupure (augmentée de 500 → 1500)
FILTER_ORDER = 4          # Ordre du filtre (2-8 recommandé)
FILTER_TYPE = "butter"    # "butter", "bessel", "cheby1"

# Option: comparer plusieurs fréquences de coupure
COMPARE_CUTOFFS = False   # True pour comparer plusieurs cutoffs
CUTOFF_LIST = [500, 1000, 1500, 2000, 5000]  # Hz

# Gestion DC offset
REMOVE_DC_OFFSET = True   # IMPORTANT: retirer le DC avant filtrage

CSV_FILE = "acquisition_1769100716.csv"


# ═══════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════

def load_csv_voltage(filepath):
    df = pd.read_csv(filepath, comment='#', header=0)
    signal_data = df.iloc[:, 0].astype(float).values
    print(f" {len(signal_data)} samples chargés")
    return signal_data


# ═══════════════════════════════════════
# FILTRES
# ═══════════════════════════════════════

def lowpass_filter(signal_data, cutoff_freq, fs, order=4, filter_type="butter", remove_dc=True):
    """
    Applique un filtre passe-bas IIR avec gestion du DC offset

    Paramètres:
        signal_data: signal d'entrée
        cutoff_freq: fréquence de coupure (Hz)
        fs: fréquence d'échantillonnage (Hz)
        order: ordre du filtre
        filter_type: 'butter', 'bessel', 'cheby1'
        remove_dc: retirer le DC offset avant filtrage

    Retourne:
        filtered_signal, dc_offset
    """
    # Sauvegarder le DC offset original
    dc_offset = np.mean(signal_data) if remove_dc else 0.0

    # Retirer le DC offset AVANT le filtrage
    if remove_dc:
        signal_centered = signal_data - dc_offset
        print(f"  DC offset retiré: {dc_offset*1000:.3f} mV")
    else:
        signal_centered = signal_data.copy()

    nyquist = fs / 2.0
    normal_cutoff = cutoff_freq / nyquist

    # Conception du filtre
    if filter_type == "butter":
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == "bessel":
        b, a = signal.bessel(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == "cheby1":
        b, a = signal.cheby1(order, 0.5, normal_cutoff, btype='low', analog=False)
    else:
        raise ValueError(f"Type de filtre inconnu: {filter_type}")

    # Application du filtre (filtfilt = zéro déphasage)
    filtered_centered = signal.filtfilt(b, a, signal_centered)

    # Option: rajouter le DC offset
    # Pour l'analyse EM, on garde généralement le signal centré sur 0
    # Si besoin de restaurer le DC, décommenter la ligne suivante:
    # filtered = filtered_centered + dc_offset

    return filtered_centered, dc_offset


def compute_fft(signal_data, fs):
    """Calcule la FFT du signal (avec DC centré)"""
    n = len(signal_data)
    # Retirer DC pour FFT
    signal_centered = signal_data - np.mean(signal_data)
    fft_vals = np.fft.rfft(signal_centered)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(fft_vals) / n
    return freqs, magnitude


# ═══════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════

def plot_filter_response(cutoff, fs, order, filter_type):
    """Affiche la réponse fréquentielle du filtre"""
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist

    if filter_type == "butter":
        b, a = signal.butter(order, normal_cutoff, btype='low')
    elif filter_type == "bessel":
        b, a = signal.bessel(order, normal_cutoff, btype='low')
    elif filter_type == "cheby1":
        b, a = signal.cheby1(order, 0.5, normal_cutoff, btype='low')

    w, h = signal.freqz(b, a, worN=2000)
    freq_hz = w * fs / (2 * np.pi)

    plt.figure(figsize=(10, 5))
    plt.plot(freq_hz, 20 * np.log10(abs(h)), 'b', linewidth=2)
    plt.axvline(cutoff, color='red', linestyle='--', linewidth=2, label=f'Cutoff: {cutoff} Hz')
    plt.axhline(-3, color='green', linestyle='--', label='-3 dB', alpha=0.7)
    plt.xlabel('Fréquence (Hz)', fontsize=12)
    plt.ylabel('Gain (dB)', fontsize=12)
    plt.title(f'Réponse fréquentielle - Filtre {filter_type.upper()} ordre {order}', 
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim(0, min(5000, fs/2))
    plt.ylim(-80, 5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_results(raw, filtered, fs, cutoff, dc_offset):
    """Affiche le signal brut vs filtré (temporel + fréquentiel)"""
    t = np.arange(len(raw)) / fs

    # Calculer FFT
    freqs_raw, fft_raw = compute_fft(raw, fs)
    freqs_filt, fft_filt = compute_fft(filtered, fs)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # ── Signal brut (temporel) ──────────────
    ax1.plot(t, raw * 1000, linewidth=0.5, label="Signal brut", alpha=0.7)
    ax1.axhline(y=dc_offset*1000, color='red', linestyle='--', linewidth=1.5,
                label=f'DC offset: {dc_offset*1000:.2f} mV', alpha=0.7)
    ax1.set_title("Signal brut", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Tension (mV)", fontsize=12)
    ax1.set_xlabel("Temps (s)", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Signal filtré (temporel) ────────────
    ax2.plot(t, filtered * 1000, linewidth=0.8, label="Signal filtré (DC centré)", color='orange')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
                label='DC = 0 mV', alpha=0.7)
    ax2.set_title(f"Signal filtré (passe-bas {cutoff} Hz, DC centré)", 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel("Tension (mV)", fontsize=12)
    ax2.set_xlabel("Temps (s)", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── FFT signal brut ─────────────────────
    freq_mask = freqs_raw <= 5000
    ax3.plot(freqs_raw[freq_mask], fft_raw[freq_mask], linewidth=0.8, alpha=0.7)
    ax3.axvline(cutoff, color='red', linestyle='--', label=f'Cutoff: {cutoff} Hz', linewidth=2)
    ax3.set_title("Spectre - Signal brut", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Magnitude", fontsize=12)
    ax3.set_xlabel("Fréquence (Hz)", fontsize=12)
    ax3.set_xlim(0, 5000)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ── FFT signal filtré ───────────────────
    ax4.plot(freqs_filt[freq_mask], fft_filt[freq_mask], linewidth=0.8, color='orange')
    ax4.axvline(cutoff, color='red', linestyle='--', label=f'Cutoff: {cutoff} Hz', linewidth=2)
    ax4.set_title("Spectre - Signal filtré", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Magnitude", fontsize=12)
    ax4.set_xlabel("Fréquence (Hz)", fontsize=12)
    ax4.set_xlim(0, 5000)
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_cutoff_frequencies(raw_signal, fs, cutoff_list, order, filter_type):
    """Compare plusieurs fréquences de coupure"""
    n_cutoffs = len(cutoff_list)
    fig, axes = plt.subplots(n_cutoffs, 2, figsize=(16, 4*n_cutoffs))

    t = np.arange(len(raw_signal)) / fs

    for i, cutoff in enumerate(cutoff_list):
        # Filtrer
        filtered, dc = lowpass_filter(raw_signal, cutoff, fs, order, filter_type, remove_dc=True)

        # FFT
        freqs_raw, fft_raw = compute_fft(raw_signal, fs)
        freqs_filt, fft_filt = compute_fft(filtered, fs)
        freq_mask = freqs_raw <= 5000

        # Signal temporel
        ax_time = axes[i, 0] if n_cutoffs > 1 else axes[0]
        ax_time.plot(t, filtered * 1000, linewidth=0.8, color='C'+str(i))
        ax_time.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax_time.set_title(f"Signal filtré - Cutoff {cutoff} Hz", fontsize=12, fontweight='bold')
        ax_time.set_ylabel("Tension (mV)", fontsize=11)
        ax_time.set_xlabel("Temps (s)", fontsize=11)
        ax_time.grid(alpha=0.3)

        # FFT
        ax_fft = axes[i, 1] if n_cutoffs > 1 else axes[1]
        ax_fft.plot(freqs_filt[freq_mask], fft_filt[freq_mask], linewidth=0.8, color='C'+str(i))
        ax_fft.axvline(cutoff, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_fft.set_title(f"Spectre - Cutoff {cutoff} Hz", fontsize=12, fontweight='bold')
        ax_fft.set_ylabel("Magnitude", fontsize=11)
        ax_fft.set_xlabel("Fréquence (Hz)", fontsize=11)
        ax_fft.set_xlim(0, 5000)
        ax_fft.grid(alpha=0.3)

        # Stats
        std_raw = (raw_signal - np.mean(raw_signal)).std()
        std_filt = filtered.std()
        noise_reduction = (1 - std_filt/std_raw) * 100
        ax_time.text(0.02, 0.98, f'σ: {std_filt*1000:.3f} mV\nRéduction: {noise_reduction:.1f}%',
                    transform=ax_time.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

if __name__ == "__main__":

    # Charger les données
    raw_signal = load_csv_voltage(CSV_FILE)

    print(f"\nConfiguration du filtre:")
    print(f"  Type: {FILTER_TYPE}")
    print(f"  Fréquence de coupure: {LOWPASS_CUTOFF} Hz")
    print(f"  Ordre: {FILTER_ORDER}")
    print(f"  Fréquence d'échantillonnage: {SAMPLING_FREQ} Hz")
    print(f"  Retrait DC offset: {'OUI ' if REMOVE_DC_OFFSET else 'NON '}")

    if COMPARE_CUTOFFS:
        # Mode comparaison
        print(f"\n🔍 Mode COMPARAISON de plusieurs cutoffs: {CUTOFF_LIST}")
        compare_cutoff_frequencies(raw_signal, SAMPLING_FREQ, CUTOFF_LIST, 
                                   FILTER_ORDER, FILTER_TYPE)
    else:
        # Mode normal
        # Afficher la réponse du filtre
        print(f"\nAffichage de la réponse fréquentielle...")
        plot_filter_response(LOWPASS_CUTOFF, SAMPLING_FREQ, FILTER_ORDER, FILTER_TYPE)

        # Appliquer le filtre passe-bas
        print(f"\nApplication du filtre passe-bas...")
        filtered_signal, dc_offset = lowpass_filter(raw_signal, LOWPASS_CUTOFF, SAMPLING_FREQ, 
                                                     FILTER_ORDER, FILTER_TYPE, REMOVE_DC_OFFSET)

        print(f"Filtrage terminé")
        print(f"\nStatistiques:")
        print(f"  DC offset original: {dc_offset*1000:.3f} mV")
        print(f"  Signal brut - σ: {(raw_signal - np.mean(raw_signal)).std()*1000:.3f} mV")
        print(f"  Signal filtré - σ: {filtered_signal.std()*1000:.3f} mV")

        std_raw = (raw_signal - np.mean(raw_signal)).std()
        std_filt = filtered_signal.std()
        noise_reduction = (1 - std_filt/std_raw) * 100
        print(f"  Réduction du bruit: {noise_reduction:.1f}%")

        # Afficher les résultats
        plot_results(raw_signal, filtered_signal, SAMPLING_FREQ, LOWPASS_CUTOFF, dc_offset)

        # Sauvegarder le signal filtré
        save = input("\nVoulez-vous sauvegarder le signal filtré? (o/n): ")
        if save.lower() in ['o', 'oui', 'y', 'yes']:
            output_file = CSV_FILE.replace('.csv', f'_filtered_{LOWPASS_CUTOFF}Hz.csv')
            np.savetxt(output_file, filtered_signal, delimiter=',', 
                      header=f'Tension filtrée (V) - Cutoff {LOWPASS_CUTOFF} Hz - DC centré', 
                      comments='')
            print(f"Signal filtré sauvegardé: {output_file}")
            print(f"   Note: DC offset retiré ({dc_offset*1000:.3f} mV)")
