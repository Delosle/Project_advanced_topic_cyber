#!/usr/bin/env python3
"""
Acquisition continue des variations de tension
Appuyez sur Entrée pour arrêter et afficher les résultats
"""


import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import threading


# ================== CONFIGURATION ==================
PORT = "COM3"          # /dev/ttyUSB0 sous Linux/Mac
BAUDRATE = 2_000_000
BUFFER_SIZE = 512

# Paramètres de conversion
VREF = 1.1             # Référence ADC interne 1.1V
ADC_MAX = 1023         # 10 bits

# Fréquence d'échantillonnage ADC (prescaler=32)
ADC_FS = 38400.0       # Hz (500kHz / 13 cycles)

# MODE D'AFFICHAGE
DISPLAY_MODE = "VOLT"   # "ADC" ou "VOLT"

# PARAMÈTRES FFT
WINDOW_TYPE = "hanning"  # "hanning", "hamming", "blackman", "bartlett", "none"
FFT_MIN_FREQ = 50       # Hz - fréquence minimale à afficher
FFT_MAX_FREQ = 600      # Hz - fréquence maximale à afficher
MARK_PEAKS = True       # Marquer les pics principaux
# ===================================================


# Variable globale pour contrôler l'arrêt
stop_acquisition = False


def check_for_stop():
    """Fonction qui attend la pression de la touche Entrée"""
    global stop_acquisition
    print("\n>>> Appuyez sur ENTRÉE pour arrêter l'acquisition <<<\n")
    input()
    stop_acquisition = True
    print("\n>>> Arrêt demandé, fin de l'acquisition en cours...\n")


def sync_serial(ser):
    """Synchronisation sur l'en-tête Arduino"""
    sync = bytes([0xFF, 0xAA, 0x55, 0xFF])
    buf = bytearray()

    print("Synchronisation...", end="", flush=True)

    while len(buf) < 4:
        buf += ser.read(1)

    while buf != sync:
        buf = buf[1:] + ser.read(1)

    print(" OK")


def read_adc_samples(ser, n):
    """Lecture des échantillons ADC"""
    data = np.empty(n, dtype=np.uint16)

    for i in range(n):
        raw = ser.read(2)
        if len(raw) < 2:
            return data[:i]
        data[i] = raw[0] | (raw[1] << 8)

    return data


def adc_to_volt(adc, vref=VREF, adc_max=ADC_MAX):
    """Conversion ADC → Volts"""
    return adc.astype(np.float32) * (vref / adc_max)


def apply_window(signal, window_type="hanning"):
    """Applique une fenêtre au signal avant FFT"""
    n = len(signal)

    if window_type == "hanning":
        window = np.hanning(n)
    elif window_type == "hamming":
        window = np.hamming(n)
    elif window_type == "blackman":
        window = np.blackman(n)
    elif window_type == "bartlett":
        window = np.bartlett(n)
    else:  # none
        window = np.ones(n)

    return signal * window


def compute_fft(signal, fs=ADC_FS, window_type="hanning"):
    """Calcul de la FFT avec fenêtrage (magnitude)"""
    
    # To prevent discontinuity of the signal at the edges, that could create artifacts in the FFT 
    windowed_signal = apply_window(signal, window_type)

    # FFT
    n = len(windowed_signal)
    fft_vals = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)

    # Magnitude normalisée
    magnitude = np.abs(fft_vals) / n

    return freqs, magnitude


def find_peaks(freqs, magnitude, num_peaks=5, min_distance=10):
    """Trouve les pics principaux dans le spectre"""
    peaks = []

    # Copie pour manipulation
    mag_copy = magnitude.copy()

    # Trouver les num_peaks pics les plus importants
    for _ in range(num_peaks):
        idx = np.argmax(mag_copy)
        freq = freqs[idx]
        mag = magnitude[idx]

        # Ignorer DC et très basses fréquences
        if freq < 10:
            mag_copy[idx] = 0
            continue

        # Vérifier que c'est significatif (Au moins 5% du max)
        if mag > magnitude.max() * 0.05:
            peaks.append((freq, mag, idx))

            # Supprimer la zone autour du pic pour trouver le suivant
            start = max(0, idx - min_distance)
            end = min(len(mag_copy), idx + min_distance)
            mag_copy[start:end] = 0
        else:
            break

    return peaks


def analyze_data(data, mode="VOLT"):
    """Statistiques sur les données"""
    print("\n" + "="*60)
    if mode == "ADC":
        print("ANALYSE DES VALEURS ADC")
    else:
        print("ANALYSE DES VARIATIONS DE TENSION")
    print("="*60)

    print(f"\nNombre total d'échantillons: {len(data)}")
    print(f"Durée approximative: {len(data) / ADC_FS:.2f} s")

    if mode == "ADC":
        print(f"\nValeurs ADC:")
        print(f"  Min:        {data.min()}")
        print(f"  Max:        {data.max()}")
        print(f"  Moyenne:    {data.mean():.2f}")
        print(f"  Écart-type: {data.std():.2f}")
        print(f"  Amplitude:  {data.max() - data.min()}")

        # Calcul ENOB
        lsb = 1.0
        print(f"\nENOB approx:")
        print(f"  σ / LSB ≈ {data.std() / lsb:.2f} bits RMS")
    else:
        print(f"\nTension:")
        print(f"  Min:        {data.min():.5f} V ({data.min()*1000:.2f} mV)")
        print(f"  Max:        {data.max():.5f} V ({data.max()*1000:.2f} mV)")
        print(f"  Moyenne:    {data.mean():.5f} V ({data.mean()*1000:.2f} mV)")
        print(f"  Écart-type: {data.std():.5f} V ({data.std()*1000:.2f} mV)")
        print(f"  Amplitude:  {(data.max() - data.min())*1000:.2f} mV")


def plot_data(data, mode="VOLT"):
    """Affichage du signal temporel et FFT"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Préparer les données selon le mode
    if mode == "ADC":
        ylabel1 = "Valeur ADC"
        title_prefix = "Valeurs ADC"
        # Retirer DC offset pour FFT
        data_fft = data.astype(np.float32) - np.mean(data)
    else:
        data = data * 1000  # Conversion en mV
        ylabel1 = "Tension (mV)"
        title_prefix = "Signal"
        # Retirer DC offset pour FFT
        data_fft = (data / 1000) - np.mean(data / 1000)

    time_axis = np.arange(len(data)) / ADC_FS

    # ── Graphique 1: Signal temporel ─────────────────
    ax1.plot(time_axis, data, linewidth=0.6, color='#2E86AB', alpha=0.8)
    ax1.set_xlabel("Temps (s)", fontsize=12)
    ax1.set_ylabel(ylabel1, fontsize=12)
    ax1.set_title(f"{title_prefix} - {len(data)} échantillons", 
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.axhline(y=data.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Moyenne: {data.mean():.2f}', alpha=0.7)
    ax1.legend(fontsize=11)

    # ── Graphique 2: FFT avec fenêtrage ──────────────
    freqs, magnitude = compute_fft(data_fft, window_type=WINDOW_TYPE)

    # Filtrer la plage de fréquences à afficher
    freq_mask = (freqs >= FFT_MIN_FREQ) & (freqs <= FFT_MAX_FREQ)
    freqs_plot = freqs[freq_mask]
    magnitude_plot = magnitude[freq_mask]

    # Tracer le spectre
    ax2.plot(freqs_plot, magnitude_plot, linewidth=0.8, color='#E63946', alpha=0.7)
    ax2.fill_between(freqs_plot, magnitude_plot, alpha=0.3, color='#E63946')

    ax2.set_xlabel("Fréquence (Hz)", fontsize=12)
    ax2.set_ylabel("Magnitude", fontsize=12)
    ax2.set_title(f"Spectre fréquentiel (FFT avec fenêtre {WINDOW_TYPE})", 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(FFT_MIN_FREQ, FFT_MAX_FREQ)
    ax2.grid(alpha=0.3, linestyle='--')

    # Marquer les pics principaux avec des triangles inversés
    if MARK_PEAKS:
        peaks = find_peaks(freqs, magnitude, num_peaks=8)

        # Filtrer les pics dans la plage visible
        visible_peaks = [(f, m, i) for f, m, i in peaks 
                        if FFT_MIN_FREQ <= f <= FFT_MAX_FREQ]

        if visible_peaks:
            peak_freqs = [f for f, m, i in visible_peaks]
            peak_mags = [m for f, m, i in visible_peaks]

            # Triangles inversés rouges (comme dans l'image)
            ax2.scatter(peak_freqs, peak_mags, marker='v', s=100, 
                       color='red', edgecolors='darkred', linewidth=1.5, 
                       zorder=5, label='Pics détectés')

            # Afficher les fréquences au-dessus des pics
            for freq, mag in zip(peak_freqs[:5], peak_mags[:5]):  # Limiter à 5 pour lisibilité
                ax2.annotate(f'{freq:.1f} Hz', 
                           xy=(freq, mag), 
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', 
                                   alpha=0.7))

            ax2.legend(fontsize=10)

            # Afficher les pics dans la console
            print(f"\nPics détectés (fenêtre: {WINDOW_TYPE}):")
            for i, (freq, mag, idx) in enumerate(visible_peaks, 1):
                print(f"  {i}. {freq:.2f} Hz - Magnitude: {mag:.6f}")

    plt.tight_layout()
    plt.show()


def main():
    global stop_acquisition

    print("="*60)
    print("ACQUISITION CONTINUE DES VARIATIONS DE TENSION")
    print("="*60)
    print(f"Configuration:")
    print(f"  VREF: {VREF}V")
    print(f"  Résolution: {(VREF/ADC_MAX)*1000:.3f} mV")
    print(f"  Taille buffer: {BUFFER_SIZE} échantillons")
    print(f"  Fréquence échantillonnage: {ADC_FS:.1f} Hz")
    print(f"  Mode d'affichage: {DISPLAY_MODE}")
    print(f"  Fenêtre FFT: {WINDOW_TYPE}")
    print(f"  Plage FFT: {FFT_MIN_FREQ}-{FFT_MAX_FREQ} Hz")

    # Connexion série
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=2)
        print(f"\nConnecté sur {PORT} à {BAUDRATE} bauds")
    except serial.SerialException as e:
        print(f"Erreur de connexion: {e}")
        return

    time.sleep(2)

    # Démarrer le thread pour détecter la touche Entrée
    stop_thread = threading.Thread(target=check_for_stop, daemon=True)
    stop_thread.start()

    # Liste pour stocker tous les échantillons
    all_samples = []

    try:
        sync_serial(ser)

        buffer_count = 0

        print(f"Acquisition en cours...")
        print(f"(Les données s'accumulent en mémoire)")

        while not stop_acquisition:
            adc = read_adc_samples(ser, BUFFER_SIZE)

            if len(adc) < BUFFER_SIZE:
                print(f"\nAttention: buffer incomplet ({len(adc)} échantillons)")
                if len(adc) > 0:
                    all_samples.extend(adc)
                break

            all_samples.extend(adc)
            buffer_count += 1

            if buffer_count % 10 == 0:
                total_samples = len(all_samples)
                print(f"  Buffers reçus: {buffer_count} | "
                      f"Échantillons totaux: {total_samples} | "
                      f"Mémoire: ~{total_samples * 2 / 1024:.1f} KB", 
                      end='\r', flush=True)

        print(f"\n\nAcquisition terminée!")
        print(f"Total: {buffer_count} buffers, {len(all_samples)} échantillons")

    except KeyboardInterrupt:
        print("\n\nInterruption clavier détectée (Ctrl+C)")

    finally:
        ser.close()
        print("Connexion série fermée")

    if len(all_samples) == 0:
        print("Aucune donnée acquise!")
        return

    # Conversion en numpy array
    print("\nTraitement des données...")
    all_adc = np.array(all_samples, dtype=np.uint16)

    # Préparer les données selon le mode
    if DISPLAY_MODE == "ADC":
        data_to_analyze = all_adc
    else:
        data_to_analyze = adc_to_volt(all_adc)

    # Analyse et affichage
    analyze_data(data_to_analyze, mode=DISPLAY_MODE)

    print("\nAffichage des graphiques...")
    plot_data(data_to_analyze, mode=DISPLAY_MODE)

    # Proposer de sauvegarder
    print("\n" + "="*60)
    save = input("Voulez-vous sauvegarder les données? (o/n): ")
    if save.lower() in ['o', 'oui', 'y', 'yes']:
        filename = f"acquisition_{int(time.time())}.csv"
        if DISPLAY_MODE == "ADC":
            np.savetxt(filename, all_adc, delimiter=',', fmt='%d',
                      header='ADC', comments='')
        else:
            all_volts = adc_to_volt(all_adc)
            np.savetxt(filename, all_volts, delimiter=',', 
                      header='Tension (V)', comments='')
        print(f"Données sauvegardées dans: {filename}")

    print("\nTerminé!")


if __name__ == "__main__":
    main()
