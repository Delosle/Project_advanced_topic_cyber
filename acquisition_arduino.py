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
import sys
import select

# ================== CONFIGURATION ==================
PORT = "COM3"          # /dev/ttyUSB0 sous Linux/Mac
BAUDRATE = 2_000_000
BUFFER_SIZE = 512

# Paramètres de conversion
VREF = 1.1             # Référence ADC interne 1.1V
ADC_MAX = 1023         # 10 bits

# ===================================================

# Variable globale pour contrôler l'arrêt
stop_acquisition = False


def check_for_stop():
    """Fonction qui attend la pression de la touche Entrée"""
    global stop_acquisition
    print("\n>>> Appuyez sur ENTRÉE pour arrêter l'acquisition <<<\n")
    input()  # Bloque jusqu'à ce que l'utilisateur appuie sur Entrée
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


def analyze_voltage(all_volts):
    """Statistiques sur les variations de tension"""
    print("\n" + "="*60)
    print("ANALYSE DES VARIATIONS DE TENSION")
    print("="*60)
    
    print(f"\nNombre total d'échantillons: {len(all_volts)}")
    print(f"Durée approximative: {len(all_volts) / (BAUDRATE / (BUFFER_SIZE * 16)):.2f} s")
    
    print(f"\nTension:")
    print(f"  Min:        {all_volts.min():.5f} V ({all_volts.min()*1000:.2f} mV)")
    print(f"  Max:        {all_volts.max():.5f} V ({all_volts.max()*1000:.2f} mV)")
    print(f"  Moyenne:    {all_volts.mean():.5f} V ({all_volts.mean()*1000:.2f} mV)")
    print(f"  Écart-type: {all_volts.std():.5f} V ({all_volts.std()*1000:.2f} mV)")
    print(f"  Amplitude:  {all_volts.max() - all_volts.min():.5f} V")
    print(f"              {(all_volts.max() - all_volts.min())*1000:.2f} mV")


def plot_voltage(all_volts):
    """Affichage des variations de tension"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Convertir en mV pour meilleure lisibilité
    volts_mv = all_volts * 1000
    
    # Graphique 1: Signal complet
    ax1.plot(volts_mv, linewidth=0.6, color='#2E86AB', alpha=0.8)
    ax1.set_xlabel("Échantillon", fontsize=12)
    ax1.set_ylabel("Tension (mV)", fontsize=12)
    ax1.set_title(f"Signal complet - {len(all_volts)} échantillons", 
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.axhline(y=volts_mv.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Moyenne: {volts_mv.mean():.2f} mV', alpha=0.7)
    ax1.legend(fontsize=11)
    
    # Graphique 2: Zoom sur les derniers 2000 échantillons
    zoom_samples = min(2000, len(volts_mv))
    volts_zoom = volts_mv[-zoom_samples:]
    
    ax2.plot(range(len(volts_mv) - zoom_samples, len(volts_mv)), 
             volts_zoom, linewidth=0.8, color='#E63946')
    ax2.set_xlabel("Échantillon", fontsize=12)
    ax2.set_ylabel("Tension (mV)", fontsize=12)
    ax2.set_title(f"Zoom sur les {zoom_samples} derniers échantillons", 
                  fontsize=13)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.axhline(y=volts_zoom.mean(), color='green', linestyle='--', linewidth=1.5,
                label=f'Moyenne (zoom): {volts_zoom.mean():.2f} mV', alpha=0.7)
    ax2.legend(fontsize=11)
    
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
    
    # Connexion série
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=2)
        print(f"\nConnecté sur {PORT} à {BAUDRATE} bauds")
    except serial.SerialException as e:
        print(f"Erreur de connexion: {e}")
        return
    
    time.sleep(2)  # Attendre reset Arduino
    
    # Démarrer le thread pour détecter la touche Entrée
    stop_thread = threading.Thread(target=check_for_stop, daemon=True)
    stop_thread.start()
    
    # Liste pour stocker tous les échantillons
    all_samples = []
    
    try:
        # Synchronisation initiale
        sync_serial(ser)
        
        buffer_count = 0
        
        print(f"Acquisition en cours...")
        print(f"(Les données s'accumulent en mémoire)")
        
        while not stop_acquisition:
            # Lire un buffer
            adc = read_adc_samples(ser, BUFFER_SIZE)
            
            if len(adc) < BUFFER_SIZE:
                print(f"\nAttention: buffer incomplet ({len(adc)} échantillons)")
                if len(adc) > 0:
                    all_samples.extend(adc)
                break
            
            # Ajouter à la liste globale
            all_samples.extend(adc)
            buffer_count += 1
            
            # Afficher la progression toutes les 10 buffers
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
    
    # Vérifier qu'on a des données
    if len(all_samples) == 0:
        print("Aucune donnée acquise!")
        return
    
    # Conversion en numpy array et en volts
    print("\nTraitement des données...")
    all_adc = np.array(all_samples, dtype=np.uint16)
    all_volts = adc_to_volt(all_adc)
    
    # Analyse et affichage
    analyze_voltage(all_volts)
    
    print("\nAffichage des graphiques...")
    plot_voltage(all_volts)
    
    # Proposer de sauvegarder
    print("\n" + "="*60)
    save = input("Voulez-vous sauvegarder les données? (o/n): ")
    if save.lower() in ['o', 'oui', 'y', 'yes']:
        filename = f"acquisition_{int(time.time())}.csv"
        np.savetxt(filename, all_volts, delimiter=',', 
                   header='Tension (V)', comments='')
        print(f"Données sauvegardées dans: {filename}")
    
    print("\nTerminé!")


if __name__ == "__main__":
    main()