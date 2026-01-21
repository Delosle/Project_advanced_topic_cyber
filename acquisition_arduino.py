#!/usr/bin/env python3
"""
Acquisition ultra-rapide des variations de tension
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
import time

# ================== CONFIGURATION ==================
PORT = "COM3"          # /dev/ttyUSB0 sous Linux/Mac
BAUDRATE = 2_000_000
BUFFER_SIZE = 512

# Paramètres de conversion
VREF = 1.1             # Référence ADC (AVCC = 5V)
ADC_MAX = 1023         # 10 bits

# ===================================================


def sync_serial(ser):
    """Synchronisation sur l'en-tête Arduino"""
    sync = bytes([0xFF, 0xAA, 0x55, 0xFF])
    buf = bytearray()
    
    print("Attente de synchronisation...", end="", flush=True)
    
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
            print(f"\nAvertissement: données incomplètes à l'échantillon {i}")
            return data[:i]
        data[i] = raw[0] | (raw[1] << 8)
    
    return data


def adc_to_volt(adc, vref=VREF, adc_max=ADC_MAX):
    """Conversion ADC → Volts"""
    return adc.astype(np.float32) * (vref / adc_max)


def analyze_voltage(volts):
    """Statistiques sur les variations de tension"""
    print("\n" + "="*50)
    print("VARIATIONS DE TENSION")
    print("="*50)
    
    print(f"\nTension:")
    print(f"  Min:        {volts.min():.4f} V")
    print(f"  Max:        {volts.max():.4f} V")
    print(f"  Moyenne:    {volts.mean():.4f} V")
    print(f"  Écart-type: {volts.std():.4f} V")
    print(f"  Amplitude:  {volts.max() - volts.min():.4f} V")


def plot_voltage(volts):
    """Affichage des variations de tension"""
    plt.figure(figsize=(14, 6))
    
    plt.plot(volts, linewidth=0.8, color='#2E86AB')
    plt.xlabel("Échantillon", fontsize=12)
    plt.ylabel("Tension (V)", fontsize=12)
    plt.title("Variation de tension (ADC → Volt)", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    
    # Ligne de moyenne
    plt.axhline(y=volts.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Moyenne: {volts.mean():.4f} V', alpha=0.7)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def main():
    print("="*50)
    print("ACQUISITION DES VARIATIONS DE TENSION")
    print("="*50)
    
    # Connexion série
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=2)
        print(f"Connecté sur {PORT} à {BAUDRATE} bauds")
    except serial.SerialException as e:
        print(f"Erreur de connexion: {e}")
        return
    
    time.sleep(2)  # Attendre reset Arduino
    
    # Acquisition
    sync_serial(ser)
    print(f"Acquisition de {BUFFER_SIZE} échantillons...")
    adc = read_adc_samples(ser, BUFFER_SIZE)
    ser.close()
    
    if len(adc) < BUFFER_SIZE:
        print(f"Attention: seulement {len(adc)} échantillons acquis")
    
    # Conversion en tension
    volts = adc_to_volt(adc)
    
    # Analyse et affichage
    analyze_voltage(volts)
    plot_voltage(volts)


if __name__ == "__main__":
    main()