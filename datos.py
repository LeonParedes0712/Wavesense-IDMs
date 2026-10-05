import os
import sys
import time

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt


# ============================================================
# UNICORN PYTHON API
# Cambia esta ruta si tu instalación está en otro lugar
# ============================================================

UNICORN_PATH = (
    r"C:\Users\alerm\Documents\gtec\Unicorn Suite"
    r"\Hybrid Black\Unicorn Python\Lib"
)

os.add_dll_directory(UNICORN_PATH)
sys.path.insert(0, UNICORN_PATH)

import UnicornPy


# ============================================================
# CONFIGURACIÓN
# ============================================================

FS = UnicornPy.SamplingRate       # normalmente 250 Hz

FRAME_LENGTH = 25                # 25 muestras = 0.1 s
WIN_SEC = 10                     # ventana móvil
WIN_SAMPLES = FS * WIN_SEC

BASELINE_SEC = 20                # REST inicial
BASELINE_SAMPLES = FS * BASELINE_SEC

UPDATE_SEC = 1                   # análisis cada segundo
UPDATE_SAMPLES = FS * UPDATE_SEC

UMBRAL_CONCENTRACION = 5.0       # +5 % respecto a REST

EEG_CHANNEL_INDEX = 0            # Canal EEG que analizaremos

TEST_SIGNAL = False


# ============================================================
# FILTRO
# ============================================================

def aplicar_filtro(data):

    # Trabajamos entre 1 y 40 Hz.
    # Gamma se limitará a 30-40 Hz.

    sos = butter(
        4,
        [1, 40],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    return sosfiltfilt(sos, data)


# ============================================================
# CALCULAR BANDAS EEG
# ============================================================

def calcular_bandas(signal):

    signal = np.asarray(signal)

    signal = signal - np.mean(signal)

    signal = aplicar_filtro(signal)

    freqs, psd = welch(
        signal,
        fs=FS,
        nperseg=min(512, len(signal))
    )

    def potencia(fmin, fmax):

        idx = (freqs >= fmin) & (freqs < fmax)

        if not np.any(idx):
            return 0.0

        return np.trapezoid(
            psd[idx],
            freqs[idx]
        )

    bandas = {
        "delta": potencia(1, 4),
        "theta": potencia(4, 8),
        "alpha": potencia(8, 12),
        "beta": potencia(12, 30),
        "gamma": potencia(30, 40)
    }

    # Índice experimental de atención
    indice = (
        bandas["beta"] + bandas["gamma"]
    ) / (
        bandas["theta"] + 1e-12
    )

    # Frecuencia con mayor potencia
    rango = (freqs >= 1) & (freqs <= 40)

    frecuencia_dominante = freqs[rango][
        np.argmax(psd[rango])
    ]

    return bandas, indice, frecuencia_dominante


# ============================================================
# CONCLUSIÓN
# ============================================================

def generar_conclusion(
    indice_actual,
    indice_rest,
    bandas,
    frecuencia_dominante
):

    cambio = (
        (indice_actual - indice_rest)
        / (indice_rest + 1e-12)
    ) * 100

    if cambio >= UMBRAL_CONCENTRACION:

        conclusion = (
            "✅ ACTIVACIÓN MAYOR AL REPOSO"
        )

    elif cambio <= -UMBRAL_CONCENTRACION:

        conclusion = (
            "🚨 ACTIVACIÓN MENOR AL REPOSO"
        )

    else:

        conclusion = (
            "➖ ACTIVIDAD SIMILAR AL REPOSO"
        )

    return cambio, conclusion


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print()
    print("🧠 WAVESENSE - EEG EN TIEMPO REAL")
    print("=" * 70)

    # --------------------------------------------------------
    # BUSCAR UNICORN
    # --------------------------------------------------------

    dispositivos = UnicornPy.GetAvailableDevices(True)

    if dispositivos is None or len(dispositivos) == 0:

        print("❌ No se encontró ningún Unicorn.")
        return

    print("\nDispositivos disponibles:")

    for i, d in enumerate(dispositivos):
        print(f"#{i}: {d}")

    device_id = int(
        input("\nSelecciona dispositivo por ID: ")
    )

    device = UnicornPy.Unicorn(
        dispositivos[device_id]
    )

    numero_canales = (
        device.GetNumberOfAcquiredChannels()
    )

    print()
    print(
        f"Conectado a: {dispositivos[device_id]}"
    )
    print(
        f"Frecuencia de muestreo: {FS} Hz"
    )
    print(
        f"Canales adquiridos: {numero_canales}"
    )

    print(
        f"Canal analizado: #{EEG_CHANNEL_INDEX + 1}"
    )

    # --------------------------------------------------------
    # BUFFER DEL UNICORN
    # --------------------------------------------------------

    buffer_length = (
        FRAME_LENGTH *
        numero_canales *
        4
    )

    receive_buffer = bytearray(
        buffer_length
    )

    # --------------------------------------------------------
    # BUFFER PARA EEG
    # --------------------------------------------------------

    eeg_buffer = []

    try:

        device.StartAcquisition(
            TEST_SIGNAL
        )

        print()
        print("=" * 70)

        print(
            f"🧘 BASELINE REST: "
            f"permanece relajado durante "
            f"{BASELINE_SEC} segundos."
        )

        print("=" * 70)

        # ====================================================
        # BASELINE REST
        # ====================================================

        while len(eeg_buffer) < BASELINE_SAMPLES:

            device.GetData(
                FRAME_LENGTH,
                receive_buffer,
                buffer_length
            )

            datos = np.frombuffer(
                receive_buffer,
                dtype=np.float32
            ).reshape(
                FRAME_LENGTH,
                numero_canales
            )

            canal = datos[
                :,
                EEG_CHANNEL_INDEX
            ]

            eeg_buffer.extend(
                canal.tolist()
            )

            segundos = (
                len(eeg_buffer) / FS
            )

            print(
                f"\rREST: "
                f"{segundos:5.1f} / "
                f"{BASELINE_SEC} s",
                end=""
            )

        print()

        # ----------------------------------------------------
        # CALCULAR REFERENCIA REST
        # ----------------------------------------------------

        rest_signal = np.array(
            eeg_buffer[-BASELINE_SAMPLES:]
        )

        bandas_rest, indice_rest, freq_rest = (
            calcular_bandas(rest_signal)
        )

        print()
        print("✅ BASELINE OBTENIDO")
        print(
            f"Índice REST: "
            f"{indice_rest:.4f}"
        )

        print(
            f"Frecuencia dominante REST: "
            f"{freq_rest:.2f} Hz"
        )

        print()
        print("=" * 70)
        print(
            "🧠 ANÁLISIS EN TIEMPO REAL"
        )
        print(
            "Presiona CTRL+C para detener."
        )
        print("=" * 70)

        # Empezamos ventana nueva
        eeg_buffer = []

        muestras_desde_analisis = 0

        # ====================================================
        # LOOP EN TIEMPO REAL
        # ====================================================

        while True:

            device.GetData(
                FRAME_LENGTH,
                receive_buffer,
                buffer_length
            )

            datos = np.frombuffer(
                receive_buffer,
                dtype=np.float32
            ).reshape(
                FRAME_LENGTH,
                numero_canales
            )

            canal = datos[
                :,
                EEG_CHANNEL_INDEX
            ]

            eeg_buffer.extend(
                canal.tolist()
            )

            muestras_desde_analisis += (
                FRAME_LENGTH
            )

            # Mantener solo últimos 10 segundos
            if len(eeg_buffer) > WIN_SAMPLES:

                eeg_buffer = eeg_buffer[
                    -WIN_SAMPLES:
                ]

            # Todavía no tenemos 10 s
            if len(eeg_buffer) < WIN_SAMPLES:

                faltan = (
                    WIN_SAMPLES -
                    len(eeg_buffer)
                ) / FS

                print(
                    f"\rLlenando ventana..."
                    f" faltan {faltan:4.1f}s",
                    end=""
                )

                continue

            # Analizar cada segundo
            if muestras_desde_analisis < UPDATE_SAMPLES:
                continue

            muestras_desde_analisis = 0

            # ------------------------------------------------
            # ANÁLISIS
            # ------------------------------------------------

            ventana = np.array(
                eeg_buffer
            )

            bandas, indice, freq_dom = (
                calcular_bandas(
                    ventana
                )
            )

            cambio, conclusion = (
                generar_conclusion(
                    indice,
                    indice_rest,
                    bandas,
                    freq_dom
                )
            )

            # ------------------------------------------------
            # MOSTRAR RESULTADOS
            # ------------------------------------------------

            print("\n")
            print("-" * 70)

            print(
                f"θ Theta  4-8 Hz : "
                f"{bandas['theta']:.4f}"
            )

            print(
                f"α Alpha  8-12 Hz: "
                f"{bandas['alpha']:.4f}"
            )

            print(
                f"β Beta  12-30 Hz: "
                f"{bandas['beta']:.4f}"
            )

            print(
                f"γ Gamma 30-40 Hz: "
                f"{bandas['gamma']:.4f}"
            )

            print()

            print(
                f"Frecuencia dominante: "
                f"{freq_dom:.2f} Hz"
            )

            print(
                f"Índice actual: "
                f"{indice:.4f}"
            )

            print(
                f"Índice REST:   "
                f"{indice_rest:.4f}"
            )

            print(
                f"Cambio vs REST: "
                f"{cambio:+.2f}%"
            )

            print()
            print(
                f"CONCLUSIÓN: {conclusion}"
            )

            print("-" * 70)

    # ========================================================
    # CTRL + C
    # ========================================================

    except KeyboardInterrupt:

        print()
        print()
        print(
            "🛑 Adquisición detenida "
            "por el usuario."
        )

    finally:

        try:
            device.StopAcquisition()
        except:
            pass

        del device

        print(
            "Unicorn desconectado."
        )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    main()