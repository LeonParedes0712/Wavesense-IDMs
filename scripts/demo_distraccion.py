"""Sesión EEG sintética con un guion de distracción; no detecta estados reales."""

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import datos


# Cada fase dura 20 segundos de señal, no de reloj en modo acelerado.
PHASES = ('ATENCION', 'DISTRACCION', 'RECUPERACION')
PHASE_SECONDS = 20


def synthetic_second(second, phase, rng):
    """Un segundo a la frecuencia de datos.py; amplitudes solo para la demo."""
    theta, beta = {'REST': (2.0, 2.0), 'ATENCION': (1.5, 3.0),
                   'DISTRACCION': (4.0, 0.8), 'RECUPERACION': (1.5, 3.0)}[phase]
    t = (second * datos.FS + np.arange(datos.FS)) / datos.FS
    return (theta * np.sin(2 * np.pi * 6 * t)
            + 2 * np.sin(2 * np.pi * 10 * t)
            + beta * np.sin(2 * np.pi * 20 * t)
            + .2 * np.sin(2 * np.pi * 35 * t)
            + rng.normal(0, .05, datos.FS))


def run(*, speed=5.0, cycles=0, sleep=time.sleep):
    """Memoria acotada; baseline único y ciclos hasta Ctrl+C o el límite indicado."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('speed debe ser positivo y finito.')
    if not isinstance(cycles, int) or cycles < 0:
        raise ValueError('cycles debe ser un entero no negativo.')
    rng = np.random.default_rng(42)
    print('[DEMO SIMULADA] Señal sintética, sin Unicorn ni modelo.', flush=True)
    print('Distracción programada por guion; no es una inferencia EEG.', flush=True)
    print(f'Velocidad: x{speed:g}. Ctrl+C para detener.', flush=True)
    baseline = []
    for second in range(datos.BASELINE_SEC):
        sleep(1 / speed)
        baseline.extend(synthetic_second(second, 'REST', rng))
        print(f'[DEMO] Baseline REST: {second + 1}/{datos.BASELINE_SEC} s', flush=True)
    _, rest_index, _ = datos.calcular_bandas(np.asarray(baseline))
    del baseline
    print(f'[DEMO] Baseline listo. Índice REST: {rest_index:.4f}', flush=True)

    buffer = deque(maxlen=datos.FS * datos.WIN_SEC)
    second = 0
    windows = 0
    while cycles == 0 or second < cycles * PHASE_SECONDS * len(PHASES):
        phase = PHASES[(second // PHASE_SECONDS) % len(PHASES)]
        phase_second = second % PHASE_SECONDS + 1
        sleep(1 / speed)
        buffer.extend(synthetic_second(datos.BASELINE_SEC + second, phase, rng))
        second += 1
        if phase_second == 1:
            print(f'\n[DEMO] Fase del guion: {phase}', flush=True)
        if len(buffer) < buffer.maxlen:
            print(f'[DEMO] Llenando ventana: {len(buffer) / datos.FS:g}/{datos.WIN_SEC} s', flush=True)
            continue
        windows += 1
        bands, index, frequency = datos.calcular_bandas(np.asarray(buffer))
        change, conclusion = datos.generar_conclusion(index, rest_index, bands, frequency)
        print(f'\n[DEMO] Ventana #{windows} | t={second}s | muestras={len(buffer)} | fs={datos.FS}', flush=True)
        print(' | '.join(f'{name}={value:.4f}' for name, value in bands.items()), flush=True)
        print(f'Índice: {index:.4f} | REST: {rest_index:.4f} | cambio: {change:+.2f}% '
              f'| frecuencia dominante: {frequency:.2f} Hz', flush=True)
        print(f'[DEMO] Resultado de datos.py: {conclusion}', flush=True)
        if phase == 'DISTRACCION':
            if phase_second < 3:
                print(f'[DEMO] Distracción programada pendiente: {phase_second}/3 actualizaciones.', flush=True)
            elif phase_second == 3:
                print('[DEMO SIMULADA] Se distrajo. Evento programado, no detección real.', flush=True)
            else:
                print('[DEMO] Continúa la fase de distracción; evento ya emitido.', flush=True)
        else:
            print(f'[DEMO] Estado del guion: {phase}', flush=True)
    print('[DEMO] Sesión finalizada.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed', type=float, default=5, help='Velocidad: 1=tiempo real, 5=demo rápida (default).')
    parser.add_argument('--cycles', type=int, default=0, help='Ciclos de 60 s simulados; 0=continuo (default).')
    args = parser.parse_args(argv)
    try:
        run(speed=args.speed, cycles=args.cycles)
    except KeyboardInterrupt:
        print('\n[DEMO] Sesión detenida por el usuario.', flush=True)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
