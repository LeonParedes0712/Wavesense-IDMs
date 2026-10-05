"""Diagnóstico directo UnicornPy: 10 bloques de 25 muestras, sin pipeline."""

import importlib
import os
from pathlib import Path
import sys

import numpy as np


def select_serial(devices, requested=None):
    devices = list(devices or [])
    if not devices:
        raise RuntimeError('No hay Unicorn disponible: enciende y empareja el dispositivo.')
    if requested:
        if requested in devices:
            return requested
        raise RuntimeError(f'UNICORN_SERIAL no disponible. Disponibles: {devices}')
    if len(devices) != 1:
        raise RuntimeError(f'Configura UNICORN_SERIAL. Disponibles: {devices}')
    return devices[0]


def acquire(sdk, serial=None):
    """Secuencia de origin/master:datos.py; el SDK se inyecta solo para pruebas."""
    serial = select_serial(sdk.GetAvailableDevices(True), serial)
    device = sdk.Unicorn(serial)
    numero_canales = device.GetNumberOfAcquiredChannels()
    frame_length = 25
    buffer_length = frame_length * numero_canales * 4
    receive_buffer = bytearray(buffer_length)
    version = getattr(sdk, 'GetApiVersion', None)
    if callable(version):
        print(f'API: {version()}')
    print(f'Serial: {serial}')
    print(f'SamplingRate: {sdk.SamplingRate}')
    print(f'Canales adquiridos: {numero_canales}')
    for number in range(1, 9):
        name = f'EEG {number}'
        try:
            index = device.GetChannelIndex(name)
        except Exception as exc:
            print(f'{name}: no disponible ({exc})')
        else:
            print(f'{name}: índice {index}' if 0 <= index < numero_canales
                  else f'{name}: no disponible (índice {index})')
    try:
        device.StartAcquisition(False)
        for _ in range(10):
            device.GetData(frame_length, receive_buffer, buffer_length)
            datos = np.frombuffer(receive_buffer, dtype=np.float32).reshape(
                frame_length, numero_canales)
            print(f'Forma: {datos.shape}; mínimo: {datos.min()}; máximo: {datos.max()}')
            print(f'Primeras 3 muestras, primeros 8 canales:\n{datos[:3, :8]}')
    finally:
        device.StopAcquisition()


def main():
    if sys.platform != 'win32':
        print('Este diagnóstico requiere Windows + Unicorn Suite/UnicornPy.', file=sys.stderr)
        return 1
    # .env local opcional; las variables de entorno tienen prioridad.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    dll_directory = None
    added_path = None
    try:
        sdk_path = os.getenv('UNICORN_PYTHON_PATH', '').strip()
        if sdk_path:
            path = Path(sdk_path).expanduser().resolve()
            if not path.is_dir():
                raise RuntimeError('UNICORN_PYTHON_PATH no es un directorio existente.')
            dll_directory = os.add_dll_directory(str(path))
            added_path = str(path)
            sys.path.insert(0, added_path)
        sdk = importlib.import_module('UnicornPy')
        acquire(sdk, os.getenv('UNICORN_SERIAL', '').strip() or None)
        return 0
    except KeyboardInterrupt:
        print('Diagnóstico interrumpido.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error Unicorn: {exc}. Revisa UNICORN_PYTHON_PATH, Python de 64 bits '
              'compatible, Unicorn Suite y licencia activa.', file=sys.stderr)
        return 1
    finally:
        if added_path is not None:
            sys.path.remove(added_path)
        if dll_directory is not None:
            dll_directory.close()


if __name__ == '__main__':
    raise SystemExit(main())
