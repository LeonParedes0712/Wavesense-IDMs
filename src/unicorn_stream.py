"""Fuente UnicornPy diferida y ventanas con fuente inyectable, sin efectos al importar."""

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Protocol

import numpy as np

from src.eeg_config import EEGConfig


class AcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampleBlock:
    samples: np.ndarray
    sample_rate: float
    has_artifact: bool = False


class SampleSource(Protocol):
    def connect(self): ...
    def read_samples(self, count: int) -> SampleBlock: ...
    def close(self): ...


class UnicornSource:
    """Usar dentro de WindowStream o llamar close() en finally.

    Sin importación de UnicornPy hasta discover/connect. UNICORN_EEG_CHANNEL_NAMES define
    el orden físico entregado como Canal_1..8, no una selección por posición.
    """

    def __init__(self, *, serial=None, python_path=None, channel_names=None):
        self.serial = serial
        self.python_path = python_path
        self.channel_names = tuple((f'EEG {i}' for i in range(1, 9))
                                   if channel_names is None else channel_names)
        if (len(self.channel_names) != 8 or len(set(self.channel_names)) != 8
                or any(not isinstance(name, str) or not name.strip() for name in self.channel_names)):
            raise ValueError('UNICORN_EEG_CHANNEL_NAMES requiere ocho nombres únicos separados por comas.')
        self._sdk = None
        self._device = None
        self._dll_directory = None
        self._started = False
        self._buffer = bytearray()
        self.sample_rate = None

    @classmethod
    def from_env(cls):
        names = os.getenv('UNICORN_EEG_CHANNEL_NAMES', '').strip()
        return cls(serial=os.getenv('UNICORN_SERIAL', '').strip() or None,
                   python_path=os.getenv('UNICORN_PYTHON_PATH', '').strip() or None,
                   channel_names=tuple(n.strip() for n in names.split(',')) if names else None)

    def _load_sdk(self):
        if self._sdk is not None:
            return self._sdk
        if sys.platform != 'win32':
            raise AcquisitionError('Unicorn requiere Windows + Unicorn Suite/UnicornPy; usa ArraySource en Linux/CI.')
        added_path = None
        try:
            if self.python_path:
                path = Path(self.python_path).expanduser().resolve()
                if not path.is_dir():
                    raise AcquisitionError('UNICORN_PYTHON_PATH no es un directorio existente del SDK.')
                self._dll_directory = os.add_dll_directory(str(path))
                added_path = str(path)
                sys.path.insert(0, added_path)
            self._sdk = importlib.import_module('UnicornPy')
            return self._sdk
        except AcquisitionError:
            self._close_after_error()
            raise
        except Exception as exc:
            self._close_after_error()
            raise AcquisitionError(
                'No se pudo cargar UnicornPy/DLL: instala Unicorn Suite, verifica su licencia y usa un Python compatible '
                'con su versión/arquitectura; configura UNICORN_PYTHON_PATH si no está en la ruta.'
            ) from exc
        finally:
            if added_path is not None:
                sys.path.remove(added_path)

    def discover(self):
        sdk = self._load_sdk()
        try:
            return list(sdk.GetAvailableDevices(True) or [])
        except Exception as exc:
            self._close_after_error()
            raise AcquisitionError('No se pudieron descubrir dispositivos: revisa Bluetooth y emparejamiento.') from exc

    def connect(self):
        if self._device is not None:
            raise AcquisitionError('La fuente Unicorn ya está conectada.')
        try:
            devices = self.discover()
            if not devices:
                raise AcquisitionError('No hay Unicorn disponible: enciende y empareja el dispositivo en Windows.')
            if self.serial is None:
                if len(devices) != 1:
                    raise AcquisitionError('Hay varios dispositivos: configura UNICORN_SERIAL explícitamente.')
                serial = devices[0]
            else:
                serial = self.serial
                if serial not in devices:
                    raise AcquisitionError('UNICORN_SERIAL no está disponible; revisa emparejamiento.')
            self._device = self._sdk.Unicorn(serial)
            self.sample_rate = float(self._sdk.SamplingRate)
            self._channel_count = self._device.GetNumberOfAcquiredChannels()
            self._indices = tuple(self._device.GetChannelIndex(name) for name in self.channel_names)
            if (len(set(self._indices)) != 8
                    or any(i < 0 or i >= self._channel_count for i in self._indices)):
                raise AcquisitionError('Revisa canales EEG habilitados y UNICORN_EEG_CHANNEL_NAMES.')
            # Marcar antes de Start para intentar Stop incluso si el SDK falla a medio inicio.
            self._started = True
            self._device.StartAcquisition(False)
        except BaseException as exc:
            self._close_after_error()
            if isinstance(exc, (AcquisitionError, KeyboardInterrupt, SystemExit)):
                raise
            raise AcquisitionError('No se pudo iniciar Unicorn: revisa SDK, canales habilitados y conexión.') from exc

    def read_samples(self, count):
        if self._device is None or not self._started:
            raise AcquisitionError('Conecta Unicorn antes de leer muestras.')
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('count debe ser un entero positivo.')
        try:
            size = count * self._channel_count * np.dtype(np.float32).itemsize
            if len(self._buffer) != size:
                self._buffer = bytearray(size)
            self._device.GetData(count, self._buffer, size)
            scans = np.frombuffer(self._buffer, dtype=np.float32).reshape(count, self._channel_count)
            return SampleBlock(scans[:, self._indices].copy(), self.sample_rate)
        except BaseException as exc:
            self._close_after_error()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise AcquisitionError('Falló GetData: adquisición cerrada; revisa enlace Bluetooth y reinicia la sesión.') from exc

    def _close_after_error(self):
        try:
            self.close()
        except Exception:
            # Conservar el error original; close libera referencias incluso si Stop falla.
            pass

    def close(self):
        device, started = self._device, self._started
        self._device, self._started = None, False
        try:
            if device is not None and started:
                device.StopAcquisition()
        except Exception as exc:
            raise AcquisitionError('StopAcquisition falló; referencias liberadas. Revisa el dispositivo antes de reconectar.') from exc
        finally:
            # UnicornPy desconecta al destruir la instancia; no expone Disconnect.
            device = None
            self._buffer = bytearray()
            self.sample_rate = None
            self._sdk = None
            if self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None


class ArraySource:
    """Fuente finita inyectable en memoria; nunca guarda datos ni crea un modelo."""

    def __init__(self, samples, *, sample_rate):
        self.samples = np.asarray(samples)
        self.sample_rate = sample_rate
        self._position = 0
        self.connected = False

    def connect(self):
        if self.connected:
            raise AcquisitionError('ArraySource ya está conectada.')
        self._position = 0
        self.connected = True

    def read_samples(self, count):
        if not self.connected:
            raise AcquisitionError('Conecta ArraySource antes de leer.')
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('count debe ser un entero positivo.')
        if self._position >= len(self.samples):
            raise EOFError('Fuente simulada agotada.')
        result = self.samples[self._position:self._position + count].copy()
        self._position += len(result)
        return SampleBlock(result, self.sample_rate)

    def close(self):
        self.connected = False


class WindowStream:
    """Acumula ventanas exactas con solapamiento, sin loop de fondo.

    Cada next_window() hace un número finito de lecturas. Las fuentes deben
    devolver entre 1 y count muestras contiguas, o lanzar una excepción.
    Una lectura nativa puede bloquear según el SDK; no se inventa un timeout.
    """

    def __init__(self, source: SampleSource, *, config: EEGConfig, chunk_samples=25):
        if not isinstance(chunk_samples, int) or isinstance(chunk_samples, bool) or chunk_samples <= 0:
            raise ValueError('chunk_samples debe ser un entero positivo.')
        self.source, self.config, self.chunk_samples = source, config, chunk_samples
        self._samples = np.empty((0, config.channel_count))
        self._bad = np.empty(0, dtype=bool)
        self._open = False

    def __enter__(self):
        if self._open:
            raise AcquisitionError('WindowStream ya está abierto.')
        try:
            self.source.connect()
            self._open = True
            return self
        except BaseException:
            self._close_after_error()
            raise

    def next_window(self):
        if not self._open:
            raise AcquisitionError('Usa WindowStream dentro de with.')
        try:
            while len(self._samples) < self.config.window_samples:
                count = min(self.chunk_samples, self.config.window_samples - len(self._samples))
                block = self.source.read_samples(count)
                array = np.asarray(block.samples)
                if (array.ndim != 2 or array.shape[1] != self.config.channel_count
                        or not 1 <= len(array) <= count or array.dtype.kind not in 'iuf'):
                    raise AcquisitionError('La fuente debe entregar de 1 a count filas numéricas de ocho canales.')
                if (not np.isfinite(block.sample_rate)
                        or not np.isclose(block.sample_rate, self.config.sample_rate, rtol=0, atol=1e-8)):
                    raise AcquisitionError('Frecuencia del SDK/fuente incompatible con EEG_SAMPLE_RATE; no se remuestrea.')
                self._samples = np.concatenate((self._samples, array))
                self._bad = np.concatenate((self._bad, np.full(len(array), block.has_artifact, dtype=bool)))
            result = SampleBlock(self._samples.copy(), self.config.sample_rate, bool(self._bad.any()))
            self._samples = self._samples[self.config.hop_samples:].copy()
            self._bad = self._bad[self.config.hop_samples:].copy()
            return result
        except BaseException:
            self._close_after_error()
            raise

    def _close_after_error(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        self._open = False
        self._samples = np.empty((0, self.config.channel_count))
        self._bad = np.empty(0, dtype=bool)
        self.source.close()

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self._close_after_error()
        return False
