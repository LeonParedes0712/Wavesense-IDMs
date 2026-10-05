"""Validación, calidad y filtrado separados; sin adquisición ni clasificación."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.eeg_config import EEGConfig, QualityConfig


@dataclass(frozen=True)
class SignalQuality:
    has_artifact: bool
    reasons: tuple[str, ...] = ()


def validate_window(window, *, sample_rate, config: EEGConfig):
    """Devuelve copia float64 (muestras, canales); rechaza ventanas incompletas."""
    if (not np.isfinite(sample_rate) or sample_rate <= 0
            or not np.isclose(sample_rate, config.sample_rate, rtol=0, atol=1e-8)):
        raise ValueError(f'Frecuencia incompatible: se requieren {config.sample_rate} Hz.')
    try:
        array = np.asarray(window)
        if array.dtype.kind not in 'iuf':
            raise ValueError('Se requieren números reales.')
        array = array.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('EEG debe ser una matriz de números reales.') from exc
    expected = (config.window_samples, config.channel_count)
    if array.shape != expected:
        raise ValueError(f'Ventana EEG: forma requerida {expected}; recibida {array.shape}.')
    if not np.isfinite(array).all():
        raise ValueError('Ventana EEG contiene NaN/inf.')
    return array


def assess_quality(window, *, sample_rate, config: EEGConfig, quality: QualityConfig):
    """Inspecciona señal sin filtrar; cualquier canal malo invalida la ventana.

    No detecta todos los artefactos EEG/EMG. Los límites no tienen interpretación
    clínica y se expresan en las unidades numéricas de la señal recibida.
    """
    array = validate_window(window, sample_rate=sample_rate, config=config)
    reasons = []
    with np.errstate(over='ignore', invalid='ignore'):
        spans = np.ptp(array, axis=0)
        steps = np.abs(np.diff(array, axis=0))
    if not np.isfinite(spans).all() or not np.isfinite(steps).all():
        reasons.append('numeric_overflow')
    for channel in np.flatnonzero(spans <= quality.flatline_ptp):
        reasons.append(f'flatline:Canal_{channel + 1}')
    if quality.max_abs is not None and (np.abs(array) > quality.max_abs).any():
        reasons.append('amplitude_limit')
    if quality.max_step is not None and (steps > quality.max_step).any():
        reasons.append('step_limit')
    return SignalQuality(bool(reasons), tuple(reasons))


def filter_window(window, *, sample_rate, config: EEGConfig):
    """Referencia opcional y Butterworth SOS adelante/atrás por ventana completa.

    Default provisional: orden 4, 1–40 Hz; no es un filtro causal continuo.
    El padding odd usa la fórmula documentada por SciPy, explicitada aquí.
    """
    array = validate_window(window, sample_rate=sample_rate, config=config)
    if config.reference == 'common_average':
        array = array - array.mean(axis=1, keepdims=True)
    if config.filter_enabled:
        sos = butter(config.filter_order, [config.filter_low, config.filter_high],
                     btype='bandpass', fs=config.sample_rate, output='sos')
        padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if len(array) <= padlen + 1:
            raise ValueError(f'Ventana demasiado corta para el filtro: requiere más de {padlen + 1} muestras.')
        array = sosfiltfilt(sos, array, axis=0, padtype='odd', padlen=padlen)
    if not np.isfinite(array).all():
        raise ValueError('El preprocesamiento produjo NaN/inf.')
    return array
