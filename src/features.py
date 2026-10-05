"""32 sumas de bins PSD; orden confirmado, receta de entrenamiento pendiente."""

import numpy as np
import pandas as pd
from scipy.signal import welch, windows

from src.eeg_config import BAND_NAMES, FEATURE_NAMES, EEGConfig
from src.preprocessing import validate_window


def extract_band_powers(window, *, sample_rate, config: EEGConfig):
    """Una ventana ya preprocesada -> DataFrame (1,32), antes del escalador.

    Receta provisional del notebook: límites inclusivos y suma de bins, NO
    integración de PSD. Los bins fronterizos pueden pertenecer a dos bandas.
    No filtra, no normaliza, no calcula ratios ni altera probabilidades.
    """
    array = validate_window(window, sample_rate=sample_rate, config=config)
    segment = min(config.welch_nperseg, len(array))
    frequencies, psd = welch(
        array, fs=config.sample_rate, axis=0,
        window=windows.hann(segment, sym=False), nperseg=segment,
        noverlap=int(segment * config.welch_overlap), nfft=segment,
        detrend='constant', return_onesided=True, scaling='density', average='mean',
    )
    powers = []
    for name, (low, high) in zip(BAND_NAMES, config.bands):
        mask = (frequencies >= low) & (frequencies <= high)
        if not mask.any():
            raise ValueError(f'Banda {name} sin bins PSD: revisa ventana, frecuencia y nperseg.')
        powers.append(psd[mask].sum(axis=0))
    # (bandas, canales) -> (canales, bandas) -> orden exacto del modelo.
    values = np.stack(powers, axis=1).reshape(1, 32)
    if not np.isfinite(values).all():
        raise ValueError('Extracción de características produjo NaN/inf.')
    return pd.DataFrame(values, columns=FEATURE_NAMES)
