"""Configuración explícita del prototipo; defaults provisionales, no entrenamiento."""

from dataclasses import dataclass
import math
import os


BAND_NAMES = ('Theta', 'Alpha', 'Beta', 'Gamma')
FEATURE_NAMES = tuple(f'Canal_{c}_{b}' for c in range(1, 9) for b in BAND_NAMES)


def _number(name, default, cast=float):
    try:
        return cast(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name}: valor numérico inválido.') from exc


def _boolean(name, default):
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in ('true', 'false'):
        raise ValueError(f'{name}: usa true o false.')
    return value == 'true'


def _positive(value, name, *, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name}: requiere un valor finito {"no negativo" if zero else "positivo"}.')


@dataclass(frozen=True)
class EEGConfig:
    # Evidencia: notebook celdas 0/1; ver src/README.md para diferencias.
    sample_rate: float = 250.0
    window_seconds: float = 1.0
    hop_seconds: float = 1.0
    channel_count: int = 8
    bands: tuple = ((4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 45.0))
    filter_enabled: bool = True
    filter_low: float = 1.0
    filter_high: float = 40.0
    filter_order: int = 4
    welch_nperseg: int = 256
    welch_overlap: float = 0.5
    reference: str = 'as_acquired'
    input_units: str = 'unconfirmed'

    def __post_init__(self):
        for name in ('sample_rate', 'window_seconds', 'hop_seconds'):
            _positive(getattr(self, name), name)
        if type(self.channel_count) is not int or self.channel_count != 8:
            raise ValueError('EEG_CHANNEL_COUNT debe ser 8 para el contrato de 32 características.')
        if self.hop_seconds > self.window_seconds:
            raise ValueError('EEG_HOP_SECONDS debe ser <= EEG_WINDOW_SECONDS (sin huecos).')
        for name in ('window_seconds', 'hop_seconds'):
            samples = getattr(self, name) * self.sample_rate
            if not math.isclose(samples, round(samples), rel_tol=0, abs_tol=1e-8) or samples < 1:
                raise ValueError(f'{name} * EEG_SAMPLE_RATE debe dar un entero positivo de muestras.')
        for name in ('filter_order', 'welch_nperseg'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} debe ser un entero positivo.')
        if not isinstance(self.filter_enabled, bool):
            raise ValueError('filter_enabled debe ser booleano.')
        if not 0 <= self.welch_overlap < 1:
            raise ValueError('EEG_WELCH_OVERLAP debe estar en [0, 1).')
        if self.window_samples < 2 or self.welch_nperseg < 2:
            raise ValueError('Welch requiere al menos dos muestras.')
        if len(self.bands) != 4:
            raise ValueError('Se requieren cuatro rangos: Theta, Alpha, Beta, Gamma.')
        for low, high in self.bands:
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high < self.sample_rate / 2):
                raise ValueError('Bandas: requiere 0 < mínimo < máximo < Nyquist.')
        if not (0 < self.filter_low < self.filter_high < self.sample_rate / 2):
            raise ValueError('Filtro: requiere 0 < mínimo < máximo < Nyquist.')
        if self.reference not in ('as_acquired', 'common_average'):
            raise ValueError('EEG_REFERENCE debe ser as_acquired o common_average.')
        if self.input_units not in ('unconfirmed', 'uV', 'V'):
            raise ValueError('EEG_INPUT_UNITS debe ser unconfirmed, uV o V; no hay conversión automática.')

    @property
    def window_samples(self):
        return round(self.window_seconds * self.sample_rate)

    @property
    def hop_samples(self):
        return round(self.hop_seconds * self.sample_rate)

    @classmethod
    def from_env(cls):
        defaults = cls()
        bands = []
        for name, default in zip(BAND_NAMES, defaults.bands):
            variable = f'EEG_{name.upper()}_HZ'
            try:
                bounds = tuple(float(x) for x in os.getenv(variable, ','.join(map(str, default))).split(','))
                if len(bounds) != 2:
                    raise ValueError()
                bands.append(bounds)
            except ValueError as exc:
                raise ValueError(f'{variable}: usa mínimo,máximo en Hz.') from exc
        return cls(
            sample_rate=_number('EEG_SAMPLE_RATE', defaults.sample_rate),
            window_seconds=_number('EEG_WINDOW_SECONDS', defaults.window_seconds),
            hop_seconds=_number('EEG_HOP_SECONDS', defaults.hop_seconds),
            channel_count=_number('EEG_CHANNEL_COUNT', defaults.channel_count, int),
            bands=tuple(bands),
            filter_enabled=_boolean('EEG_FILTER_ENABLED', defaults.filter_enabled),
            filter_low=_number('EEG_FILTER_LOW_HZ', defaults.filter_low),
            filter_high=_number('EEG_FILTER_HIGH_HZ', defaults.filter_high),
            filter_order=_number('EEG_FILTER_ORDER', defaults.filter_order, int),
            welch_nperseg=_number('EEG_WELCH_NPERSEG', defaults.welch_nperseg, int),
            welch_overlap=_number('EEG_WELCH_OVERLAP', defaults.welch_overlap),
            reference=os.getenv('EEG_REFERENCE', defaults.reference),
            input_units=os.getenv('EEG_INPUT_UNITS', defaults.input_units),
        )


@dataclass(frozen=True)
class QualityConfig:
    # Un canal totalmente constante invalida la ventana. Otros límites necesitan
    # calibración y están desactivados (None), en las unidades de entrada.
    flatline_ptp: float = 0.0
    max_abs: float | None = None
    max_step: float | None = None

    def __post_init__(self):
        _positive(self.flatline_ptp, 'EEG_FLATLINE_PTP', zero=True)
        for name in ('max_abs', 'max_step'):
            if getattr(self, name) is not None:
                _positive(getattr(self, name), name)

    @classmethod
    def from_env(cls):
        def optional(name):
            return _number(name, '') if os.getenv(name, '').strip() else None
        return cls(
            flatline_ptp=_number('EEG_FLATLINE_PTP', 0.0),
            max_abs=optional('EEG_MAX_ABS'),
            max_step=optional('EEG_MAX_STEP'),
        )
