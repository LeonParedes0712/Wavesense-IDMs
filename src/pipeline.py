"""Procesa una ventana; el llamador controla adquisición, sesión y tutor opcional."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import TYPE_CHECKING

import pandas as pd

from src.eeg_config import EEGConfig, FEATURE_NAMES, QualityConfig
from src.features import extract_band_powers
from src.model_adapter import to_wavesense_probabilities
from src.models import EEGModelRuntime, ModelContractError
from src.preprocessing import SignalQuality, assess_quality, filter_window, validate_window
from src.triggering import TemporalTrigger, TriggerDecision

if TYPE_CHECKING:
    from src.tutor import Tutor


@dataclass(frozen=True)
class PipelineResult:
    features: pd.DataFrame | None
    model_probabilities: dict[str, float] | None
    wavesense_probabilities: dict[str, float] | None
    quality: SignalQuality
    decision: TriggerDecision | None
    tutor_text: str | None = None


class EEGPipeline:
    def __init__(self, runtime: EEGModelRuntime | None, *, config: EEGConfig,
                 quality: QualityConfig, trigger: TemporalTrigger, tutor: Tutor | None = None):
        if runtime is not None and tuple(runtime.feature_names) != FEATURE_NAMES:
            raise ModelContractError('El extractor requiere 32 columnas en orden Canal_1..8 × Theta/Alpha/Beta/Gamma.')
        self.runtime = runtime
        self.config = config
        self.quality_config = quality
        self.trigger = trigger
        self.tutor = tutor

    def _artifact(self, reasons):
        probabilities = to_wavesense_probabilities({}, signal_has_artifact=True)
        # Evaluar ARTIFACT es necesario para romper la racha de ventanas anteriores.
        decision = self.trigger.evaluate(probabilities)
        return PipelineResult(None, None, probabilities, SignalQuality(True, reasons), decision)

    def process_window(self, window, *, sample_rate, signal_has_artifact=False):
        """Una ventana -> resultados observables. Sin loops ni herramientas UI.

        runtime=None: devuelve calidad y características sin probabilidades ni decisión.
        Señal inválida/mala: ARTIFACT sin inferencia. Fallos de extracción/modelo
        se propagan tras romper la racha; no se fabrican probabilidades originales.
        """
        try:
            array = validate_window(window, sample_rate=sample_rate, config=self.config)
        except (ValueError, TypeError) as exc:
            return self._artifact((f'invalid_window: {exc}',))
        quality = assess_quality(array, sample_rate=sample_rate, config=self.config,
                                 quality=self.quality_config)
        reasons = quality.reasons + (('source_artifact',) if signal_has_artifact else ())
        if reasons:
            return self._artifact(reasons)
        try:
            filtered = filter_window(array, sample_rate=sample_rate, config=self.config)
            features = extract_band_powers(filtered, sample_rate=sample_rate, config=self.config)
            if self.runtime is None:
                return PipelineResult(features, None, None, quality, None)
            model_output = self.runtime.predict_proba(features)
            probabilities = to_wavesense_probabilities(model_output)
        except Exception:
            self._artifact(('processing_failed',))
            raise
        decision = self.trigger.evaluate(probabilities)
        text = None
        if (self.tutor is not None and isinstance(decision, TriggerDecision)
                and decision.triggered is True and decision.state == 'HIGH_LOAD'
                and 0 <= decision.confidence <= 1):
            try:
                text = self.tutor.respond(decision)
            except Exception as exc:
                print(f'Error del tutor/OpenAI: {exc}; continúa la adquisición.',
                      file=sys.stderr, flush=True)
        return PipelineResult(features, model_output, probabilities, quality, decision, text)
