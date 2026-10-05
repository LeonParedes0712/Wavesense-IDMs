"""
Temporal intervention decision utilities for Wavesense-IDMs.

This module decides whether a sustained local model prediction should
produce a tutor intervention. It is independent from EEG processing,
machine-learning models and the OpenAI API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Mapping

from dotenv import load_dotenv


def _read_float_from_env(name: str, default: float) -> float:
    """Read a float environment variable or return its default value."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, not {value!r}.") from error


def _read_int_from_env(name: str, default: int) -> int:
    """Read an integer environment variable or return its default value."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, not {value!r}.") from error


@dataclass(frozen=True)
class TriggerConfig:
    """Configurable rules for temporal intervention triggering."""

    high_load_threshold: float = 0.82
    required_consecutive_windows: int = 3
    artifact_threshold: float = 0.50
    cooldown_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "TriggerConfig":
        """Create configuration from local environment variables."""
        load_dotenv()
        defaults = cls()

        return cls(
            high_load_threshold=_read_float_from_env(
                "HIGH_LOAD_THRESHOLD",
                defaults.high_load_threshold,
            ),
            required_consecutive_windows=_read_int_from_env(
                "REQUIRED_CONSECUTIVE_WINDOWS",
                defaults.required_consecutive_windows,
            ),
            artifact_threshold=_read_float_from_env(
                "ARTIFACT_THRESHOLD",
                defaults.artifact_threshold,
            ),
            cooldown_seconds=_read_float_from_env(
                "COOLDOWN_SECONDS",
                defaults.cooldown_seconds,
            ),
        )


@dataclass(frozen=True)
class TriggerDecision:
    """Result of evaluating one prediction window."""

    triggered: bool
    reason: str
    state: str
    confidence: float
    consecutive_high_load_windows: int
    cooldown_remaining_seconds: float


class TemporalTrigger:
    """Tracks temporal state and decides when an intervention is valid."""

    def __init__(
        self,
        config: TriggerConfig | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config = config or TriggerConfig.from_env()
        self._clock = clock
        self._consecutive_high_load_windows = 0
        self._last_trigger_time: float | None = None

    def evaluate(self, probabilities: Mapping[str, float]) -> TriggerDecision:
        """
        Evaluate one local-model prediction window.

        Expected states are REST, LOW_LOAD, HIGH_LOAD and ARTIFACT.
        """
        state, confidence = max(probabilities.items(), key=lambda item: item[1])
        artifact_confidence = probabilities.get("ARTIFACT", 0.0)
        now = self._clock()

        cooldown_remaining = self._cooldown_remaining(now)
        if cooldown_remaining > 0:
            self._consecutive_high_load_windows = 0
            return self._decision(
                triggered=False,
                reason="cooldown_active",
                state=state,
                confidence=confidence,
                cooldown_remaining=cooldown_remaining,
            )

        if state == "ARTIFACT" or artifact_confidence >= self.config.artifact_threshold:
            self._consecutive_high_load_windows = 0
            return self._decision(
                triggered=False,
                reason="artifact_detected",
                state=state,
                confidence=confidence,
            )

        high_load_confidence = probabilities.get("HIGH_LOAD", 0.0)
        if high_load_confidence < self.config.high_load_threshold:
            self._consecutive_high_load_windows = 0
            return self._decision(
                triggered=False,
                reason="high_load_below_threshold",
                state=state,
                confidence=confidence,
            )

        self._consecutive_high_load_windows += 1

        if self._consecutive_high_load_windows < self.config.required_consecutive_windows:
            return self._decision(
                triggered=False,
                reason="waiting_for_persistence",
                state=state,
                confidence=confidence,
            )

        self._last_trigger_time = now
        self._consecutive_high_load_windows = 0
        return self._decision(
            triggered=True,
            reason="sustained_high_load",
            state=state,
            confidence=confidence,
            cooldown_remaining=self.config.cooldown_seconds,
        )

    def _cooldown_remaining(self, now: float) -> float:
        """Return remaining cooldown time, or zero when it has expired."""
        if self._last_trigger_time is None:
            return 0.0

        elapsed = now - self._last_trigger_time
        return max(0.0, self.config.cooldown_seconds - elapsed)

    def _decision(
        self,
        *,
        triggered: bool,
        reason: str,
        state: str,
        confidence: float,
        cooldown_remaining: float = 0.0,
    ) -> TriggerDecision:
        return TriggerDecision(
            triggered=triggered,
            reason=reason,
            state=state,
            confidence=confidence,
            consecutive_high_load_windows=self._consecutive_high_load_windows,
            cooldown_remaining_seconds=round(cooldown_remaining, 2),
        )
