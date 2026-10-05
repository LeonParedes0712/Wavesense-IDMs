"""Generate tutor text only for decisions authorized by TemporalTrigger.

Importing or running this module does not create clients or execute tools.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from openai import OpenAI

    from src.triggering import TriggerDecision


def _create_client() -> OpenAI:
    """Read credentials only when an intervention needs the API."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Configura OPENAI_API_KEY para habilitar el tutor.")

    from openai import OpenAI

    return OpenAI(api_key=api_key, max_retries=0, timeout=30.0)


class Tutor:
    """One instance per session; pass each TemporalTrigger decision once.

    The caller chooses the model. An injected client permits offline tests;
    the caller owns that client's lifecycle. Visual alerts are opt-in.
    """

    def __init__(
        self,
        *,
        model: str,
        client: OpenAI | None = None,
        visual_alert: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self._visual_alert = visual_alert

    def respond(self, decision: TriggerDecision) -> str | None:
        """Return help text, or None without side effects for blocked windows.

        REST, LOW_LOAD and ARTIFACT are not intervenable in this policy,
        even if an inconsistent decision marks them as triggered.
        """
        from src.triggering import TriggerDecision

        if not isinstance(decision, TriggerDecision):
            raise TypeError("El tutor requiere un TriggerDecision de TemporalTrigger.")
        if decision.triggered is not True or decision.state != "HIGH_LOAD":
            return None
        if not 0.0 <= decision.confidence <= 1.0:
            raise ValueError("La confianza debe estar entre 0 y 1.")

        if self._client is None:
            with _create_client() as client:
                text = self._generate_text(client, decision)
        else:
            text = self._generate_text(self._client, decision)

        if self._visual_alert is not None:
            self._visual_alert(text)
        return text

    def _generate_text(self, client: OpenAI, decision: TriggerDecision) -> str:
        response = client.responses.create(
            model=self.model,
            instructions=(
                "Redacta en español una ayuda breve para abordar la tarea actual "
                "en pasos pequeños. No conoces el contenido de la tarea: no lo "
                "inventes. El estado y la confianza son estimaciones del modelo "
                "local, no un diagnóstico de distracción o frustración. "
                "Genera solo texto de apoyo; no decidas acciones ni herramientas."
            ),
            input=json.dumps({
                "state": decision.state,
                "confidence": decision.confidence,
            }),
            store=False,
        )
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("OpenAI no devolvió texto de ayuda.")
        return text
