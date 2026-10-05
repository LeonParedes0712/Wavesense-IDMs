import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.model_adapter import to_wavesense_probabilities
from src.tools.visual_alert import show_visual_alert
from src.triggering import TemporalTrigger, TriggerConfig, TriggerDecision
from src.tutor import Tutor


def decision(*, triggered=True, state="HIGH_LOAD", confidence=0.9):
    return TriggerDecision(
        triggered=triggered,
        reason="sustained_high_load" if triggered else "waiting_for_persistence",
        state=state,
        confidence=confidence,
        consecutive_high_load_windows=0,
        cooldown_remaining_seconds=30.0,
    )


class TutorTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(
            output_text=" Vamos paso a paso. "
        )
        self.alert = Mock()
        self.tutor = Tutor(
            model="test-model", client=self.client, visual_alert=self.alert
        )

    def test_blocked_decisions_never_call_api_or_tools(self):
        for state in ("REST", "LOW_LOAD", "HIGH_LOAD", "ARTIFACT"):
            with self.subTest(state=state):
                self.assertIsNone(self.tutor.respond(decision(triggered=False, state=state)))
        self.client.responses.create.assert_not_called()
        self.alert.assert_not_called()

    def test_only_high_load_is_intervenable_even_if_triggered(self):
        for state in ("REST", "LOW_LOAD", "ARTIFACT", "focused", "distracted"):
            with self.subTest(state=state):
                self.assertIsNone(self.tutor.respond(decision(state=state)))
        self.client.responses.create.assert_not_called()
        self.alert.assert_not_called()

    def test_requires_trigger_decision_not_raw_model_output(self):
        with self.assertRaises(TypeError):
            self.tutor.respond({"HIGH_LOAD": 0.9})
        self.client.responses.create.assert_not_called()
        self.alert.assert_not_called()

    def test_invalid_confidence_does_not_reach_api(self):
        for confidence in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                self.tutor.respond(decision(confidence=confidence))
        self.client.responses.create.assert_not_called()
        self.alert.assert_not_called()

    def test_trigger_sends_only_state_and_confidence_and_returns_text(self):
        text = self.tutor.respond(decision())

        self.assertEqual(text, "Vamos paso a paso.")
        self.client.responses.create.assert_called_once()
        payload = self.client.responses.create.call_args.kwargs
        self.assertEqual(set(payload), {"model", "instructions", "input", "store"})
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(json.loads(payload["input"]), {
            "state": "HIGH_LOAD", "confidence": 0.9,
        })
        self.assertFalse(payload["store"])
        self.alert.assert_called_once_with(text)

    def test_visual_alert_is_disabled_by_default(self):
        with patch("src.tools.visual_alert.show_visual_alert") as alert:
            tutor = Tutor(model="test-model", client=self.client)
            self.assertEqual(tutor.respond(decision()), "Vamos paso a paso.")
        alert.assert_not_called()

    def test_blocked_decision_does_not_initialize_client_or_require_key(self):
        with patch.dict(os.environ, {}, clear=True), patch("src.tutor._create_client") as factory:
            tutor = Tutor(model="test-model", visual_alert=self.alert)
            self.assertIsNone(tutor.respond(decision(triggered=False)))
            self.assertIsNone(tutor.respond(decision(state="ARTIFACT")))
        factory.assert_not_called()
        self.alert.assert_not_called()

    def test_missing_or_blank_key_has_clear_error_without_client_creation(self):
        sdk = SimpleNamespace(OpenAI=MagicMock())
        for environment in ({}, {"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": "  "}):
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, {"openai": sdk}):
                    tutor = Tutor(model="test-model", visual_alert=self.alert)
                    with self.assertRaisesRegex(RuntimeError, "Configura OPENAI_API_KEY"):
                        tutor.respond(decision())
        sdk.OpenAI.assert_not_called()
        self.alert.assert_not_called()

    def test_production_client_uses_environment_and_closes_after_request(self):
        sdk = SimpleNamespace(OpenAI=MagicMock())
        sdk.OpenAI.return_value.__enter__.return_value = self.client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-placeholder"}, clear=True):
            with patch.dict(sys.modules, {"openai": sdk}):
                tutor = Tutor(model="test-model")
                sdk.OpenAI.assert_not_called()
                self.assertEqual(tutor.respond(decision()), "Vamos paso a paso.")
        sdk.OpenAI.assert_called_once_with(
            api_key="unit-test-placeholder", max_retries=0, timeout=30.0
        )
        sdk.OpenAI.return_value.__exit__.assert_called_once()

    def test_api_error_does_not_execute_alert_or_retry(self):
        self.client.responses.create.side_effect = RuntimeError("API unavailable")
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            self.tutor.respond(decision())
        self.client.responses.create.assert_called_once()
        self.alert.assert_not_called()

    def test_empty_response_does_not_execute_alert(self):
        self.client.responses.create.return_value.output_text = "  "
        with self.assertRaisesRegex(RuntimeError, "no devolvió texto"):
            self.tutor.respond(decision())
        self.alert.assert_not_called()

    def test_real_adapter_and_trigger_control_calls_across_windows(self):
        clock = Mock(return_value=0.0)
        trigger = TemporalTrigger(config=TriggerConfig(), clock=clock)
        high = {"REST": 0.05, "DIBUJANDO": 0.05, "MATH_LOAD": 0.9}
        below_threshold = {"REST": 0.15, "DIBUJANDO": 0.15, "MATH_LOAD": 0.7}

        def process(model_output, *, artifact=False):
            probabilities = to_wavesense_probabilities(model_output, artifact)
            evaluated = trigger.evaluate(probabilities)
            result = self.tutor.respond(evaluated)
            self.assertEqual(result is not None, evaluated.triggered)
            return evaluated.reason

        self.assertEqual(process(high), "waiting_for_persistence")
        self.assertEqual(process(below_threshold), "high_load_below_threshold")
        self.assertEqual(process(high), "waiting_for_persistence")
        self.assertEqual(process(high, artifact=True), "artifact_detected")
        for _ in range(2):
            self.assertEqual(process(high), "waiting_for_persistence")
        self.client.responses.create.assert_not_called()
        self.alert.assert_not_called()

        self.assertEqual(process(high), "sustained_high_load")
        self.client.responses.create.assert_called_once()
        self.alert.assert_called_once()
        for _ in range(5):
            self.assertEqual(process(high), "cooldown_active")
        self.client.responses.create.assert_called_once()
        self.alert.assert_called_once()

        clock.return_value = 30.0
        for _ in range(2):
            self.assertEqual(process(high), "waiting_for_persistence")
        self.assertEqual(process(high), "sustained_high_load")
        self.assertEqual(self.client.responses.create.call_count, 2)
        self.assertEqual(self.alert.call_count, 2)

    def test_import_and_direct_execution_have_no_external_side_effects(self):
        sdk = SimpleNamespace(OpenAI=MagicMock())
        tkinter = SimpleNamespace(messagebox=Mock())
        path = Path(__file__).resolve().parents[1] / "src" / "tutor.py"
        with patch.dict(sys.modules, {"openai": sdk, "tkinter": tkinter}):
            with patch("webbrowser.open") as browser:
                runpy.run_path(str(path), run_name="isolated_tutor")
                runpy.run_path(str(path), run_name="__main__")
        sdk.OpenAI.assert_not_called()
        tkinter.messagebox.showinfo.assert_not_called()
        browser.assert_not_called()

    def test_visual_tool_displays_text_only_when_explicitly_called(self):
        tkinter = SimpleNamespace(messagebox=Mock())
        with patch.dict(sys.modules, {"tkinter": tkinter}):
            show_visual_alert("Ayuda de prueba")
        tkinter.messagebox.showinfo.assert_called_once_with(
            title="Wavesense — ayuda", message="Ayuda de prueba"
        )


if __name__ == "__main__":
    unittest.main()
