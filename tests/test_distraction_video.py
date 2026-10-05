import importlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.triggering import TriggerDecision
from src.tutor import Tutor


class DistractionVideoTests(unittest.TestCase):
    def setUp(self):
        self.environment = {}

    def test_import_does_not_open_browser(self):
        with patch.dict(os.environ, self.environment, clear=True):
            with patch("webbrowser.open") as browser:
                module = importlib.import_module("src.tools.distraction_video")
                # Re-execute even if another test already imported the module.
                importlib.reload(module)
        browser.assert_not_called()

    def test_explicit_call_opens_fixed_url_once(self):
        from src.tools.distraction_video import open_distraction_video

        with patch.dict(os.environ, self.environment, clear=True):
            with patch("webbrowser.open", return_value=True) as browser:
                self.assertTrue(open_distraction_video())
        browser.assert_called_once_with(
            "https://www.tiktok.com/@ingenierossiningenio/video/7693213355208609044")

    def test_public_constant_and_environment_cannot_override_video(self):
        from src.tools.distraction_video import TIKTOK_VIDEO_URL, open_distraction_video

        expected = "https://www.tiktok.com/@ingenierossiningenio/video/7693213355208609044"
        self.assertEqual(TIKTOK_VIDEO_URL, expected)
        for environment in ({}, {"DISTRACTION_VIDEO_URL": ""},
                            {"DISTRACTION_VIDEO_URL": "https://example.com/other"}):
            with self.subTest(environment=environment), \
                 patch.dict(os.environ, environment, clear=True), \
                 patch("webbrowser.open", return_value=False) as browser:
                self.assertFalse(open_distraction_video())
                browser.assert_called_once_with(expected)

    def test_blocked_decisions_cause_no_api_alert_or_video(self):
        from src.tools.distraction_video import open_distraction_video

        client = Mock()
        alert = Mock()
        tutor = Tutor(model="test-model", client=client, visual_alert=alert)
        with patch.dict(os.environ, self.environment, clear=True):
            with patch("webbrowser.open") as browser:
                for triggered in (False, True):
                    for state in ("HIGH_LOAD", "ARTIFACT", "REST", "LOW_LOAD"):
                        if triggered and state == "HIGH_LOAD":
                            continue
                        with self.subTest(triggered=triggered, state=state):
                            decision = TriggerDecision(
                                triggered=triggered, reason="test", state=state,
                                confidence=0.9, consecutive_high_load_windows=0,
                                cooldown_remaining_seconds=0.0,
                            )
                            text = tutor.respond(decision)
                            if text:
                                open_distraction_video()
                            self.assertIsNone(text)
                browser.assert_not_called()
        client.responses.create.assert_not_called()
        alert.assert_not_called()

    def test_valid_trigger_does_not_open_video_until_application_calls_tool(self):
        from src.tools.distraction_video import open_distraction_video

        client = Mock()
        client.responses.create.return_value = SimpleNamespace(output_text="Ayuda breve.")
        tutor = Tutor(model="test-model", client=client)
        decision = TriggerDecision(
            triggered=True, reason="sustained_high_load", state="HIGH_LOAD",
            confidence=0.9, consecutive_high_load_windows=0,
            cooldown_remaining_seconds=30.0,
        )
        with patch.dict(os.environ, self.environment, clear=True):
            with patch("webbrowser.open", return_value=True) as browser:
                text = tutor.respond(decision)
                self.assertEqual(text, "Ayuda breve.")
                browser.assert_not_called()
                if text:
                    open_distraction_video()
                browser.assert_called_once_with(
                    "https://www.tiktok.com/@ingenierossiningenio/video/7693213355208609044")
        client.responses.create.assert_called_once()
        self.assertNotIn("tools", client.responses.create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
