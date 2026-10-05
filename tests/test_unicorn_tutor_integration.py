"""Flujo real del ejecutable con EEG simulado, API y navegador sustituidos."""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from scripts import check_openai_tutor, run_unicorn_pipeline as runner
from src.eeg_config import FEATURE_NAMES
from src.models import EEGModelRuntime, ModelContractError
from src.pipeline import PipelineResult
from src.preprocessing import SignalQuality
from src.triggering import TemporalTrigger, TriggerConfig, TriggerDecision
from src.unicorn_stream import ArraySource


class TutorRunnerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'ENABLE_TUTOR': 'true', 'OPENAI_MODEL': 'offline-model',
            'OPEN_TIKTOK_ON_TRIGGER': 'true',
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(output_text=' Un paso pequeño. ')
        factory = MagicMock()
        factory.return_value.__enter__.return_value = self.client
        self.factory = self.enterContext(patch('src.tutor._create_client', factory))
        self.browser = self.enterContext(patch('webbrowser.open', return_value=True))
        self.enterContext(patch('dotenv.load_dotenv', return_value=False))
        self.model = Mock(feature_names_in_=np.array(FEATURE_NAMES), n_features_in_=32,
                          classes_=np.array(['REST', 'DIBUJANDO', 'MATH_LOAD']))
        self.model.predict_proba.return_value = np.array([[.05, .05, .9]])
        self.runtime = EEGModelRuntime(self.model)
        time = np.arange(2500) / 250
        self.window = np.sin(2 * np.pi * 10 * time[:, None]) * np.arange(1, 9)

    def pipeline(self):
        with patch.object(Path, 'is_file', return_value=True), \
             patch.object(EEGModelRuntime, 'from_env', return_value=self.runtime), \
             patch.dict(os.environ, {'EEG_MODEL_PATH': 'fake.pkl'}):
            pipeline = runner.create_pipeline()
        pipeline.trigger = TemporalTrigger(TriggerConfig(), clock=lambda: 0)
        return pipeline

    def run_windows(self, pipeline, windows=None):
        windows = [self.window] * 5 if windows is None else windows
        source = ArraySource(np.concatenate(windows), sample_rate=250)
        source.serial, source.frame_length, source.channel_diagnostics = 'test', 25, ()
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            with self.assertRaises(EOFError):
                runner.run(source, pipeline)
        self.assertFalse(source.connected)
        return output.getvalue(), errors.getvalue()

    def test_disabled_tutor_never_creates_client_or_opens_browser(self):
        for value in ('false', '', '1'):
            with self.subTest(value=value), patch.dict(os.environ, {'ENABLE_TUTOR': value}):
                pipeline = self.pipeline()
                self.assertIsNone(pipeline.tutor)
                self.run_windows(pipeline)
        self.factory.assert_not_called()
        self.browser.assert_not_called()

    def test_persistent_high_load_prints_text_and_opens_once_during_cooldown(self):
        pipeline = self.pipeline()
        self.assertEqual(pipeline.tutor.model, 'offline-model')
        self.assertIsNone(pipeline.tutor._visual_alert)
        output, errors = self.run_windows(pipeline)
        self.assertIn('Tutor: Un paso pequeño.', output)
        self.assertIn('cooldown_active', output)
        self.assertIn('Ventana #5', output)
        self.assertEqual(errors, '')
        self.client.responses.create.assert_called_once()
        self.browser.assert_called_once_with(
            'https://www.tiktok.com/@ingenierossiningenio/video/7693213355208609044')

    def test_video_requires_explicit_flag(self):
        for flag in ('false', '', '1'):
            with self.subTest(flag=flag), patch.dict(os.environ, {
                    'OPEN_TIKTOK_ON_TRIGGER': flag}):
                output, _ = self.run_windows(self.pipeline())
                self.assertIn('Tutor:', output)
        self.browser.assert_not_called()

    def test_rest_low_load_and_artifact_never_call_api_or_browser(self):
        for probabilities, window in (([[.9, .05, .05]], self.window),
                                      ([[.05, .9, .05]], self.window),
                                      ([[.05, .05, .9]], np.zeros_like(self.window))):
            with self.subTest(probabilities=probabilities):
                self.model.predict_proba.return_value = np.array(probabilities)
                self.run_windows(self.pipeline(), [window] * 5)
        self.factory.assert_not_called()
        self.browser.assert_not_called()

    def test_api_failure_or_empty_response_continues_without_video_or_retry(self):
        for fail in (True, False):
            with self.subTest(fail=fail):
                self.client.responses.create.reset_mock()
                self.client.responses.create.side_effect = RuntimeError('API unavailable') if fail else None
                self.client.responses.create.return_value.output_text = '  '
                output, errors = self.run_windows(self.pipeline())
                self.assertIn('Error del tutor/OpenAI:', errors)
                self.assertIn('Ventana #5', output)
                self.assertIn('cooldown_active', output)
                self.client.responses.create.assert_called_once()
        self.browser.assert_not_called()

    def test_browser_exception_or_false_result_continues_acquisition(self):
        for error in (RuntimeError('browser unavailable'), None):
            with self.subTest(error=error):
                self.browser.reset_mock()
                self.browser.side_effect = error
                self.browser.return_value = False
                output, errors = self.run_windows(self.pipeline())
                self.assertIn('Error de navegador:', errors)
                self.assertIn('Ventana #5', output)
                self.browser.assert_called_once()

    def test_model_error_never_calls_api_or_browser(self):
        self.model.predict_proba.side_effect = ModelContractError('bad model')
        output, errors = self.run_windows(self.pipeline())
        self.assertIn('Error de modelo', errors)
        self.assertIn('Ventana #5', output)
        self.factory.assert_not_called()
        self.browser.assert_not_called()

    def test_video_rejects_inconsistent_decisions_and_blank_text(self):
        for state, triggered, text in (
                ('REST', True, 'text'), ('LOW_LOAD', True, 'text'),
                ('ARTIFACT', True, 'text'), ('HIGH_LOAD', False, 'text'),
                ('HIGH_LOAD', True, None), ('HIGH_LOAD', True, ''),
                ('HIGH_LOAD', True, '  ')):
            with self.subTest(state=state, triggered=triggered, text=text):
                decision = TriggerDecision(triggered, 'test', state, .9, 0, 30)
                runner.maybe_open_video(PipelineResult(None, None, None,
                    SignalQuality(False, ()), decision, text))
        self.browser.assert_not_called()


class ManualCheckTests(unittest.TestCase):
    def test_manual_check_loads_env_requires_both_variables_and_never_opens_ui(self):
        for missing in ('OPENAI_API_KEY', 'OPENAI_MODEL'):
            environment = {'OPENAI_API_KEY': 'test-placeholder', 'OPENAI_MODEL': 'test-model'}
            environment[missing] = ' '
            with self.subTest(missing=missing), patch.dict(os.environ, environment, clear=True), \
                 patch('dotenv.load_dotenv') as load, patch('src.tutor._create_client') as factory, \
                 contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(check_openai_tutor.main(), 1)
                self.assertIn(missing, errors.getvalue())
                load.assert_called_once_with(Path(check_openai_tutor.__file__).resolve().parents[1] / '.env')
                factory.assert_not_called()

    def test_manual_check_with_mock_api_only(self):
        client = Mock()
        client.responses.create.return_value.output_text = 'Respuesta de prueba.'
        for error, status in ((None, 0), (RuntimeError('API unavailable'), 1)):
            client.responses.create.side_effect = error
            with self.subTest(status=status), patch.dict(os.environ, {
                    'OPENAI_API_KEY': 'test-placeholder', 'OPENAI_MODEL': 'test-model'}, clear=True), \
                 patch('dotenv.load_dotenv'), patch('src.tutor._create_client') as factory, \
                 patch('webbrowser.open') as browser, \
                 patch('src.tools.visual_alert.show_visual_alert') as alert, \
                 contextlib.redirect_stdout(io.StringIO()) as output, \
                 contextlib.redirect_stderr(io.StringIO()) as errors:
                factory.return_value.__enter__.return_value = client
                self.assertEqual(check_openai_tutor.main(), status)
                self.assertIn('llamada REAL', output.getvalue())
                if status == 0:
                    self.assertIn('Respuesta de prueba.', output.getvalue())
                else:
                    self.assertIn('Error del tutor/OpenAI', errors.getvalue())
                browser.assert_not_called()
                alert.assert_not_called()
