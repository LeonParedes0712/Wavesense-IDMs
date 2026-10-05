from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.eeg_config import EEGConfig, FEATURE_NAMES, QualityConfig
from src.models import EEGModelRuntime, ModelContractError
from src.pipeline import EEGPipeline
from src.triggering import TemporalTrigger, TriggerConfig
from src.tutor import Tutor
from src.unicorn_stream import ArraySource, WindowStream


class FakeEEGModel:
    feature_names_in_ = np.array(FEATURE_NAMES)
    n_features_in_ = 32
    classes_ = np.array(['MATH_LOAD', 'REST', 'DIBUJANDO'])

    def predict_proba(self, values):
        self.seen = values.copy()
        return np.array([[.9, .05, .05]])


class FakeEEGScaler:
    feature_names_in_ = np.array(FEATURE_NAMES)
    n_features_in_ = 32

    def transform(self, values):
        return np.asarray(values) * 2


class EEGPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = EEGConfig()
        time = np.arange(250) / 250
        self.window = np.sin(2 * np.pi * 10 * time[:, None]) * np.arange(1, 9)
        self.runtime = EEGModelRuntime(FakeEEGModel(), FakeEEGScaler())
        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(output_text='Paso pequeño.')
        self.tutor = Tutor(model='offline-test', client=self.client)
        self.trigger = TemporalTrigger(config=TriggerConfig(), clock=lambda: 0)
        self.pipeline = EEGPipeline(self.runtime, config=self.config, quality=QualityConfig(),
                                    trigger=self.trigger, tutor=self.tutor)

    def test_complete_simulated_flow_no_hardware_network_or_browser(self):
        source = ArraySource(np.tile(self.window, (3, 1)), sample_rate=250)
        with patch('src.tutor._create_client', side_effect=AssertionError('API forbidden')), \
             patch('webbrowser.open', side_effect=AssertionError('browser forbidden')), \
             patch('src.unicorn_stream.importlib.import_module', side_effect=AssertionError('SDK forbidden')), \
             WindowStream(source, config=self.config, chunk_samples=17) as stream:
            results = []
            for _ in range(3):
                window = stream.next_window()
                results.append(self.pipeline.process_window(window.samples, sample_rate=window.sample_rate,
                                                            signal_has_artifact=window.has_artifact))
        self.assertFalse(source.connected)
        self.assertEqual([r.decision.triggered for r in results], [False, False, True])
        self.assertEqual(results[-1].tutor_text, 'Paso pequeño.')
        self.assertEqual(tuple(results[-1].features.columns), FEATURE_NAMES)
        np.testing.assert_allclose(self.runtime.model.seen, results[-1].features * 2)
        self.assertEqual(results[-1].model_probabilities['MATH_LOAD'], .9)
        self.assertEqual(results[-1].wavesense_probabilities['HIGH_LOAD'], .9)
        self.client.responses.create.assert_called_once()

    def test_artifact_blocks_inference_and_tutor_and_resets_persistence(self):
        self.pipeline.process_window(self.window, sample_rate=250)
        with patch.object(self.runtime, 'predict_proba') as predict, patch.object(self.tutor, 'respond') as respond:
            for window, source_bad in ((self.window, True), (np.zeros((250, 8)), False),
                                       (np.full((250, 8), np.nan), False), (np.zeros((3, 2)), False)):
                result = self.pipeline.process_window(window, sample_rate=250, signal_has_artifact=source_bad)
                self.assertEqual(result.wavesense_probabilities['ARTIFACT'], 1)
                self.assertFalse(result.decision.triggered)
                self.assertTrue(result.quality.has_artifact)
                self.assertIsNone(result.features)
                self.assertIsNone(result.model_probabilities)
                self.assertIsNone(result.tutor_text)
            predict.assert_not_called()
            respond.assert_not_called()
        result = self.pipeline.process_window(self.window, sample_rate=250)
        self.assertEqual(result.decision.consecutive_high_load_windows, 1)

    def test_quality_limits_are_applied_before_filter(self):
        self.pipeline.quality_config = QualityConfig(max_abs=10)
        result = self.pipeline.process_window(self.window + 100, sample_rate=250)
        self.assertIn('amplitude_limit', result.quality.reasons)
        self.assertFalse(hasattr(self.runtime.model, 'seen'))

    def test_tutor_optional_and_no_call_for_pending_decision(self):
        with patch.object(self.tutor, 'respond') as respond:
            self.pipeline.process_window(self.window, sample_rate=250)
            respond.assert_not_called()
        self.pipeline.tutor = None
        for _ in range(2):
            result = self.pipeline.process_window(self.window, sample_rate=250)
        self.assertTrue(result.decision.triggered)
        self.assertIsNone(result.tutor_text)
        self.client.responses.create.assert_not_called()

    def test_mismatched_runtime_is_rejected_before_processing(self):
        self.runtime.feature_names = tuple(reversed(FEATURE_NAMES))
        with self.assertRaises(ModelContractError):
            EEGPipeline(self.runtime, config=self.config, quality=QualityConfig(), trigger=self.trigger)

    def test_processing_error_propagates_and_breaks_streak(self):
        self.pipeline.process_window(self.window, sample_rate=250)
        with patch.object(self.runtime, 'predict_proba', side_effect=ModelContractError('broken model')):
            with self.assertRaisesRegex(ModelContractError, 'broken model'):
                self.pipeline.process_window(self.window, sample_rate=250)
        result = self.pipeline.process_window(self.window, sample_rate=250)
        self.assertEqual(result.decision.consecutive_high_load_windows, 1)
        self.client.responses.create.assert_not_called()

    def test_invalid_trigger_decision_does_not_reach_tutor(self):
        with patch.object(self.trigger, 'evaluate', return_value=SimpleNamespace(triggered=True)), \
             patch.object(self.tutor, 'respond') as respond:
            self.pipeline.process_window(self.window, sample_rate=250)
            respond.assert_not_called()
