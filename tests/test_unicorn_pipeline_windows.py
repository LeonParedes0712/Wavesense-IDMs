"""Integración por ventanas simuladas de 10 s; sin SDK ni servicios externos."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import run_unicorn_pipeline as runner
from src.eeg_config import EEGConfig, FEATURE_NAMES, QualityConfig
from src.models import EEGModelRuntime, ModelContractError
from src.pipeline import EEGPipeline
from src.triggering import TemporalTrigger, TriggerConfig
from src.unicorn_stream import ArraySource


class WindowRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = EEGConfig(window_seconds=10, hop_seconds=10)
        time = np.arange(2500) / 250
        self.window = np.sin(2 * np.pi * 10 * time[:, None]) * np.arange(1, 9)
        self.model = Mock()
        self.model.feature_names_in_ = np.array(FEATURE_NAMES)
        self.model.n_features_in_ = 32
        self.model.classes_ = np.array(['REST', 'MATH_LOAD'])
        self.model.predict_proba.return_value = np.array([[.1, .9]])
        self.runtime = EEGModelRuntime(self.model)
        self.trigger = TemporalTrigger(TriggerConfig(required_consecutive_windows=2), clock=lambda: 0)

    def pipeline(self, runtime=None):
        return EEGPipeline(runtime, config=self.config, quality=QualityConfig(), trigger=self.trigger)

    def run_windows(self, windows, pipeline):
        source = ArraySource(np.concatenate(windows), sample_rate=250)
        source.serial = 'simulated'
        source.frame_length = 25
        source.channel_diagnostics = tuple((f'EEG {i}', i - 1) for i in range(1, 9))
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            with self.assertRaises(EOFError):
                runner.run(source, pipeline)
        self.assertFalse(source.connected)
        return output.getvalue(), errors.getvalue()

    def test_valid_window_without_model_uses_existing_filter_and_features(self):
        pipeline = self.pipeline()
        with patch.object(pipeline, 'process_window', wraps=pipeline.process_window) as process, \
             patch.object(self.trigger, 'evaluate', wraps=self.trigger.evaluate) as evaluate:
            output, errors = self.run_windows([self.window], pipeline)
        process.assert_called_once()
        evaluate.assert_not_called()
        self.assertIn('Ventana #1 | shape=(2500, 8) | fs=250', output)
        self.assertIn('Calidad: válida', output)
        self.assertIn('Características EEG (32)', output)
        self.assertEqual(output.count('Theta='), 8)
        self.assertNotIn('Probabilidades', output)
        self.assertEqual(errors, '')
        result = pipeline.process_window(self.window, sample_rate=250)
        self.assertEqual(tuple(result.features.columns), FEATURE_NAMES)
        self.assertTrue(np.isfinite(result.features.to_numpy()).all())
        self.assertIsNone(result.model_probabilities)
        self.assertIsNone(result.decision)

    def test_valid_windows_with_runtime_adapter_and_temporal_trigger(self):
        output, errors = self.run_windows([self.window] * 2, self.pipeline(self.runtime))
        self.assertEqual(self.model.predict_proba.call_count, 2)
        features = self.model.predict_proba.call_args.args[0]
        self.assertEqual(features.shape, (1, 32))
        self.assertEqual(tuple(features.columns), FEATURE_NAMES)
        self.assertIn("Probabilidades originales: {'REST': 0.1, 'MATH_LOAD': 0.9}", output)
        self.assertIn("'HIGH_LOAD': 0.9", output)
        self.assertIn('triggered=True', output)
        self.assertEqual(errors, '')

    def test_artifact_blocks_model_and_trigger_activation_resets_streak(self):
        pipeline = self.pipeline(self.runtime)
        output, _ = self.run_windows([self.window, np.zeros_like(self.window), self.window], pipeline)
        self.assertEqual(self.model.predict_proba.call_count, 2)
        self.assertIn('Calidad: ARTIFACT', output)
        self.assertIn("state='ARTIFACT'", output)
        self.assertNotIn('triggered=True', output)
        self.assertEqual(output.count('Características EEG (32)'), 2)

    def test_runtime_contract_error_is_reported_and_acquisition_continues(self):
        self.model.predict_proba.side_effect = [ModelContractError('Se esperaban 32 variables'),
                                                np.array([[.1, .9]])]
        output, errors = self.run_windows([self.window] * 2, self.pipeline(self.runtime))
        self.assertIn('contrato de 32 columnas', errors)
        self.assertIn('Se esperaban 32 variables', errors)
        self.assertIn('Ventana #2', output)
        self.assertIn('Probabilidades originales:', output)

    def test_unexpected_processing_failure_closes_stream(self):
        source = ArraySource(self.window, sample_rate=250)
        source.serial = 'simulated'
        source.frame_length = 25
        source.channel_diagnostics = ()
        pipeline = self.pipeline()
        with patch.object(pipeline, 'process_window', side_effect=RuntimeError('processing failed')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'processing failed'):
                runner.run(source, pipeline)
        self.assertFalse(source.connected)

    def test_empty_or_missing_path_does_not_load_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ('', str(Path(directory) / 'absent.pkl')):
                with self.subTest(path=value), patch.dict(os.environ, {'EEG_MODEL_PATH': value}, clear=True), \
                     patch.object(EEGModelRuntime, 'from_env') as load, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    pipeline = runner.create_pipeline()
                load.assert_not_called()
                self.assertIsNone(pipeline.runtime)
                self.assertEqual(pipeline.config.window_seconds, 10)
                self.assertIn('Modelo no configurado: mostrando adquisición, calidad y características EEG.', output.getvalue())

    def test_existing_path_loads_runtime_once_without_inventing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            path.touch()
            with patch.dict(os.environ, {'EEG_MODEL_PATH': str(path)}, clear=True), \
                 patch.object(EEGModelRuntime, 'from_env', return_value=self.runtime) as load:
                pipeline = runner.create_pipeline()
            load.assert_called_once_with()
            self.assertIs(pipeline.runtime, self.runtime)
            self.assertIsNone(pipeline.tutor)

    def test_incompatible_schema_rejected_before_connecting(self):
        self.runtime.feature_names = FEATURE_NAMES[:-1]
        source = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            path.touch()
            with patch.dict(os.environ, {'EEG_MODEL_PATH': str(path)}, clear=True), \
                 patch.object(EEGModelRuntime, 'from_env', return_value=self.runtime):
                with self.assertRaisesRegex(ModelContractError, '32 columnas'):
                    runner.run(source)
        source.connect.assert_not_called()
