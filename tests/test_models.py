"""Contrato del runtime con artefactos falsos, sin datos ni hardware."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.models import EEGModelRuntime, ModelContractError
from src.model_adapter import to_wavesense_probabilities


class FakeModel:
    n_features_in_ = 2
    feature_names_in_ = np.array(['theta', 'beta'])
    classes_ = np.array(['MATH_LOAD', 'REST', 'DIBUJANDO'])

    def predict_proba(self, features):
        self.seen = features.copy()
        # La salida depende de la entrada: detecta errores de orden o escalado.
        high = np.asarray(features)[:, 0] / 10
        return np.column_stack([high, np.full(len(features), .2), .8 - high])


class FakeScaler:
    n_features_in_ = 2
    feature_names_in_ = np.array(['theta', 'beta'])

    def transform(self, features):
        self.seen = features.copy()
        return np.asarray(features) * [2, 3]


class TestEEGModelRuntime(unittest.TestCase):
    names = ('theta', 'beta')

    def test_load_paths_and_scale_before_predicting(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path, scaler_path = Path(tmp) / 'model.pkl', Path(tmp) / 'scaler.pkl'
            joblib.dump(FakeModel(), model_path)
            joblib.dump(FakeScaler(), scaler_path)
            with patch.dict(os.environ, {'EEG_MODEL_PATH': str(model_path),
                                         'EEG_SCALER_PATH': str(scaler_path)}):
                runtime = EEGModelRuntime.from_env()
            result = runtime.predict_proba([3, 4], feature_names=self.names)
            np.testing.assert_array_equal(runtime.scaler.seen, [[3, 4]])
            np.testing.assert_array_equal(runtime.model.seen, [[6, 12]])
            self.assertEqual(tuple(runtime.model.seen.columns), self.names)
            self.assertEqual(tuple(runtime.scaler.seen.columns), self.names)
            self.assertEqual(list(result), list(FakeModel.classes_))
            self.assertAlmostEqual(result['MATH_LOAD'], .6)
            self.assertTrue(all(type(p) is float for p in result.values()))

    def test_load_without_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pkl'
            joblib.dump(FakeModel(), path)
            with patch.dict(os.environ, {'EEG_MODEL_PATH': str(path), 'EEG_SCALER_PATH': ''}):
                runtime = EEGModelRuntime.from_env()
            self.assertIsNone(runtime.scaler)
            self.assertAlmostEqual(runtime.predict_proba([3, 4], feature_names=self.names)['MATH_LOAD'], .3)

    def test_configuration_errors(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ModelContractError, 'EEG_MODEL_PATH'):
                EEGModelRuntime.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pkl'
            with patch.dict(os.environ, {'EEG_MODEL_PATH': str(path), 'EEG_SCALER_PATH': ''}):
                with self.assertRaisesRegex(ModelContractError, 'no existe'):
                    EEGModelRuntime.from_env()
                path.write_text('invalid artifact')
                with self.assertRaisesRegex(ModelContractError, 'no se pudo cargar'):
                    EEGModelRuntime.from_env()
                joblib.dump(FakeModel(), path)
                os.environ['EEG_SCALER_PATH'] = str(Path(tmp) / 'missing.pkl')
                with self.assertRaisesRegex(ModelContractError, 'EEG_SCALER_PATH'):
                    EEGModelRuntime.from_env()

    def test_invalid_features(self):
        runtime = EEGModelRuntime(FakeModel())
        for value in ([], [[]], np.empty((0, 2)), [1], [1, 2, 3], [np.nan, 2],
                      [1, np.inf], [1, -np.inf], 3, [[[1, 2]]], ['1', '2'],
                      [1j, 2], [[1], [1, 2]]):
            with self.subTest(value=value), self.assertRaises(ModelContractError):
                runtime.predict_proba(value, feature_names=self.names)
        self.assertFalse(hasattr(runtime.model, 'seen'))

    def test_names_are_required_and_order_is_strict(self):
        runtime = EEGModelRuntime(FakeModel())
        with self.assertRaisesRegex(ModelContractError, 'Faltan nombres'):
            runtime.predict_proba([1, 2])
        for names in [('beta', 'theta'), ('theta', 'other'), ('theta', 'theta')]:
            with self.subTest(names=names), self.assertRaises(ModelContractError):
                runtime.predict_proba([1, 2], feature_names=names)
        with self.assertRaises(ModelContractError):
            runtime.predict_proba(pd.DataFrame([[1, 2]], columns=['beta', 'theta']))
        with self.assertRaises(ModelContractError):
            runtime.predict_proba(pd.DataFrame([[1, 2]], columns=self.names), feature_names=['beta', 'theta'])
        result = runtime.predict_proba(pd.DataFrame([[1, 2]], columns=self.names))
        self.assertEqual(result['MATH_LOAD'], .1)

    def test_batch_preserves_rows_and_single_rejects_multiple(self):
        runtime = EEGModelRuntime(FakeModel())
        result = runtime.predict_proba_batch([[1, 2], [3, 4]], feature_names=self.names)
        self.assertEqual([r['MATH_LOAD'] for r in result], [.1, .3])
        with self.assertRaisesRegex(ModelContractError, 'predict_proba_batch'):
            runtime.predict_proba([[1, 2], [3, 4]], feature_names=self.names)
        self.assertEqual(runtime.predict_proba([[1, 2]], feature_names=self.names), result[0])

    def test_adapter_integration(self):
        result = EEGModelRuntime(FakeModel(), FakeScaler()).predict_proba([3, 4], feature_names=self.names)
        self.assertEqual(to_wavesense_probabilities(result),
                         {'REST': .2, 'LOW_LOAD': .2, 'HIGH_LOAD': .6, 'ARTIFACT': 0.0})
        self.assertEqual(to_wavesense_probabilities(result, signal_has_artifact=True),
                         {'REST': 0.0, 'LOW_LOAD': 0.0, 'HIGH_LOAD': 0.0, 'ARTIFACT': 1.0})

    def test_missing_schema_requires_explicit_training_contract(self):
        class UnnamedModel:
            n_features_in_ = 2
            classes_ = FakeModel.classes_
            predict_proba = FakeModel.predict_proba
        with self.assertRaisesRegex(ModelContractError, 'Falta el contrato'):
            EEGModelRuntime(UnnamedModel())
        runtime = EEGModelRuntime(UnnamedModel(), feature_names=self.names)
        runtime.predict_proba([1, 2], feature_names=self.names)
        self.assertIsInstance(runtime.model.seen, np.ndarray)
        # Los nombres pueden proceder exclusivamente del escalador.
        runtime = EEGModelRuntime(UnnamedModel(), FakeScaler())
        self.assertEqual(runtime.feature_names, self.names)
        runtime.predict_proba([1, 2], feature_names=self.names)

    def test_inconsistent_artifact_metadata(self):
        for change in ({'feature_names_in_': ['beta', 'theta']}, {'n_features_in_': 3}):
            scaler = FakeScaler()
            scaler.__dict__.update(change)
            with self.subTest(change=change), self.assertRaises(ModelContractError):
                EEGModelRuntime(FakeModel(), scaler)
        with self.assertRaises(ModelContractError):
            EEGModelRuntime(FakeModel(), feature_names=['other', 'names'])
        for change in ({'classes_': []}, {'classes_': ['REST', 'REST']},
                       {'classes_': [0, 1]}, {'predict_proba': None}):
            model = FakeModel()
            model.__dict__.update(change)
            with self.subTest(change=change), self.assertRaises(ModelContractError):
                EEGModelRuntime(model)
        with self.assertRaises(ModelContractError):
            EEGModelRuntime(FakeModel(), object())

    def test_invalid_scaler_output(self):
        for value in ([1, 2], [[np.nan, 2]], [[np.inf, 2]], [[1]], [[1, 2], [3, 4]],
                      pd.DataFrame([[1, 2]], columns=['beta', 'theta'])):
            scaler = FakeScaler()
            scaler.transform = lambda x: value
            runtime = EEGModelRuntime(FakeModel(), scaler)
            with self.subTest(value=value), self.assertRaises(ModelContractError):
                runtime.predict_proba([1, 2], feature_names=self.names)
            self.assertFalse(hasattr(runtime.model, 'seen'))

    def test_invalid_probabilities(self):
        for value in ([.1, .2, .7], [[np.nan, .2, .8]], [[1, 0]], [[-.1, .2, .9]],
                      [[1.1, 0, -.1]], [[.1, .1, .1]], [[.1, .2, .7], [.1, .2, .7]]):
            model = FakeModel()
            model.predict_proba = lambda x: value
            with self.subTest(value=value), self.assertRaises(ModelContractError):
                EEGModelRuntime(model).predict_proba([1, 2], feature_names=self.names)


if __name__ == '__main__':
    unittest.main()
