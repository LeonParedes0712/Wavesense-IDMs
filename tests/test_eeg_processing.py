import os
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import welch, windows

from src.eeg_config import EEGConfig, FEATURE_NAMES, QualityConfig
from src.features import extract_band_powers
from src.preprocessing import assess_quality, filter_window, validate_window


def sine_window(config, frequency=10):
    time = np.arange(config.window_samples) / config.sample_rate
    return np.sin(2 * np.pi * frequency * time[:, None]) * np.arange(1, 9)[None, :]


class EEGProcessingTests(unittest.TestCase):
    def setUp(self):
        self.config = EEGConfig()
        self.window = sine_window(self.config)

    def test_invalid_signals_and_rates(self):
        for value in ([], np.zeros((249, 8)), np.zeros((250, 7)), np.zeros((8, 250)),
                      np.zeros((250, 8, 1)), [['text']], np.ones((250, 8), dtype=bool),
                      np.full((250, 8), np.nan), np.full((250, 8), np.inf),
                      np.full((250, 8), -np.inf), np.ones((250, 8), dtype=complex)):
            with self.subTest(shape=np.shape(value)), self.assertRaises(ValueError):
                validate_window(value, sample_rate=250, config=self.config)
        for rate in (0, -1, np.nan, np.inf, 200):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                validate_window(self.window, sample_rate=rate, config=self.config)

    def test_validation_and_filter_do_not_mutate_input(self):
        original = self.window.copy()
        validated = validate_window(self.window, sample_rate=250, config=self.config)
        validated[0] = 100
        filtered = filter_window(self.window, sample_rate=250, config=self.config)
        np.testing.assert_array_equal(self.window, original)
        self.assertEqual(filtered.shape, original.shape)
        self.assertTrue(np.isfinite(filtered).all())

    def test_filter_attenuates_out_of_band_signal(self):
        config = replace(self.config, window_seconds=4)
        array = sine_window(config, frequency=80)
        filtered = filter_window(array, sample_rate=250, config=config)
        self.assertLess(np.std(filtered[250:-250]), np.std(array[250:-250]) * .05)

    def test_filter_disabled_reference_and_short_window(self):
        config = replace(self.config, filter_enabled=False)
        np.testing.assert_array_equal(filter_window(self.window, sample_rate=250, config=config), self.window)
        config = replace(config, reference='common_average')
        filtered = filter_window(self.window, sample_rate=250, config=config)
        np.testing.assert_allclose(filtered.mean(axis=1), 0, atol=1e-14)
        config = replace(self.config, window_seconds=.04, hop_seconds=.04)
        with self.assertRaisesRegex(ValueError, 'corta'):
            filter_window(np.ones((10, 8)), sample_rate=250, config=config)

    def test_quality_checks_unfiltered_signal_and_returns_reasons(self):
        self.assertFalse(assess_quality(self.window, sample_rate=250, config=self.config,
                                       quality=QualityConfig()).has_artifact)
        bad = self.window.copy()
        bad[:, 2] = 10
        quality = assess_quality(bad, sample_rate=250, config=self.config, quality=QualityConfig())
        self.assertIn('flatline:Canal_3', quality.reasons)
        for limits, reason in ((QualityConfig(max_abs=2), 'amplitude_limit'),
                               (QualityConfig(max_step=.01), 'step_limit')):
            quality = assess_quality(self.window, sample_rate=250, config=self.config, quality=limits)
            self.assertTrue(quality.has_artifact)
            self.assertIn(reason, quality.reasons)

    def test_exact_schema_and_sine_power_with_channel_order(self):
        features = extract_band_powers(self.window, sample_rate=250, config=self.config)
        expected = tuple(f'Canal_{c}_{b}' for c in range(1, 9) for b in ('Theta', 'Alpha', 'Beta', 'Gamma'))
        self.assertEqual(tuple(features.columns), expected)
        self.assertEqual(FEATURE_NAMES, expected)
        self.assertEqual(features.shape, (1, 32))
        # 1 Hz bins y seno puro: suma de PSD = amplitud²/2.
        for c in range(1, 9):
            self.assertAlmostEqual(features[f'Canal_{c}_Alpha'][0], c * c / 2, places=10)
            self.assertLess(features[f'Canal_{c}_Gamma'][0], 1e-20)

    def test_inclusive_boundaries_and_sum_not_integral(self):
        config = replace(self.config, window_seconds=2, welch_nperseg=500)
        array = sine_window(config, frequency=8)
        features = extract_band_powers(array, sample_rate=250, config=config)
        frequencies, psd = welch(array, fs=250, window=windows.hann(500, sym=False),
                                 nperseg=500, noverlap=250, axis=0)
        for band, low, high in (('Theta', 4, 8), ('Alpha', 8, 12)):
            expected = psd[(frequencies >= low) & (frequencies <= high), 0].sum()
            self.assertAlmostEqual(features[f'Canal_1_{band}'][0], expected)
        self.assertGreater(features.Canal_1_Theta[0] + features.Canal_1_Alpha[0], psd[:, 0].sum())

    def test_empty_frequency_band_is_explicit_error(self):
        config = replace(self.config, bands=((4.1, 4.2), (8, 12), (12, 30), (30, 45)))
        with self.assertRaisesRegex(ValueError, 'sin bins'):
            extract_band_powers(self.window, sample_rate=250, config=config)

    def test_environment_and_invalid_configurations(self):
        with patch.dict(os.environ, {'EEG_WINDOW_SECONDS': '2', 'EEG_HOP_SECONDS': '.5',
                                     'EEG_GAMMA_HZ': '30,40', 'EEG_FILTER_ENABLED': 'false',
                                     'EEG_MAX_ABS': '100', 'EEG_REFERENCE': 'common_average'}, clear=True):
            config = EEGConfig.from_env()
            self.assertEqual(config.window_samples, 500)
            self.assertEqual(config.hop_samples, 125)
            self.assertEqual(config.bands[-1], (30, 40))
            self.assertFalse(config.filter_enabled)
            self.assertEqual(config.reference, 'common_average')
            self.assertEqual(QualityConfig.from_env().max_abs, 100)
        for change in ({'sample_rate': 0}, {'sample_rate': np.nan}, {'channel_count': 7},
                       {'window_seconds': .001}, {'hop_seconds': 2}, {'filter_high': 125},
                       {'welch_overlap': 1}, {'welch_nperseg': 1}, {'filter_order': 0},
                       {'reference': 'unknown'}, {'input_units': 'mystery'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)
        for name, value in (('EEG_SAMPLE_RATE', 'bad'), ('EEG_GAMMA_HZ', '30'),
                            ('EEG_FILTER_ENABLED', 'maybe'), ('EEG_MAX_ABS', '-1')):
            with patch.dict(os.environ, {name: value}, clear=True), self.assertRaises(ValueError):
                if name == 'EEG_MAX_ABS':
                    QualityConfig.from_env()
                else:
                    EEGConfig.from_env()
