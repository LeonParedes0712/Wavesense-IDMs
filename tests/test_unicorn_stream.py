from dataclasses import replace
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.eeg_config import EEGConfig
from src.unicorn_stream import AcquisitionError, ArraySource, SampleBlock, UnicornSource, WindowStream


class UnicornStreamTests(unittest.TestCase):
    def setUp(self):
        self.config = EEGConfig()

    def sdk(self):
        device = Mock()
        device.GetNumberOfAcquiredChannels.return_value = 10
        # Primeros dos campos auxiliares: detecta selección posicional incorrecta.
        device.GetChannelIndex.side_effect = lambda name: int(name.split()[1]) + 1

        def get_data(count, buffer, size):
            array = np.tile(np.arange(10, dtype=np.float32), (count, 1))
            self.assertEqual(size, array.nbytes)
            buffer[:] = array.tobytes()
        device.GetData.side_effect = get_data
        sdk = SimpleNamespace(SamplingRate=250, GetAvailableDevices=Mock(return_value=['unit']),
                              Unicorn=Mock(return_value=device))
        return sdk, device

    def test_import_and_module_execution_do_not_load_sdk(self):
        # run_path en nombre distinto de __main__: ambos módulos carecen de bucle de entrada.
        with patch('importlib.import_module', side_effect=AssertionError('SDK import forbidden')):
            for module in ('unicorn_stream', 'pipeline'):
                runpy.run_path(str(Path(__file__).parents[1] / 'src' / f'{module}.py'), run_name='__main__')

    def test_linux_and_missing_windows_sdk_errors(self):
        with patch('src.unicorn_stream.sys.platform', 'linux'), \
             patch('src.unicorn_stream.importlib.import_module') as load:
            with self.assertRaisesRegex(AcquisitionError, 'Windows'):
                UnicornSource().connect()
            load.assert_not_called()
        with patch('src.unicorn_stream.sys.platform', 'win32'), \
             patch('src.unicorn_stream.importlib.import_module', side_effect=ImportError('missing')):
            with self.assertRaisesRegex(AcquisitionError, 'UNICORN_PYTHON_PATH'):
                UnicornSource().connect()

    def test_sdk_channel_mapping_buffer_and_safe_close(self):
        sdk, device = self.sdk()
        with patch('src.unicorn_stream.sys.platform', 'win32'), \
             patch('src.unicorn_stream.importlib.import_module', return_value=sdk):
            source = UnicornSource()
            with WindowStream(source, config=self.config, chunk_samples=25) as stream:
                block = stream.next_window()
                self.assertEqual(block.samples.shape, (250, 8))
                np.testing.assert_array_equal(block.samples[0], np.arange(2, 10))
                self.assertEqual(block.sample_rate, 250)
                self.assertEqual(device.GetData.call_count, 10)
            source.close()
        device.StartAcquisition.assert_called_once_with(False)
        device.StopAcquisition.assert_called_once()
        self.assertIsNone(source._device)

    def test_missing_ambiguous_and_selected_device(self):
        for devices, serial in (([], None), (['a', 'b'], None), (['a'], 'b')):
            sdk, device = self.sdk()
            sdk.GetAvailableDevices.return_value = devices
            with self.subTest(devices=devices), patch.object(UnicornSource, '_load_sdk', return_value=sdk):
                source = UnicornSource(serial=serial)
                # _load_sdk normally stores SDK; injection here keeps the same state.
                source._sdk = sdk
                with self.assertRaises(AcquisitionError):
                    source.connect()
                sdk.Unicorn.assert_not_called()
        sdk, device = self.sdk()
        sdk.GetAvailableDevices.return_value = ['a', 'b']
        source = UnicornSource(serial='b')
        source._sdk = sdk
        source.connect()
        sdk.Unicorn.assert_called_once_with('b')
        source.close()

    def test_sdk_start_read_and_stop_failures_release_references(self):
        for operation in ('StartAcquisition', 'GetData', 'StopAcquisition'):
            sdk, device = self.sdk()
            source = UnicornSource()
            source._sdk = sdk
            getattr(device, operation).side_effect = RuntimeError('SDK failed')
            with self.subTest(operation=operation), self.assertRaises(AcquisitionError):
                if operation == 'StartAcquisition':
                    source.connect()
                else:
                    source.connect()
                    if operation == 'GetData':
                        source.read_samples(25)
                    else:
                        source.close()
            self.assertIsNone(source._device)
            self.assertFalse(source._started)
            device.StopAcquisition.assert_called_once()

    def test_sdk_interrupt_closes_device(self):
        sdk, device = self.sdk()
        source = UnicornSource()
        source._sdk = sdk
        source.connect()
        device.GetData.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            source.read_samples(25)
        device.StopAcquisition.assert_called_once()

    def test_python_and_dll_paths_are_scoped_to_source(self):
        sdk, device = self.sdk()
        handle = Mock()
        before = list(sys.path)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('src.unicorn_stream.sys.platform', 'win32'), \
             patch('src.unicorn_stream.os.add_dll_directory', return_value=handle, create=True) as add, \
             patch('src.unicorn_stream.importlib.import_module', return_value=sdk):
            source = UnicornSource(python_path=tmp)
            source.connect()
            add.assert_called_once_with(str(Path(tmp).resolve()))
            self.assertEqual(sys.path, before)
            handle.close.assert_not_called()
            source.close()
            handle.close.assert_called_once()
        with patch('src.unicorn_stream.sys.platform', 'win32'):
            with self.assertRaisesRegex(AcquisitionError, 'directorio'):
                UnicornSource(python_path='/this-directory-does-not-exist').connect()

    def test_overlapping_windows_preserve_samples_across_partial_blocks(self):
        config = replace(self.config, hop_seconds=.5)
        array = np.arange(500 * 8).reshape(500, 8)
        source = ArraySource(array, sample_rate=250)
        with WindowStream(source, config=config, chunk_samples=17) as stream:
            for start in (0, 125, 250):
                np.testing.assert_array_equal(stream.next_window().samples, array[start:start + 250])
            with self.assertRaises(EOFError):
                stream.next_window()
        self.assertFalse(source.connected)

    def test_source_artifact_propagates_to_overlapping_windows(self):
        config = replace(self.config, hop_seconds=.5)
        source = Mock()
        source.read_samples.side_effect = [SampleBlock(np.ones((125, 8)), 250),
                                          SampleBlock(np.ones((125, 8)), 250, True),
                                          SampleBlock(np.ones((125, 8)), 250),
                                          SampleBlock(np.ones((125, 8)), 250)]
        with WindowStream(source, config=config, chunk_samples=125) as stream:
            self.assertTrue(stream.next_window().has_artifact)
            self.assertTrue(stream.next_window().has_artifact)
            self.assertFalse(stream.next_window().has_artifact)
        source.close.assert_called_once()

    def test_bad_source_reads_and_partial_windows_never_hang(self):
        for block in (SampleBlock(np.empty((0, 8)), 250), SampleBlock(np.zeros((26, 8)), 250),
                      SampleBlock(np.zeros((25, 7)), 250), SampleBlock(np.zeros((25, 8)), 200)):
            source = Mock()
            source.read_samples.return_value = block
            with self.subTest(block=block), self.assertRaises(AcquisitionError):
                with WindowStream(source, config=self.config) as stream:
                    stream.next_window()
            self.assertGreaterEqual(source.close.call_count, 1)
        source = ArraySource(np.zeros((249, 8)), sample_rate=250)
        with self.assertRaises(EOFError):
            with WindowStream(source, config=self.config) as stream:
                stream.next_window()
        self.assertFalse(source.connected)

    def test_pipeline_failure_inside_context_closes_source(self):
        source = ArraySource(np.ones((250, 8)), sample_rate=250)
        with self.assertRaisesRegex(RuntimeError, 'processing failed'):
            with WindowStream(source, config=self.config) as stream:
                stream.next_window()
                raise RuntimeError('processing failed')
        self.assertFalse(source.connected)

    def test_environment_has_no_connection_side_effect(self):
        with patch.dict(os.environ, {'UNICORN_SERIAL': 'chosen', 'UNICORN_PYTHON_PATH': 'local-sdk',
                                     'UNICORN_EEG_CHANNEL_NAMES': ','.join(f'EEG {i}' for i in range(8, 0, -1))}):
            source = UnicornSource.from_env()
        self.assertEqual(source.serial, 'chosen')
        self.assertEqual(source.channel_names[0], 'EEG 8')
        self.assertIsNone(source._sdk)
