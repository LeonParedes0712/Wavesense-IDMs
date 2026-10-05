"""Contrato de adquisición; estos dobles no validan hardware físico."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import unicorn_smoke_test as smoke
from src.unicorn_stream import AcquisitionError, UnicornSource


class AcquisitionTests(unittest.TestCase):
    def sdk(self, frame_length=25):
        state = {'started': False, 'offset': 0, 'buffers': []}
        device = Mock()

        def idle(value):
            def call(*args):
                self.assertFalse(state['started'], 'consulta durante adquisición')
                return value(*args) if callable(value) else value
            return call

        device.GetNumberOfAcquiredChannels.side_effect = idle(17)
        device.GetChannelIndex.side_effect = idle(lambda name: int(name.split()[1]) + 1)

        def start(test_signal):
            self.assertIs(test_signal, False)
            state['started'] = True

        def read(count, buffer, size):
            self.assertTrue(state['started'])
            self.assertEqual(count, frame_length)
            self.assertEqual(size, frame_length * 17 * 4)
            self.assertEqual(len(buffer), size)
            state['buffers'].append(buffer)
            values = np.arange(state['offset'], state['offset'] + count * 17, dtype=np.float32)
            buffer[:] = values.tobytes()
            state['offset'] += count * 17

        device.StartAcquisition.side_effect = start
        device.GetData.side_effect = read
        sdk = SimpleNamespace(SamplingRate=250, GetApiVersion=Mock(return_value='test-api'),
                              GetAvailableDevices=Mock(side_effect=idle(['unit'])),
                              Unicorn=Mock(return_value=device))
        return sdk, device, state

    def test_smoke_ten_float32_blocks_and_persistent_buffer(self):
        sdk, device, state = self.sdk()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            smoke.acquire(sdk)
        sdk.GetAvailableDevices.assert_called_once_with(True)
        sdk.Unicorn.assert_called_once_with('unit')
        device.StartAcquisition.assert_called_once_with(False)
        device.StopAcquisition.assert_called_once_with()
        self.assertEqual(device.GetData.call_count, 10)
        self.assertTrue(all(b is state['buffers'][0] for b in state['buffers']))
        self.assertEqual(output.getvalue().count('Forma: (25, 17)'), 10)
        self.assertIn('mínimo: 0.0; máximo: 424.0', output.getvalue())
        self.assertIn('mínimo: 3825.0; máximo: 4249.0', output.getvalue())
        self.assertIn('EEG 1: índice 2', output.getvalue())
        self.assertIn('test-api', output.getvalue())

    def test_smoke_stop_on_start_read_and_interrupt_failures(self):
        for operation, error in [('StartAcquisition', RuntimeError), ('GetData', RuntimeError),
                                 ('GetData', KeyboardInterrupt)]:
            with self.subTest(operation=operation, error=error):
                sdk, device, _ = self.sdk()
                getattr(device, operation).side_effect = error('failure')
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(error):
                    smoke.acquire(sdk)
                device.StopAcquisition.assert_called_once_with()

    def test_reshape_failure_stops_both_readers(self):
        for reader in ('smoke', 'source'):
            with self.subTest(reader=reader):
                sdk, device, _ = self.sdk()
                # Simular respuesta truncada: frombuffer funciona, reshape debe fallar.
                device.GetData.side_effect = lambda count, buffer, size: buffer.__setitem__(
                    slice(None), np.zeros(3, dtype=np.float32).tobytes())
                if reader == 'smoke':
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                        smoke.acquire(sdk)
                else:
                    source = UnicornSource()
                    source._sdk = sdk
                    source.connect()
                    with self.assertRaises(AcquisitionError):
                        source.read_samples(25)
                device.StopAcquisition.assert_called_once_with()

    def test_selection(self):
        self.assertEqual(smoke.select_serial(['one']), 'one')
        self.assertEqual(smoke.select_serial(['one', 'two'], 'two'), 'two')
        for devices, requested in [(None, None), (['one', 'two'], None), (['one'], 'bad')]:
            with self.subTest(devices=devices), self.assertRaises(RuntimeError):
                smoke.select_serial(devices, requested)
        with self.assertRaisesRegex(RuntimeError, "one.*two"):
            smoke.select_serial(['one', 'two'])

    def test_source_fixed_frames_preserve_samples_across_reads(self):
        sdk, device, state = self.sdk(frame_length=7)
        source = UnicornSource(frame_length=7)
        source._sdk = sdk
        original_start = device.StartAcquisition.side_effect

        def start_with_diagnostics(flag):
            self.assertEqual(len(source.channel_diagnostics), 8)
            self.assertEqual(len(source._buffer), 7 * 17 * 4)
            original_start(flag)

        device.StartAcquisition.side_effect = start_with_diagnostics
        source.connect()
        try:
            self.assertEqual(len(source._buffer), 7 * 17 * 4)
            self.assertEqual(source.channel_diagnostics, tuple((f'EEG {i}', i + 1) for i in range(1, 9)))
            with self.assertRaises(AcquisitionError):
                source.discover()
            blocks = [source.read_samples(n).samples for n in (3, 12, 6)]
            actual = np.concatenate(blocks)
            expected = np.arange(21 * 17, dtype=np.float32).reshape(21, 17)[:, 2:10]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(device.GetData.call_count, 3)
            self.assertTrue(all(b is state['buffers'][0] for b in state['buffers']))
            sdk.GetAvailableDevices.assert_called_once_with(True)
            device.GetConfiguration.assert_not_called()
        finally:
            source.close()
        device.StopAcquisition.assert_called_once_with()

    def test_frame_configuration(self):
        with patch.dict(os.environ, {'UNICORN_FRAME_LENGTH': '13'}):
            self.assertEqual(UnicornSource.from_env().frame_length, 13)
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                UnicornSource(frame_length=value)

    def test_smoke_windows_loader_keeps_dll_handle_until_stop(self):
        sdk, device, _ = self.sdk()
        handle = Mock()
        device.StopAcquisition.side_effect = lambda: handle.close.assert_not_called()
        before = list(sys.path)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'UNICORN_PYTHON_PATH': directory, 'UNICORN_SERIAL': ''}), \
             patch.object(smoke.sys, 'platform', 'win32'), \
             patch('dotenv.load_dotenv'), \
             patch.object(smoke.os, 'add_dll_directory', return_value=handle, create=True), \
             patch.object(smoke.importlib, 'import_module', return_value=sdk), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(), 0)
        self.assertEqual(sys.path, before)
        handle.close.assert_called_once_with()

    def test_smoke_is_independent_and_rejects_linux(self):
        code = '''
import sys
from scripts import unicorn_smoke_test as smoke
assert not any(name in sys.modules for name in
               ('src.unicorn_stream', 'src.eeg_config', 'src.models', 'src.pipeline', 'pandas'))
sys.platform = 'linux'
assert smoke.main() == 1
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
