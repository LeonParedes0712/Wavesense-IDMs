"""Pruebas del ejecutable continuo sin dispositivo físico."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import run_unicorn_pipeline as runner
from src.unicorn_stream import AcquisitionError, SampleBlock, UnicornSource


class ContinuousRunnerTests(unittest.TestCase):
    def source(self):
        return Mock(serial='unit', sample_rate=250, frame_length=25,
                    channel_diagnostics=tuple((f'EEG {i}', i - 1) for i in range(1, 9)))

    def test_continues_beyond_ten_blocks_until_interrupt(self):
        source = self.source()
        block = SampleBlock(np.ones((25, 8), dtype=np.float32), 250)
        source.read_samples.side_effect = [block] * 15 + [KeyboardInterrupt()]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(source)
        source.connect.assert_called_once_with()
        self.assertGreaterEqual(source.close.call_count, 1)
        self.assertEqual(source.read_samples.call_count, 16)
        self.assertTrue(all(call.args == (25,) for call in source.read_samples.call_args_list))
        self.assertEqual(output.getvalue().count('Adquisición continua'), 1)
        self.assertNotIn('Ventana #', output.getvalue())  # Aún no hay diez segundos.

    def test_real_source_wrapper_stops_sdk_on_interrupt(self):
        device = Mock()
        device.GetNumberOfAcquiredChannels.return_value = 17
        device.GetChannelIndex.side_effect = lambda name: int(name.split()[1]) - 1
        calls = 0

        def read(count, buffer, size):
            nonlocal calls
            calls += 1
            if calls == 13:
                raise KeyboardInterrupt
            self.assertEqual((count, len(buffer), size), (25, 1700, 1700))
            buffer[:] = np.ones((25, 17), dtype=np.float32).tobytes()

        device.GetData.side_effect = read
        sdk = Mock(SamplingRate=250)
        sdk.GetAvailableDevices.return_value = ['unit']
        sdk.Unicorn.return_value = device
        source = UnicornSource()
        source._sdk = sdk
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            runner.run(source)
        sdk.GetAvailableDevices.assert_called_once_with(True)
        device.StartAcquisition.assert_called_once_with(False)
        device.StopAcquisition.assert_called_once_with()
        self.assertEqual(calls, 13)

    def test_connection_and_read_failures_close_source(self):
        for operation in ('connect', 'read_samples'):
            source = self.source()
            getattr(source, operation).side_effect = AcquisitionError('original error')
            source.close.side_effect = AcquisitionError('close error')
            with self.subTest(operation=operation), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaisesRegex(AcquisitionError, 'original error'):
                    runner.run(source)
            self.assertGreaterEqual(source.close.call_count, 1)
            # WindowStream preserva el error original si también falla close().

    def test_main_loads_environment_and_reports_exit_status(self):
        for error, expected in ((KeyboardInterrupt(), 0), (AcquisitionError('failure'), 1)):
            with self.subTest(expected=expected), patch.object(runner.sys, 'platform', 'win32'), \
                 patch('dotenv.load_dotenv') as load, \
                 patch.object(runner.UnicornSource, 'from_env') as factory, \
                 patch.object(runner, 'run', side_effect=error) as run, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(), expected)
                load.assert_called_once_with(Path(runner.__file__).resolve().parents[1] / '.env')
                run.assert_called_once_with(factory.return_value)

    def test_direct_script_resolves_source_and_rejects_non_windows(self):
        if sys.platform == 'win32':
            self.skipTest('La ejecución directa en Windows requiere hardware; cubierta con mocks.')
        script = Path(runner.__file__).resolve()
        result = subprocess.run([sys.executable, str(script)], cwd=script.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('requiere Windows', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_import_does_not_load_model_or_connect(self):
        code = '''
import sys
from scripts import run_unicorn_pipeline
assert not any(name in sys.modules for name in
    ('UnicornPy', 'src.models', 'src.pipeline', 'src.tutor', 'openai', 'pandas'))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
