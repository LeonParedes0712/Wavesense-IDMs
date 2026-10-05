"""Entrada directa al archivo datos.py aportado por el usuario."""
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

import datos
import main as entry


class DatosMainTests(unittest.TestCase):
    def test_entry_calls_existing_process_only_once(self):
        with patch.object(datos, 'main') as run:
            self.assertEqual(entry.main(), 0)
        run.assert_called_once_with()

    def test_import_does_not_load_sdk_or_other_pipeline(self):
        result = subprocess.run([sys.executable, '-c',
            "import main, datos, sys; assert datos.UnicornPy is None; "
            "assert not any(m in sys.modules for m in "
            "('UnicornPy', 'src.unicorn_stream', 'src.pipeline', 'openai'))"],
            cwd=Path(__file__).parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sdk_setup_uses_environment_and_runs_once(self):
        handle = Mock()
        sdk = SimpleNamespace(SamplingRate=250)
        before = list(sys.path)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'UNICORN_PYTHON_PATH': directory}), \
             patch.object(datos.sys, 'platform', 'win32'), \
             patch('dotenv.load_dotenv'), \
             patch.object(datos.os, 'add_dll_directory', return_value=handle, create=True) as add, \
             patch('importlib.import_module', return_value=sdk), \
             patch.object(datos, 'adquirir_y_analizar') as run:
            datos.main()
            add.assert_called_once_with(str(Path(directory).resolve()))
        run.assert_called_once_with()
        handle.close.assert_called_once_with()
        self.assertEqual(sys.path, before)
        self.assertIsNone(datos.UnicornPy)

    def test_original_session_starts_once_and_stops_on_read_error(self):
        device = Mock()
        device.GetNumberOfAcquiredChannels.return_value = 17
        device.GetData.side_effect = RuntimeError('read failed')
        sdk = Mock()
        sdk.GetAvailableDevices.return_value = ['unit']
        sdk.Unicorn.return_value = device
        with patch.object(datos, 'UnicornPy', sdk), patch('builtins.input', return_value='0'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'read failed'):
                datos.adquirir_y_analizar()
        sdk.GetAvailableDevices.assert_called_once_with(True)
        sdk.Unicorn.assert_called_once_with('unit')
        device.StartAcquisition.assert_called_once_with(False)
        device.StopAcquisition.assert_called_once_with()
        frame, buffer, size = device.GetData.call_args.args
        self.assertEqual((frame, len(buffer), size), (25, 1700, 1700))
