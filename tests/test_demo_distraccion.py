import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from scripts import demo_distraccion as demo


class DistractionDemoTests(unittest.TestCase):
    def test_complete_session_uses_datos_and_emits_one_event_per_cycle(self):
        output = io.StringIO()
        sleep = Mock()
        with contextlib.redirect_stdout(output), \
             patch.object(demo.datos, 'calcular_bandas', wraps=demo.datos.calcular_bandas) as analyze:
            demo.run(cycles=2, sleep=sleep)
        text = output.getvalue()
        self.assertEqual(text.count('Baseline listo'), 1)
        self.assertEqual(text.count('Se distrajo.'), 2)
        self.assertIn('Fase del guion: RECUPERACION', text)
        self.assertIn('Evento programado, no detección real', text)
        self.assertIn('Sesión finalizada', text)
        self.assertEqual(analyze.call_count, 112)  # Baseline + 111 ventanas móviles.
        self.assertEqual(len(analyze.call_args_list[0].args[0]), 5000)
        self.assertTrue(all(len(call.args[0]) == 2500 for call in analyze.call_args_list[1:]))
        self.assertEqual(sleep.call_count, 140)

    def test_invalid_options_rejected(self):
        for speed in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                demo.run(speed=speed)
        with self.assertRaises(ValueError):
            demo.run(cycles=-1)

    def test_ctrl_c_exits_cleanly(self):
        with patch.object(demo, 'run', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(demo.main([]), 0)
        self.assertIn('detenida por el usuario', output.getvalue())

    def test_import_does_not_start_hardware_or_session(self):
        result = subprocess.run([sys.executable, '-c',
            "from scripts import demo_distraccion; import sys; "
            "assert 'UnicornPy' not in sys.modules; assert 'openai' not in sys.modules"],
            cwd=Path(__file__).parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
