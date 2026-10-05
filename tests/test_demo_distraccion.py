import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from scripts import demo_distraccion as demo


class DistractionDemoTests(unittest.TestCase):
    def setUp(self):
        self.browser = self.enterContext(patch('webbrowser.open', return_value=True))
        self.enterContext(patch.dict(os.environ, {
            'DISTRACTION_VIDEO_URL': 'https://www.tiktok.com/'
        }))

    def test_complete_session_uses_datos_and_emits_one_event_per_cycle(self):
        output = io.StringIO()
        sleep = Mock()
        opening_times = []
        self.browser.side_effect = lambda url: opening_times.append(sleep.call_count) or True
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
        self.assertEqual(opening_times, [43, 103])  # Primer evento: 43 / 5 = 8.6 s.
        self.assertEqual(self.browser.call_count, 2)
        self.browser.assert_called_with('https://www.tiktok.com/')

    def test_browser_failure_does_not_stop_demo(self):
        for result in (False, RuntimeError('browser failed')):
            with self.subTest(result=result):
                self.browser.reset_mock(side_effect=True)
                if isinstance(result, Exception):
                    self.browser.side_effect = result
                else:
                    self.browser.return_value = result
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    demo.run(cycles=1, sleep=Mock())
                self.assertIn('No se pudo abrir', output.getvalue())
                self.assertIn('Sesión finalizada', output.getvalue())
                self.browser.assert_called_once()

    def test_missing_url_keeps_running_without_browser(self):
        with patch.dict(os.environ, {'DISTRACTION_VIDEO_URL': ''}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            demo.run(cycles=1, sleep=Mock())
        self.browser.assert_not_called()
        self.assertIn('Configura DISTRACTION_VIDEO_URL', output.getvalue())
        self.assertIn('Sesión finalizada', output.getvalue())

    def test_main_loads_repo_dotenv(self):
        with patch('dotenv.load_dotenv') as load, patch.object(demo, 'run') as run:
            self.assertEqual(demo.main(['--cycles', '1']), 0)
        load.assert_called_once_with(Path(demo.__file__).resolve().parents[1] / '.env')
        run.assert_called_once_with(speed=5, cycles=1)

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
