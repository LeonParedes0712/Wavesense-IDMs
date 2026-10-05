"""Comprobación manual: hace una llamada REAL a OpenAI, sin Unicorn ni navegador."""

import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from dotenv import load_dotenv
    from src.triggering import TriggerDecision
    from src.tutor import Tutor

    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    for name in ('OPENAI_API_KEY', 'OPENAI_MODEL'):
        if not os.getenv(name, '').strip():
            print(f'Configura {name} en tu .env local antes de ejecutar la prueba.', file=sys.stderr)
            return 1
    print('Prueba manual: se hará una llamada REAL a OpenAI (puede generar costo).', flush=True)
    decision = TriggerDecision(
        triggered=True, reason='sustained_high_load', state='HIGH_LOAD',
        confidence=0.9, consecutive_high_load_windows=0, cooldown_remaining_seconds=30.0,
    )
    try:
        text = Tutor(model=os.environ['OPENAI_MODEL'].strip()).respond(decision)
    except Exception as exc:
        print(f'Error del tutor/OpenAI: {exc}', file=sys.stderr)
        return 1
    print(text, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
