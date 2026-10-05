"""Entrada directa al proceso de datos.py, sin otra capa de adquisición."""

import sys


def main():
    from datos import main as ejecutar_datos

    try:
        ejecutar_datos()
    except KeyboardInterrupt:
        print('\nSesión interrumpida.')
        return 0
    except Exception as exc:
        print(f'Error en datos.py: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
