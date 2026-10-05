"""Adquisición EEG continua en Windows; detener con Ctrl+C, pipeline por ventanas de diez segundos."""

from pathlib import Path
import sys

# Permite ejecutar el archivo directamente desde cualquier directorio.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.unicorn_stream import UnicornSource


def create_pipeline():
    # Importación diferida: importar el ejecutable no carga modelos ni servicios.
    from dataclasses import replace
    import os

    from src.eeg_config import EEGConfig, QualityConfig
    from src.models import EEGModelRuntime
    from src.pipeline import EEGPipeline
    from src.triggering import TemporalTrigger

    config = replace(EEGConfig.from_env(), window_seconds=10)
    model_path = os.getenv('EEG_MODEL_PATH', '').strip()
    runtime = None
    if model_path and Path(model_path).expanduser().is_file():
        runtime = EEGModelRuntime.from_env()
    else:
        print('Modelo no configurado: mostrando adquisición, calidad y características EEG.', flush=True)
    return EEGPipeline(runtime, config=config, quality=QualityConfig.from_env(),
                       trigger=TemporalTrigger(), tutor=None)


def print_result(result):
    print('Calidad: ' + ('ARTIFACT' if result.quality.has_artifact else 'válida'), flush=True)
    if result.quality.reasons:
        print('Motivos: ' + ', '.join(result.quality.reasons), flush=True)
    if result.features is not None:
        # Mostrar las 32 columnas exactas, agrupadas por canal, sin recalcular bandas.
        values = result.features.iloc[0]
        print('Características EEG (32):', flush=True)
        for channel in range(1, 9):
            prefix = f'Canal_{channel}_'
            print(f'  Canal {channel}: ' + ' | '.join(
                f'{name[len(prefix):]}={value:.6g}'
                for name, value in values.items() if name.startswith(prefix)), flush=True)
    if result.model_probabilities is not None:
        print(f'Probabilidades originales: {result.model_probabilities}', flush=True)
        print(f'Probabilidades Wavesense: {result.wavesense_probabilities}', flush=True)
    if result.decision is not None:
        print(f'TemporalTrigger: {result.decision}', flush=True)


def run(source, pipeline=None):
    """Ventanas reales de diez segundos; WindowStream gestiona el cierre seguro."""
    from src.models import ModelContractError
    from src.unicorn_stream import WindowStream

    pipeline = create_pipeline() if pipeline is None else pipeline
    with WindowStream(source, config=pipeline.config,
                      chunk_samples=source.frame_length) as stream:
        print(f'Serial: {source.serial}; SamplingRate: {source.sample_rate} Hz', flush=True)
        print('Canales EEG: ' + ', '.join(
            f'{name}={index}' for name, index in source.channel_diagnostics), flush=True)
        print('Adquisición continua, ventanas de 10 segundos. Ctrl+C para detener.', flush=True)
        number = 0
        while True:
            block = stream.next_window()
            number += 1
            print(f'Ventana #{number} | shape={block.samples.shape} | fs={block.sample_rate}', flush=True)
            try:
                result = pipeline.process_window(
                    block.samples, sample_rate=block.sample_rate,
                    signal_has_artifact=block.has_artifact)
            except ModelContractError as exc:
                print(f'Error de modelo (contrato de 32 columnas): {exc}', file=sys.stderr, flush=True)
                continue
            print_result(result)


def main():
    if sys.platform != 'win32':
        print('La adquisición real requiere Windows + Unicorn Suite/UnicornPy.', file=sys.stderr)
        return 1
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / '.env')
        run(UnicornSource.from_env())
    except KeyboardInterrupt:
        print('\nAdquisición detenida por el usuario.', flush=True)
        return 0
    except Exception as exc:
        print(f'Error de adquisición: {exc}\n'
              'Para diagnosticar el SDK, ejecuta: python scripts/unicorn_smoke_test.py',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
