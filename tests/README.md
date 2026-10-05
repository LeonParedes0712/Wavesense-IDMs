# Tests

Esta carpeta contiene pruebas unitarias del pipeline EEG, adquisición simulada, runtime del modelo, adaptador, triggering y tutor, incluyendo el flujo completo desde muestras en memoria hasta una respuesta simulada.

Ejecutar desde la raíz del repositorio:

```bash
python -m unittest discover -s tests -v
```

Las pruebas del tutor usan clientes y alertas simulados; no requieren API key ni hacen llamadas reales a OpenAI o al navegador. Verifican persistencia, umbral, artefactos, cooldown, contexto mínimo, configuración y ausencia de efectos al cargar o ejecutar el tutor.

`test_models.py` usa modelos y escaladores falsos y archivos joblib temporales.
Verifica carga por variables de entorno, escalado previo, orden de clases y filas,
DataFrames y nombres explícitos, contratos ausentes/incompatibles, dimensiones,
NaN/inf, salidas inválidas y conexión directa con el adaptador (incluido ARTIFACT).
No depende de los `.pkl` locales reales, hardware, datasets ni OpenAI. Los archivos
temporales se eliminan al terminar cada prueba.

Pruebas del pipeline EEG:

- `test_eeg_processing.py`: configuración, forma/frecuencia/valores inválidos,
  calidad cruda, atenuación del filtro, ausencia de mutaciones, 32 columnas y
  potencia conocida de senos deterministas, límites inclusivos y suma de bins.
- `test_unicorn_stream.py`: importación sin SDK, errores Windows/Linux, selección
  de dispositivo, índices no contiguos al inicio del scan, buffer float32, cierre
  ante fallos/interrupción, fuentes finitas, ventanas solapadas y marcas de calidad.
- `test_eeg_pipeline.py`: runtime real con estimadores falsos en memoria, escalado,
  adaptador/trigger/tutor reales con cliente simulado, bloqueo por artefactos,
  reinicio de persistencia y ausencia de llamadas externas. No escribe modelos.

Ejecutar solo la integración simulada:

```bash
python -m unittest discover -s tests -p 'test_eeg_pipeline.py' -v
```

Los mocks verifican el contrato usado por el wrapper, no prueban compatibilidad
binaria ni comportamiento real del SDK. Falta validar en Windows los canales,
unidades, pérdidas de muestras, tiempo de bloqueo, cierre y tamaño de buffer.
Los senos son datos de prueba generados en memoria, no grabaciones ni evidencia
de rendimiento del clasificador. No se generan CSV, `.pkl` persistentes ni figuras.
