# Tests

Esta carpeta contiene pruebas unitarias del runtime del modelo, adaptador, triggering y tutor, incluyendo el flujo integrado de probabilidades a intervención.

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

A medida que el proyecto madure podrán agregarse pruebas para:

- filtros;
- segmentación;
- extracción de características;
- dimensiones de arrays;
- validación de inputs;
- triggering;
- construcción de payloads.

No crear tests falsos únicamente para aparentar cobertura.

No agregar datasets reales, modelos entrenados ni resultados inventados. Las pruebas se escribirán cuando exista comportamiento implementado que verificar.
