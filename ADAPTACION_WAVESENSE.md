# Adaptación del Modelo EEG a Estados Wavesense

Este documento describe la normalización de salidas del modelo EEG hacia los 4 estados estandarizados del protocolo Wavesense-IDMs: `REST`, `LOW_LOAD`, `HIGH_LOAD` y `ARTIFACT`.

## Mapeo Oficial de Clases

| Clase Original del Modelo | Estado Wavesense | Descripción |
| --- | --- | --- |
| `REST` / `DEMOSTRACION_LUZMA` | `REST` | Reposo despierto o línea base de calibración. |
| `DIBUJANDO` | `LOW_LOAD` | Tarea motora/visual de baja demanda cognitiva. |
| `MATH_LOAD` | `HIGH_LOAD` | Tarea de esfuerzo mental explícito. |
| `BRAWL_STARS` | `HIGH_LOAD` | Tarea de alta demanda atencional/reactiva. |
| `SUBWAY_SURFERS` | `HIGH_LOAD` | Tarea de exigencia cognitiva (sujeto a calidad de señal). |

## Reglas de Procesamiento

1. **Entrada Directa de Probabilidades:** Las probabilidades provienen de `dict(zip(modelo.classes_, modelo.predict_proba(features)[0]))`. Se procesan mediante suma directa sin aplicar funciones Softmax adicionales.
2. **Control de Calidad (`ARTIFACT`):** El estado `ARTIFACT` se activa mediante el parámetro `signal_has_artifact=True`, proveniente de un control de calidad previo. Si se detecta un artefacto severo, se retorna `ARTIFACT: 1.0` bloqueando decisiones de tutoría en `triggering.py`.
3. **Acceso por Nombres:** El adaptador lee las probabilidades usando llaves de diccionario (`.get()`), asegurando compatibilidad aunque el orden de las clases cambie en futuros reentrenamientos.