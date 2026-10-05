# Data Collection Protocol

Este documento describe cómo se recolectan y etiquetan los datos utilizados por Wavesense-IDMs.

## Initial Experimental Labels

Las clases iniciales consideradas para el MVP son:

- REST
- LOW_LOAD
- HIGH_LOAD
- ARTIFACT

Estas etiquetas pueden cambiar conforme avance el proyecto.

## REST

Ejemplo de contexto:

- reposo en estado de vigilia;
- ojos abiertos o cerrados según el protocolo.

## LOW_LOAD

Tareas cognitivas de baja dificultad relativa.

## HIGH_LOAD

Tareas cognitivas de mayor dificultad relativa.

## ARTIFACT

Sesiones o segmentos utilizados para identificar interferencias.

Ejemplos:

- parpadeo;
- movimiento de mandíbula;
- movimiento corporal;
- actividad muscular intencional.

## Important Scientific Note

La dificultad de una tarea no debe interpretarse automáticamente como una medición directa de frustración, fatiga o sobrecarga psicológica.

Las etiquetas utilizadas por el modelo deben estar definidas por el protocolo experimental y documentadas claramente.

## Metadata

Cuando sea posible, registrar:

- participant/session ID;
- tarea;
- condición;
- timestamp;
- duración;
- etiqueta;
- observaciones;
- calidad de señal.

No guardar información personal innecesaria dentro del repositorio.
