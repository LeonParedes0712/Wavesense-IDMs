# Documentation

> Estado inicial: documentación conceptual; las especificaciones de hardware y el protocolo siguen pendientes de confirmación.

Esta carpeta contiene documentación técnica y experimental del proyecto.

## architecture.md

Describe la arquitectura del sistema y cómo fluye la información desde la señal EEG hasta la posible intervención del tutor.

## hardware.md

Documenta las características reales del dispositivo EEG utilizado.

Debe incluir solamente información confirmada.

## data_protocol.md

Describe cómo se obtienen los datos y cómo se generan las etiquetas experimentales.

Esta documentación es importante para poder interpretar correctamente los resultados del modelo.

No agregar aquí código, datasets, credenciales ni especificaciones o resultados sin confirmar. Los documentos `architecture.md`, `hardware.md` y `data_protocol.md` contienen las bases que también se conservan a continuación.

## System Architecture

La arquitectura inicial de Wavesense-IDMs sigue este flujo:

EEG acquisition
→ preprocessing
→ signal quality / artifact detection
→ feature extraction
→ cognitive-state classifier
→ temporal decision logic
→ trigger
→ adaptive tutor

### EEG Acquisition

Recibe las señales obtenidas del dispositivo EEG.

### Preprocessing

Prepara la señal mediante operaciones como filtrado, normalización y segmentación.

### Artifact / Signal Quality

Busca identificar actividad que no represente correctamente actividad cerebral útil para el modelo, por ejemplo movimientos o interferencia muscular.

### Feature Extraction

Convierte las ventanas EEG en variables utilizables por los modelos.

Inicialmente pueden usarse características temporales, espectrales y de conectividad.

### Cognitive-State Classifier

Modelo encargado de estimar probabilidades de estados como:

- REST
- LOW_LOAD
- HIGH_LOAD
- ARTIFACT

### Temporal Decision Logic

Una predicción aislada no debe activar automáticamente una intervención.

El sistema podrá considerar:

- confianza;
- persistencia;
- ventanas consecutivas;
- cooldown.

### Adaptive Tutor

Cuando se produzca un trigger válido, el sistema podrá cambiar la estrategia pedagógica o enviar contexto estructurado a un LLM.

## EEG Hardware

Este documento debe registrar las especificaciones reales del dispositivo EEG usado por Wavesense-IDMs.

No inventar datos que todavía no hayan sido confirmados.

### Device

- Device name:
- Manufacturer:
- Model:

### Acquisition

- Number of EEG channels:
- Electrode locations:
- Sampling rate:
- Reference electrode:
- Signal resolution:
- Streaming interface:
- Exported file format:

### Notes

Agregar aquí cualquier limitación conocida del hardware, problemas de conexión, calidad de señal o consideraciones relevantes para el procesamiento.

## Data Collection Protocol

Este documento describe cómo se recolectan y etiquetan los datos utilizados por Wavesense-IDMs.

### Initial Experimental Labels

Las clases iniciales consideradas para el MVP son:

- REST
- LOW_LOAD
- HIGH_LOAD
- ARTIFACT

Estas etiquetas pueden cambiar conforme avance el proyecto.

### REST

Ejemplo de contexto:

- reposo en estado de vigilia;
- ojos abiertos o cerrados según el protocolo.

### LOW_LOAD

Tareas cognitivas de baja dificultad relativa.

### HIGH_LOAD

Tareas cognitivas de mayor dificultad relativa.

### ARTIFACT

Sesiones o segmentos utilizados para identificar interferencias.

Ejemplos:

- parpadeo;
- movimiento de mandíbula;
- movimiento corporal;
- actividad muscular intencional.

### Important Scientific Note

La dificultad de una tarea no debe interpretarse automáticamente como una medición directa de frustración, fatiga o sobrecarga psicológica.

Las etiquetas utilizadas por el modelo deben estar definidas por el protocolo experimental y documentadas claramente.

### Metadata

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
