# System Architecture

La arquitectura inicial de Wavesense-IDMs sigue este flujo:

EEG acquisition
→ preprocessing
→ signal quality / artifact detection
→ feature extraction
→ cognitive-state classifier
→ temporal decision logic
→ trigger
→ adaptive tutor

## EEG Acquisition

Recibe las señales obtenidas del dispositivo EEG.

## Preprocessing

Prepara la señal mediante operaciones como filtrado, normalización y segmentación.

## Artifact / Signal Quality

Busca identificar actividad que no represente correctamente actividad cerebral útil para el modelo, por ejemplo movimientos o interferencia muscular.

## Feature Extraction

Convierte las ventanas EEG en variables utilizables por los modelos.

Inicialmente pueden usarse características temporales, espectrales y de conectividad.

## Cognitive-State Classifier

Modelo encargado de estimar probabilidades de estados como:

- REST
- LOW_LOAD
- HIGH_LOAD
- ARTIFACT

## Temporal Decision Logic

Una predicción aislada no debe activar automáticamente una intervención.

El sistema podrá considerar:

- confianza;
- persistencia;
- ventanas consecutivas;
- cooldown.

## Adaptive Tutor

Cuando se produzca un trigger válido, el sistema podrá cambiar la estrategia pedagógica o enviar contexto estructurado a un LLM.
