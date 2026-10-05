# Source Code

> El adaptador, triggering y tutor están implementados. Preprocesamiento, características y modelos siguen siendo plantillas; no hay resultados experimentales.

Esta carpeta contiene el código reutilizable y estable del proyecto.

El objetivo es mover aquí las funciones que ya hayan sido probadas en los notebooks y que formen parte del pipeline real.

## preprocessing.py

Funciones relacionadas con limpieza y preparación de señales EEG.

Ejemplos:

- filtros;
- segmentación;
- normalización;
- detección o manejo de artefactos.

## features.py

Funciones para extraer características de las señales.

Ejemplos:

- band power;
- características temporales;
- características espectrales;
- ratios;
- conectividad como PLV o coherencia.

## models.py

Código relacionado con los modelos de clasificación.

Ejemplos:

- creación de modelos;
- entrenamiento;
- evaluación;
- inferencia;
- probabilidades de clase.

## triggering.py

Implementa la lógica que decide cuándo una predicción debe producir una intervención.

Reglas implementadas:

- threshold de probabilidad;
- persistencia durante varias ventanas;
- bloqueo por artefactos;
- cooldown entre triggers.

## tutor.py

`Tutor.respond(decision)` recibe un `TriggerDecision` de `src.triggering`. Devuelve `None` sin llamar a OpenAI ni ejecutar herramientas cuando `triggered=False` o el estado no es `HIGH_LOAD`; `ARTIFACT` nunca es intervenible. Si la decisión es válida, devuelve texto generado con estado y confianza como único contexto del modelo local.

La política inicial definida en Python es ofrecer ayuda breve para trabajar en pasos pequeños. OpenAI solo redacta el texto: no selecciona acciones ni ejecuta herramientas. No hay escalado automático. El equipo debe definir los niveles de ayuda, cuándo avanzar o reiniciarlos y cómo incorporar contexto de la tarea antes de añadir una progresión pedagógica.

Ejemplo de conexión para que la aplicación lo invoque con cada salida del modelo local:

```python
from dotenv import load_dotenv

from src.model_adapter import to_wavesense_probabilities
from src.triggering import TemporalTrigger
from src.tutor import Tutor

load_dotenv()  # Opcional: cargar el .env local de la aplicación.
trigger = TemporalTrigger()  # Conservar la instancia durante la sesión.
tutor = Tutor(model="<modelo habilitado por el equipo>")

def process_prediction(model_output, signal_has_artifact=False):
    probabilities = to_wavesense_probabilities(model_output, signal_has_artifact)
    decision = trigger.evaluate(probabilities)
    return tutor.respond(decision)
```

Entregar cada decisión una sola vez al tutor. No construir decisiones manualmente ni recrear `TemporalTrigger` por ventana, pues perdería persistencia y cooldown. La API key se lee exclusivamente de `OPENAI_API_KEY`, solo al necesitar una intervención. Si falta o está vacía, se lanza `RuntimeError` con un mensaje de configuración sin secretos. Los errores de API se propagan a la aplicación sin reintentos automáticos ni alertas; el cooldown ya aplicado por triggering se conserva. Un cliente inyectado mediante `client=` permite pruebas sin credenciales y su ciclo de vida corresponde al llamador.

Se utiliza `responses.create` del [SDK oficial de OpenAI](https://developers.openai.com/api/reference/python), sin herramientas del LLM. Importar o ejecutar `tutor.py` no inicia simulaciones, clientes ni ventanas.

## tools/visual_alert.py

`show_visual_alert(text)` muestra texto en un diálogo local y requiere Tk y una sesión gráfica. Está aislada y desactivada por defecto. La aplicación puede habilitarla explícitamente con `Tutor(model=..., visual_alert=show_visual_alert)`; se ejecuta únicamente después de obtener ayuda para un trigger válido. No abre el navegador. Las pruebas sustituyen la interfaz gráfica por mocks.

## Important

No agregar código experimental desorganizado aquí.

Los experimentos iniciales deben realizarse primero en `notebooks/`.

No guardar aquí datos, modelos entrenados, credenciales ni resultados de experimentos.
