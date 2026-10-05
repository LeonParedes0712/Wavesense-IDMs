# Source Code

> El runtime del modelo, adaptador, triggering y tutor están implementados. Preprocesamiento y extracción de características siguen siendo plantillas; el runtime no valida el rendimiento experimental del modelo.

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

`EEGModelRuntime` carga una vez el modelo local y el escalador opcional mediante
`EEGModelRuntime.from_env()`. Usa `joblib.load`, compatible con los artefactos
locales inspeccionados. `EEG_MODEL_PATH` es obligatoria; `EEG_SCALER_PATH` vacía
omite el escalador. Si se configura una ruta inexistente o ilegible, falla sin
continuar silenciosamente. Las rutas relativas parten del directorio de trabajo;
se admite `~`. La aplicación carga `.env` si lo necesita. Importar el módulo no
carga artefactos ni inicia servicios. Cargar solo archivos pickle/joblib locales
de confianza: su deserialización puede ejecutar código.

### Contrato de entrada y salida

- `predict_proba(features, *, feature_names=None)` recibe un vector `(p,)` o una
  matriz `(1, p)` de números reales, finitos, no vacía. Devuelve
  `dict[str, float]`, con las etiquetas originales y el orden de `model.classes_`.
- `predict_proba_batch(features, *, feature_names=None)` acepta `(n, p)` y devuelve
  una lista de esos diccionarios, uno por fila y en el mismo orden. También admite
  un vector como lote de una fila. El método individual rechaza varias filas para
  evitar descartar ventanas o promediar probabilidades implícitamente.
- Se acepta `pandas.DataFrame` con columnas exactas o arrays/listas acompañados de
  `feature_names`, que declara el orden real de sus valores. No se reordenan
  columnas automáticamente. No se aceptan strings numéricos, booleanos, complejos,
  NaN/inf, variables adicionales ni nombres duplicados.
- `runtime.feature_names` es la tupla ordenada del contrato y `runtime.n_features`
  es su dimensión. Se obtienen de `feature_names_in_` del modelo/escalador. Si
  ambos lo tienen, deben coincidir. También se valida `n_features_in_`.
  Si faltan nombres en ambos artefactos, se exige
  `from_env(feature_names=esquema_confirmado_por_entrenamiento)` (o el mismo
  argumento al constructor). La dimensión sola no identifica características.
  El error `ModelContractError` explica qué falta; no se inventa un esquema.
- El escalador ejecuta `transform` antes de `model.predict_proba`; nunca `fit`.
  Debe conservar filas, variables y orden. Se entregan DataFrames a estimadores
  que guardan nombres. Las probabilidades deben tener una columna por clase,
  ser finitas, estar en `[0, 1]` y sumar aproximadamente uno por fila.
  No se aplica softmax, normalización ni redondeo en el runtime.
- Cada diccionario se pasa directamente a
  `to_wavesense_probabilities(output, signal_has_artifact=...)`. La calidad de
  señal se determina fuera del runtime. El adaptador conserva la responsabilidad
  de convertir clases a REST / LOW_LOAD / HIGH_LOAD / ARTIFACT.
  Las etiquetas de nuevos modelos deben ser compatibles con el mapeo documentado
  en `ADAPTACION_WAVESENSE.md`; el runtime no renombra clases.

### Características observadas y trabajo pendiente de adquisición

Los archivos locales ignorados `modelo_eeg_unicorn.pkl` (RandomForestClassifier)
y `escalador_eeg.pkl` (StandardScaler) declaran 32 características idénticas:

```python
# Orden exacto observado en feature_names_in_ de ambos artefactos:
observed_names = tuple(
    f"Canal_{channel}_{band}"
    for channel in range(1, 9)
    for band in ("Theta", "Alpha", "Beta", "Gamma")
)
```

Primero las cuatro bandas de Canal_1, luego Canal_2, hasta Canal_8. Las clases
observadas, en orden, son `BRAWL_STARS`, `DIBUJANDO`, `MATH_LOAD`, `REST`,
`SUBWAY_SURFERS`. Estos datos describen los artefactos inspeccionados, no se
codifican como valores obligatorios para futuros modelos.

El notebook `analisis_eeg.ipynb`, idéntico en la referencia local `origin/modelo`,
muestra una extracción candidata: 250 Hz; Butterworth de orden 4 entre 1 y
40 Hz con `filtfilt`; ventanas de hasta 250 muestras; `welch` por canal y suma
de bins PSD con límites inclusivos Theta 4–8, Alpha 8–12, Beta 12–30 y Gamma
30–45 Hz. Selecciona canales desde las columnas CSV y contiene otros análisis
con ventanas de 10 segundos. Su StandardScaler se presenta como normalización
para visualización; no contiene el entrenamiento/exportación que vincule
inequívocamente esa receta con ambos `.pkl`.

Por tanto, **se conoce el esquema de 32 columnas, pero falta confirmar la receta
de entrenamiento**: correspondencia física y unidades de Canal_1…Canal_8,
ventana y paso usados, parámetros/versiones exactos de Welch y filtros, estado
entre ventanas y si el clasificador fue entrenado con ese escalador. No se debe
sustituir la suma de bins por integrales, ratios o porcentajes de línea base sin
confirmarlo. La presencia de Gamma hasta 45 Hz junto al filtro hasta 40 Hz también
debe reconciliarse con el entrenamiento. Los nombres no permiten resolverlo.

El futuro módulo de adquisición/preprocesamiento/extracción deberá entregar una
fila por ventana, con esas 32 potencias **antes del escalado externo**, una vez
confirmada la receta. Debe adjuntar los nombres reales de extracción o construir
un DataFrame en ese orden; copiar `runtime.feature_names` como etiqueta de valores
desconocidos no valida su significado. El indicador de artefacto viajará separado
al adaptador. Este módulo no adquiere Unicorn, filtra, extrae características,
decide triggers ni llama al tutor/OpenAI.

### Uso local

Desde la raíz, con las dependencias de `requirements.txt` instaladas y los
artefactos de confianza ya presentes (no se versionan):

```bash
export EEG_MODEL_PATH=./modelo_eeg_unicorn.pkl
export EEG_SCALER_PATH=./escalador_eeg.pkl
python -c 'from src.models import EEGModelRuntime; r = EEGModelRuntime.from_env(); print(r.feature_names); print(r.classes)'
```

Inferencia sobre un CSV **local** de características ya calculadas, con encabezados
exactos y sin columna de etiqueta, índice ni timestamp:

```bash
python - <<'PYCODE'
import pandas as pd
from src.models import EEGModelRuntime
from src.model_adapter import to_wavesense_probabilities

runtime = EEGModelRuntime.from_env()
features = pd.read_csv("data/processed/ventana_features.csv")  # Una fila real.
output = runtime.predict_proba(features)
print(output)
print(to_wavesense_probabilities(output, signal_has_artifact=False))
PYCODE
```

Para un array proporcionado por el extractor:
`output = runtime.predict_proba(vector, feature_names=nombres_del_extractor)`.
Para varias ventanas, usar `predict_proba_batch` y aplicar el adaptador a cada
salida con el indicador de calidad correspondiente. Reutilizar la instancia;
no cargar los archivos en cada ventana. El CSV del ejemplo debe producirlo el
extractor futuro: no se proporciona un vector inventado ni se ejecuta el notebook.

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

## tools/distraction_video.py

`open_distraction_video()` abre el video configurado en `DISTRACTION_VIDEO_URL`
en el navegador predeterminado. El equipo debe configurar este enlace no secreto
en el entorno o en su `.env` local, cargado por la aplicación. Si falta o está
vacío, lanza `RuntimeError` sin abrir nada. Devuelve el resultado de
`webbrowser.open` (un booleano que indica si se pudo iniciar el navegador).
Importar la herramienta no abre el navegador.

El tutor no invoca esta herramienta y OpenAI no la recibe ni decide ejecutarla.
La aplicación puede llamarla explícitamente solo después de recibir texto:

```python
from src.tools.distraction_video import open_distraction_video

# decision proviene de la instancia persistente de TemporalTrigger.
text = tutor.respond(decision)
if text:
    # Llamada opcional, habilitada por la aplicación según su política.
    open_distraction_video()
```

No pasar esta función como `visual_alert` del tutor ni registrarla como herramienta
del LLM. La alerta de video es una acción separada y explícita de la aplicación.
Las pruebas simulan el navegador y no abren TikTok.

## Important

No agregar código experimental desorganizado aquí.

Los experimentos iniciales deben realizarse primero en `notebooks/`.

No guardar aquí datos, modelos entrenados, credenciales ni resultados de experimentos.
