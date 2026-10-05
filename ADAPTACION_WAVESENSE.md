# Guía de adaptación de salida del modelo a estados Wavesense

## 1. Resumen del dataset y del modelo

- **Dataset utilizado:** Registros EEG (Unicorn) recolectados durante 5 condiciones experimentales (`datos_demostracion.csv`, `datos_dibujando.csv`, `datos_fracciones.csv`, `datos_subway.csv`, `datos_Brawlstars.csv`).
- **Etiquetas originales (5 clases):**
  - `'REST'`: Período de reposo / línea base.
  - `'DIBUJANDO'`: Tarea de dibujo / demanda cognitiva baja.
  - `'MATH_LOAD'`: Tarea de resolución de ejercicios matemáticos.
  - `'BRAWL_STARS'`: Videojuego de alta atención y procesamiento de estímulos.
  - `'SUBWAY_SURFERS'`: Videojuego de reacción continua y alta demanda visual/motora.
- **Modelo:** Creado con `modelo_eeg_unicorn.pkl` y normalizado con `escalador_eeg.pkl`. Retorna un vector con 5 probabilidades mediante el método `predict_proba()`.

---

## 2. Tabla de mapeo

| Etiqueta o salida original | Clase Wavesense | Justificación experimental | Limitación |
|---|---|---|---|
| `'REST'` | `REST` | Período de reposo/línea base en vigilia sin tarea cognitiva activa. | No implica relajación clínica ni estado de sueño. |
| `'DIBUJANDO'` | `LOW_LOAD` | Tarea motora/creativa de demanda mental moderada o baja dentro del protocolo. | La carga percibida puede variar según la habilidad artística del sujeto. |
| `'MATH_LOAD'`, `'BRAWL_STARS'`, `'SUBWAY_SURFERS'` | `HIGH_LOAD` | Tareas que requieren alta concentración, cálculo mental o procesamiento continuo visual/motor. | No demuestra frustración, estrés, ansiedad ni ninguna condición clínica. |
| *No disponible en dataset* | `ARTIFACT` | El dataset no cuenta con etiquetas específicas de artefactos (ruido eléctrico, parpadeos o movimientos). | Se fija temporalmente en `0.0`. Requiere un filtro de calidad de señal previo. |

---

## 3. Función de adaptación

```python
import numpy as np

def to_wavesense_probabilities(model_output) -> dict[str, float]:
    """
    Convierte la salida de 5 clases del modelo local (modelo_eeg_unicorn.pkl) 
    a las 4 probabilidades estandarizadas de Wavesense.
    
    Formatos aceptados:
    - Lista/Array NumPy de 5 probabilidades en orden alfabético de clases:
      [BRAWL_STARS, DIBUJANDO, MATH_LOAD, REST, SUBWAY_SURFERS]
    - Diccionario con las clases originales como llaves.
    """
    if isinstance(model_output, dict):
        p_brawl = float(model_output.get("BRAWL_STARS", 0.0))
        p_dibujando = float(model_output.get("DIBUJANDO", 0.0))
        p_math = float(model_output.get("MATH_LOAD", 0.0))
        p_rest_orig = float(model_output.get("REST", 0.0))
        p_subway = float(model_output.get("SUBWAY_SURFERS", 0.0))
    elif isinstance(model_output, (list, tuple, np.ndarray)) and len(model_output) == 5:
        p_brawl, p_dibujando, p_math, p_rest_orig, p_subway = map(float, model_output)
    else:
        raise ValueError("La salida del modelo debe ser un vector de 5 probabilidades o un diccionario.")

    # Agregación según el mapeo experimental
    p_rest = p_rest_orig
    p_low = p_dibujando
    p_high = p_math + p_brawl + p_subway
    p_artifact = 0.0

    # Normalización para garantizar que la suma sea exactamente 1.0
    total = p_rest + p_low + p_high + p_artifact
    if total > 0:
        p_rest /= total
        p_low /= total
        p_high /= total

    return {
        "REST": round(p_rest, 4),
        "LOW_LOAD": round(p_low, 4),
        "HIGH_LOAD": round(p_high, 4),
        "ARTIFACT": round(p_artifact, 4),
    }

## 4. Ejemplo de uso

```python
import joblib

# Cargar el modelo guardado
modelo = joblib.load("modelo_eeg_unicorn.pkl")

# Salida simulada de predict_proba para las 5 clases:
# Orden: [BRAWL_STARS: 0.10, DIBUJANDO: 0.05, MATH_LOAD: 0.70, REST: 0.05, SUBWAY_SURFERS: 0.10]
raw_probs = [0.10, 0.05, 0.70, 0.05, 0.10]

wavesense_dict = to_wavesense_probabilities(raw_probs)
print(wavesense_dict)
# Resultado: {'REST': 0.05, 'LOW_LOAD': 0.05, 'HIGH_LOAD': 0.9, 'ARTIFACT': 0.0}