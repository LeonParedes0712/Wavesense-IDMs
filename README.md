# Wavesense-IDMs

## Project Overview

### Sistema de Tutoría Neuro-Adaptativa Basado en Umbrales de Carga Cognitiva

El proyecto consiste en un sistema de aprendizaje inteligente que evalúa la actividad cerebral del usuario mediante señales de EEG procesadas en tiempo real. Utilizando un modelo clasificador local ligero (basado en Redes Neuronales de Grafos o Machine Learning), el sistema monitorea estados cognitivos clave (como enfoque, frustración, fatiga o sobrecarga) entrenados a partir de diversos contextos de prueba (resolver matemáticas, jugar o tocar música).

Para maximizar la eficiencia y reducir drásticamente el consumo de tokens y costos de API, el modelo local actúa como un "guardián": solo cuando la carga mental del estudiante cruza un umbral crítico determinado, el sistema dispara una llamada a la API de ChatGPT. El LLM recibe el estado fisiológico actual como contexto y adapta dinámicamente la estrategia pedagógica (explicaciones más simples, cambio de tema o pausas sugeridas) exactamente cuando el usuario lo necesita.

La descripción anterior expresa la visión del proyecto. Hay un pipeline reutilizable desde una ventana EEG hasta el runtime local, adaptador, trigger y tutor opcional, con pruebas simuladas sin servicios externos. La adquisición UnicornPy se carga de forma diferida en Windows. Falta validar el dispositivo físico y confirmar que la extracción reproduce el entrenamiento de los artefactos históricos. **El prototipo no diagnostica condiciones médicas ni psicológicas.** No hay rendimiento experimental demostrado; una GNN queda como exploración posterior.

## System Architecture

```text
UnicornSource (Windows) / ArraySource (simulada)
→ WindowStream (ventana completa)
→ validación y calidad de señal cruda
→ referencia/filtro configurados → 32 potencias por canal y banda
→ EEGModelRuntime (escalador → predict_proba)
→ model_adapter → TemporalTrigger → Tutor opcional

Ventana inválida / artefacto → ARTIFACT → reset de persistencia, sin modelo ni tutor
```

`TemporalTrigger` considera confianza, persistencia durante varias ventanas, artefactos y un intervalo mínimo entre intervenciones (cooldown). El tutor recibe su `TriggerDecision` y solo llama a OpenAI cuando `triggered=True` y el estado es `HIGH_LOAD`. Envía únicamente estado y confianza, sin EEG crudo. Consulta la [interfaz de integración](src/README.md#tutorpy).

## Repository Structure

| Carpeta | Uso previsto |
| --- | --- |
| `data/raw/` | Grabaciones EEG originales locales. |
| `data/processed/` | Señales procesadas y características locales. |
| `notebooks/` | Exploración, preprocesamiento, características y evaluación de modelos. |
| `src/` | Código reutilizable de preprocesamiento, características, clasificación, triggering y tutor. |
| `models/` | Artefactos locales de modelos entrenados. |
| `results/figures/` | Visualizaciones de experimentos. |
| `results/metrics/` | Métricas de evaluación. |
| `docs/` | Arquitectura, especificaciones de hardware y protocolo experimental. |
| `tests/` | Pruebas del pipeline EEG, adquisición simulada, runtime, adaptador, triggering y tutor, sin llamadas externas. |

Cada carpeta incluye su README. El contrato, la configuración y los ejemplos de adquisición y procesamiento están en [src/README.md](src/README.md). Los documentos técnicos se encuentran en `docs/`.

## Initial Classification Goal

- `REST`: reposo en vigilia según el protocolo.
- `LOW_LOAD`: tareas de menor carga cognitiva relativa.
- `HIGH_LOAD`: tareas de mayor carga cognitiva relativa.
- `ARTIFACT`: segmentos con interferencias o calidad inadecuada.

Las etiquetas finales dependerán del protocolo experimental y de la calidad de los datos. La dificultad de una tarea no constituye por sí sola una medida de frustración, fatiga o sobrecarga psicológica.

## Installation

Para obtener la estructura base:

```bash
git clone https://github.com/LeonParedes0712/Wavesense-IDMs.git
cd Wavesense-IDMs
```

Para preparar el entorno de desarrollo, crear un entorno con `python -m venv .venv` (o `python3`, según la instalación). Se activa con `source .venv/bin/activate` en Linux/macOS, `.venv\Scripts\Activate.ps1` en PowerShell o `.venv\Scripts\activate.bat` en Windows CMD.

Con el entorno activo, ejecutar `pip install -r requirements.txt`. Las dependencias incluyen NumPy, pandas, SciPy, scikit-learn, Matplotlib, MNE, NetworkX, Jupyter, python-dotenv y el SDK de OpenAI.

Para habilitar llamadas reales del tutor, configurar `OPENAI_API_KEY` en el entorno o en un `.env` local ignorado por Git, y elegir explícitamente el modelo al crear `Tutor`. `.env.example` contiene solo un placeholder vacío. Importar el tutor o ejecutar `python -m src.tutor` no llama a la API ni abre ventanas.

## Pipeline EEG en Windows y simulación

En PowerShell, desde la raíz (con Python compatible con el SDK de Unicorn):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Copiar solo si aún no existe; no sobrescribir tu configuración local
Copy-Item .env.example .env
python -m unittest discover -s tests -v
```

En Linux, copiar con `cp .env.example .env` si aún no existe. Las pruebas no
necesitan Unicorn instalado, dispositivo, `.pkl`, OpenAI ni navegador:

```bash
python -m unittest discover -s tests -p 'test_eeg_pipeline.py' -v
python -m unittest discover -s tests -p 'test_unicorn_stream.py' -v
```

Para hardware, instalar Unicorn Suite/UnicornPy y su licencia manualmente en
Windows, emparejar el dispositivo y confirmar arquitectura/versiones de Python y
DLL. No se instala UnicornPy mediante `requirements.txt`. En `.env`, configurar:

- `EEG_MODEL_PATH=./models/modelo_eeg_unicorn.pkl` y
  `EEG_SCALER_PATH=./models/escalador_eeg.pkl`, o sus rutas locales reales.
- `UNICORN_PYTHON_PATH`: carpeta `Lib` del SDK solo cuando no sea accesible.
- `UNICORN_SERIAL`: selección explícita si hay más de un dispositivo.
- `UNICORN_EEG_CHANNEL_NAMES`: nombres SDK en el orden físico de Canal_1…Canal_8.

El ejemplo de [una ventana real y cierre seguro](src/README.md#windows-preparación-y-una-ventana-real)
está en `src/README.md`, junto con el ejemplo `ArraySource` en memoria. El pipeline
no arranca al importar ni al ejecutar el módulo; la aplicación llama explícitamente
`EEGPipeline.process_window(...)`. El tutor está deshabilitado por defecto y el
video solo puede invocarse externamente. Los archivos de datos/modelos y `.env`
permanecen locales e ignorados.

Los defaults son **provisionales**: 250 Hz, 8 canales, ventanas de 1 s con paso de
1 s; Theta 4–8, Alpha 8–12, Beta 12–30, Gamma 30–45 Hz; Butterworth de orden 4,
1–40 Hz; suma de bins Welch inclusivos. Son configurables en `.env.example`.
La referencia se conserva (`as_acquired`) y las unidades están sin confirmar.
No se afirma equivalencia con el entrenamiento: faltan unidades, orden físico,
referencia, ventana/paso, parámetros de Welch/filtro y relación modelo-escalador.
El notebook y `master/datos.py` usan recetas diferentes; ver la evidencia y
[configuración completa](src/README.md#eeg_configpy).

Antes del uso real falta verificar en Windows la lectura/cierre del SDK, la
frecuencia y orden de canales, buffer, indicadores de validación/pérdida de
muestras, calibración de calidad y latencia. El tutor síncrono debe separarse de
la lectura en una futura aplicación continua. **La detección básica de artefactos
no certifica calidad clínica ni diagnostica estados psicológicos.**

## Development Workflow

La estructura base se integra desde `chore/project-structure` hacia `main`. La rama personal `leon/python` sigue disponible.

Cada integrante debe crear su rama independiente desde `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/preprocessing
```

Otras ramas posibles son `feature/features`, `feature/models`, `feature/triggering`, `feature/tutor` y `feature/data-analysis`.

Al terminar un cambio, revisar los archivos y añadir únicamente los que correspondan:

```bash
git status
git add <archivos-del-cambio>
git commit -m "descripcion del cambio"
git push -u origin feature/preprocessing
```

Abrir un Pull Request hacia `main`. No usar `main` para experimentación directa.

## Data Handling

Los datos EEG originales y procesados, modelos entrenados y resultados generados no deben subirse por defecto. No incluir datos personales, API keys, tokens, credenciales ni entornos virtuales.

El `.gitignore` excluye datasets, modelos, resultados generados, secretos, cachés y entornos virtuales, conservando los README y `.gitkeep`. Revisar siempre `git status` antes de agregar archivos.

## Project Status

Etapa inicial de un prototipo de hackathon. El pipeline por ventana tiene implementación y pruebas unitarias; la integración física con Unicorn en Windows y la equivalencia con el entrenamiento siguen pendientes de validación. También quedan pendientes el protocolo y las etiquetas. No hay resultados experimentales ni rendimiento demostrado.
