# Wavesense-IDMs

## Project Overview

### Sistema de Tutoría Neuro-Adaptativa Basado en Umbrales de Carga Cognitiva

El proyecto consiste en un sistema de aprendizaje inteligente que evalúa la actividad cerebral del usuario mediante señales de EEG procesadas en tiempo real. Utilizando un modelo clasificador local ligero (basado en Redes Neuronales de Grafos o Machine Learning), el sistema monitorea estados cognitivos clave (como enfoque, frustración, fatiga o sobrecarga) entrenados a partir de diversos contextos de prueba (resolver matemáticas, jugar o tocar música).

Para maximizar la eficiencia y reducir drásticamente el consumo de tokens y costos de API, el modelo local actúa como un "guardián": solo cuando la carga mental del estudiante cruza un umbral crítico determinado, el sistema dispara una llamada a la API de ChatGPT. El LLM recibe el estado fisiológico actual como contexto y adapta dinámicamente la estrategia pedagógica (explicaciones más simples, cambio de tema o pausas sugeridas) exactamente cuando el usuario lo necesita.

La descripción anterior expresa la visión del proyecto. Ya existen el adaptador de probabilidades, las reglas temporales y una integración del tutor con OpenAI, con pruebas sin servicios externos. La adquisición y el procesamiento reutilizables siguen pendientes; no hay modelos entrenados validados en este repositorio. El MVP comenzará con modelos clásicos como Logistic Regression, SVM o Random Forest. Una GNN queda como exploración posterior.

## System Architecture

```text
EEG acquisition
→ preprocessing
→ artifact / signal quality detection
→ feature extraction
→ cognitive-state classification
→ temporal decision logic
→ trigger
→ adaptive tutor
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
| `tests/` | Pruebas del adaptador, triggering y tutor, sin llamadas externas. |

Cada carpeta incluye su README. Los módulos de adquisición, procesamiento y modelos aún contienen plantillas. Los documentos técnicos se encuentran en `docs/`.

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

Todavía no hay un pipeline ejecutable. Para preparar el entorno de desarrollo, crear un entorno con `python -m venv .venv` (o `python3`, según la instalación). Se activa con `source .venv/bin/activate` en Linux/macOS, `.venv\Scripts\Activate.ps1` en PowerShell o `.venv\Scripts\activate.bat` en Windows CMD.

Con el entorno activo, ejecutar `pip install -r requirements.txt`. Las dependencias incluyen NumPy, pandas, SciPy, scikit-learn, Matplotlib, MNE, NetworkX, Jupyter, python-dotenv y el SDK de OpenAI.

Para habilitar llamadas reales del tutor, configurar `OPENAI_API_KEY` en el entorno o en un `.env` local ignorado por Git, y elegir explícitamente el modelo al crear `Tutor`. `.env.example` contiene solo un placeholder vacío. Importar el tutor o ejecutar `python -m src.tutor` no llama a la API ni abre ventanas.

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

Etapa inicial de un prototipo de hackathon. El flujo adaptador → triggering → tutor tiene implementación y pruebas unitarias; el pipeline EEG completo sigue pendiente de integración y validación. También quedan pendientes la confirmación del hardware, el protocolo y las etiquetas. No hay resultados experimentales ni rendimiento demostrado.
