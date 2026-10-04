# Wavesense-IDMs

## Project Overview

### Sistema de Tutoría Neuro-Adaptativa Basado en Umbrales de Carga Cognitiva

El proyecto consiste en un sistema de aprendizaje inteligente que evalúa la actividad cerebral del usuario mediante señales de EEG procesadas en tiempo real. Utilizando un modelo clasificador local ligero (basado en Redes Neuronales de Grafos o Machine Learning), el sistema monitorea estados cognitivos clave (como enfoque, frustración, fatiga o sobrecarga) entrenados a partir de diversos contextos de prueba (resolver matemáticas, jugar o tocar música).

Para maximizar la eficiencia y reducir drásticamente el consumo de tokens y costos de API, el modelo local actúa como un "guardián": solo cuando la carga mental del estudiante cruza un umbral crítico determinado, el sistema dispara una llamada a la API de ChatGPT. El LLM recibe el estado fisiológico actual como contexto y adapta dinámicamente la estrategia pedagógica (explicaciones más simples, cambio de tema o pausas sugeridas) exactamente cuando el usuario lo necesita.

La descripción anterior expresa la visión del proyecto. En esta etapa solo existe la estructura documental del prototipo de hackathon; todavía no hay adquisición, procesamiento, modelos entrenados ni integración con un LLM. El MVP comenzará con modelos clásicos como Logistic Regression, SVM o Random Forest. Una GNN queda como exploración posterior.

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

La decisión temporal considerará confianza, persistencia durante varias ventanas y un intervalo mínimo entre intervenciones (cooldown). Una predicción aislada no debería activar automáticamente al tutor. Solo un trigger válido enviará contexto al LLM.

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
| `tests/` | Futuras pruebas del código reutilizable. |

Cada carpeta incluye su README. Por alcance de esta entrega, no se crean módulos Python, notebooks, archivos de dependencias ni otros archivos de configuración. Los documentos técnicos previstos se describen en `docs/README.md`.

## Initial Classification Goal

- `REST`: reposo en vigilia según el protocolo.
- `LOW_LOAD`: tareas de menor carga cognitiva relativa.
- `HIGH_LOAD`: tareas de mayor carga cognitiva relativa.
- `ARTIFACT`: segmentos con interferencias o calidad inadecuada.

Las etiquetas finales dependerán del protocolo experimental y de la calidad de los datos. La dificultad de una tarea no constituye por sí sola una medida de frustración, fatiga o sobrecarga psicológica.

## Installation

Para obtener esta estructura documental:

```bash
git clone --branch chore/project-structure https://github.com/LeonParedes0712/Wavesense-IDMs.git
cd Wavesense-IDMs
```

Todavía no hay código que ejecutar ni dependencias que instalar. Cuando comience el desarrollo, se podrá crear un entorno con `python -m venv .venv` (o `python3`, según la instalación). Se activa con `source .venv/bin/activate` en Linux/macOS, `.venv\Scripts\Activate.ps1` en PowerShell o `.venv\Scripts\activate.bat` en Windows CMD.

Las dependencias propuestas para una futura etapa son NumPy, pandas, SciPy, scikit-learn, Matplotlib, MNE, NetworkX y Jupyter; se añadirán cuando sean necesarias. No existe todavía `requirements.txt`.

## Development Workflow

Esta entrega se publica en `chore/project-structure`, sin merge automático. La rama personal `leon/python` sigue disponible.

Después de que el equipo integre la estructura en `main`, cada integrante creará su rama independiente:

```bash
git checkout main
git pull origin main
git checkout -b feature/preprocessing
```

Otras ramas posibles son `feature/features`, `feature/models`, `feature/triggering`, `feature/tutor` y `feature/data-analysis`.

Mientras la estructura no esté integrada en `main`, se puede crear una rama desde esta entrega:

```bash
git fetch origin
git checkout -b feature/preprocessing origin/chore/project-structure
```

Al terminar un cambio, revisar los archivos y añadir únicamente los que correspondan:

```bash
git status
git add <archivos-del-cambio>
git commit -m "descripcion del cambio"
git push -u origin feature/preprocessing
```

Después de disponer de `main` como rama de integración, abrir un Pull Request hacia ella. No usar `main` para experimentación directa.

## Data Handling

Los datos EEG originales y procesados, modelos entrenados y resultados generados no deben subirse por defecto. No incluir datos personales, API keys, tokens, credenciales ni entornos virtuales.

Esta entrega contiene exclusivamente archivos README y no incluye `.gitignore`. Antes de incorporar datos locales o comenzar el desarrollo, configurar las exclusiones de Git para datasets, modelos, resultados, secretos, cachés y entornos virtuales. Los README documentan estas reglas, pero no impiden por sí mismos que Git agregue archivos.

## Project Status

Etapa inicial de un prototipo de hackathon. Solo se ha preparado la estructura de carpetas y su documentación. Quedan pendientes la confirmación del hardware, el protocolo y las etiquetas, así como la implementación y validación del pipeline. No hay resultados experimentales ni rendimiento demostrado.
