# Source Code

> Estado inicial: contiene documentación y plantillas sin lógica real ni resultados. Las funcionalidades descritas son trabajo futuro.

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

Posibles reglas:

- threshold de probabilidad;
- persistencia durante varias ventanas;
- cooldown entre triggers.

## tutor.py

Conecta el estado detectado con la lógica del tutor adaptativo.

Posibles responsabilidades:

- construir payloads;
- seleccionar acciones pedagógicas;
- preparar información para un LLM;
- acciones como `CONTINUE`, `SIMPLIFY`, `SHOW_EXAMPLE` o `BREAK_PROBLEM`.

## Important

No agregar código experimental desorganizado aquí.

Los experimentos iniciales deben realizarse primero en `notebooks/`.

Los módulos anteriores contienen únicamente docstrings y comentarios TODO. No guardar aquí datos, modelos entrenados, credenciales ni resultados de experimentos.
