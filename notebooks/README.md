# Notebooks

> Estado inicial: esta carpeta contiene únicamente documentación. Los archivos, datos y funcionalidades descritos son trabajo futuro.

Esta carpeta se utiliza para experimentación, análisis exploratorio y pruebas rápidas.

Los notebooks NO deben convertirse en la única fuente del código importante del proyecto. Cuando una función sea estable y reutilizable, debe moverse a `src/`.

## 01_exploration.ipynb

Exploración inicial de los datos EEG.

Posibles tareas:

- revisar dimensiones;
- identificar canales;
- verificar frecuencia de muestreo;
- revisar duración;
- detectar valores faltantes;
- visualizar señales en el dominio temporal;
- calcular PSD.

## 02_preprocessing.ipynb

Experimentación con el preprocesamiento.

Posibles tareas:

- notch filtering;
- band-pass filtering;
- normalización;
- segmentación en ventanas;
- inspección de artefactos.

## 03_features.ipynb

Experimentación con extracción de características.

Ejemplos:

- delta power;
- theta power;
- alpha power;
- beta power;
- gamma power;
- ratios entre bandas;
- RMS;
- varianza;
- entropía espectral;
- coherencia;
- PLV.

## 04_models.ipynb

Pruebas de modelos baseline.

Ejemplos:

- Logistic Regression;
- SVM;
- Random Forest;
- comparación de métricas;
- matriz de confusión;
- validación.

No guardar modelos finales directamente dentro de los notebooks.

No incorporar credenciales ni datos personales en celdas o salidas. Los nombres de notebooks anteriores son una propuesta para la siguiente etapa; aún no se han creado.
