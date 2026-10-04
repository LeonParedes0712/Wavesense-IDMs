# Data

> Estado inicial: esta carpeta contiene únicamente documentación. Los archivos, datos y funcionalidades descritos son trabajo futuro.

Esta carpeta contiene los datos utilizados por el proyecto Wavesense-IDMs.

## raw/

Contiene las grabaciones EEG originales obtenidas directamente del dispositivo o del sistema de adquisición.

Los archivos en esta carpeta:

- no deben modificarse manualmente;
- deben conservarse como referencia del experimento original;
- no deben subirse a GitHub por defecto.

Ejemplos de información que podría encontrarse aquí:

- señales EEG sin procesar;
- timestamps;
- metadatos de sesión;
- etiquetas experimentales originales.

## processed/

Contiene datos derivados de `raw/` después de aplicar procesamiento.

Ejemplos:

- señales filtradas;
- ventanas temporales;
- matrices de características;
- datos preparados para Machine Learning;
- etiquetas reorganizadas.

Los archivos de datos reales tampoco deben subirse a GitHub por defecto.

## Important

Nunca guardar API keys, información privada del participante o credenciales dentro de esta carpeta.
