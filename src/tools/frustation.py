# src/tools/frustration.py

def offer_academic_help(custom_question: str = None) -> str:
    """
    Notifica al sistema que se alcanzó el umbral de 3 distracciones.
    Devuelve la pregunta abierta redactada por ChatGPT para averiguar qué estudia el usuario.
    """
    return custom_question or "¿En qué tema estás trabajando en este momento y qué parte te está causando confusión?"

FRUSTRATION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "offer_academic_help",
        "description": "Llamar obligatoriamente cuando el estudiante acumule 3 o más distracciones seguidas.",
        "parameters": {
            "type": "object",
            "properties": {
                "custom_question": {
                    "type": "string",
                    "description": "Pregunta empática redactada por el tutor preguntando al usuario qué tema está estudiando actualmente y en qué tiene dudas."
                }
            },
            "required": ["custom_question"]
        }
    }
}