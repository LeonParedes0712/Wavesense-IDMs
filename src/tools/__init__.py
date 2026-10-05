# src/tools/__init__.py

from .tools import trigger_distraction_alert, TIKTOK_TOOL_SCHEMA
from .frustation import offer_academic_help, FRUSTRATION_TOOL_SCHEMA

# Lista maestra de herramientas para la API de OpenAI
ALL_TOOLS = [
    TIKTOK_TOOL_SCHEMA,
    FRUSTRATION_TOOL_SCHEMA
]

# Mapeo para ejecutar la función correspondiente
TOOL_MAP = {
    "trigger_distraction_alert": trigger_distraction_alert,
    "offer_academic_help": offer_academic_help
}