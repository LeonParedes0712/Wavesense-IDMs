# src/tools/distraction.py

import webbrowser

# URL limpia sin parámetros de rastreo
TIKTOK_VIDEO_URL = "https://www.tiktok.com/@ingenierossiningenio/video/7693213355208609044"

def trigger_distraction_alert(tiktok_url: str = None) -> str:
    """Abre el video específico de TikTok en el navegador predeterminado."""
    target_url = TIKTOK_VIDEO_URL

    print(f"[DEBUG]: Abriendo enlace limpio: {target_url}")

    try:
        webbrowser.open(target_url, new=2)
        return f"Alerta de TikTok abierta exitosamente: {target_url}"
    except Exception as e:
        return f"Error al abrir el navegador: {str(e)}"

TIKTOK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "trigger_distraction_alert",
        "description": "Abre el video de TikTok configurado para llamar la atención del estudiante.",
        "parameters": {
            "type": "object",
            "properties": {
                "tiktok_url": {
                    "type": "string",
                    "description": "URL del video."
                }
            },
            "required": []
        }
    }
}