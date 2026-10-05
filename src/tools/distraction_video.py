"""Explicit, application-controlled video alert; safe to import."""

import os
import webbrowser


def open_distraction_video() -> bool:
    """Open the locally configured video in the default browser on request.

    Return the browser launch result. The application, never the tutor or
    the LLM, decides whether to call this after receiving tutor text.
    """
    url = os.environ.get("DISTRACTION_VIDEO_URL", "").strip()
    if not url:
        raise RuntimeError(
            "Configura DISTRACTION_VIDEO_URL localmente para abrir el video."
        )
    return webbrowser.open(url)
