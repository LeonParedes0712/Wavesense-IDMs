"""Explicit, application-controlled video alert; safe to import."""

import webbrowser

TIKTOK_VIDEO_URL = "https://www.tiktok.com/@ingenierossiningenio/video/7693213355208609044"


def open_distraction_video() -> bool:
    """Open the project's fixed video in the default browser on request.

    Return the browser launch result. The application, never the tutor or
    the LLM, decides whether to call this after receiving tutor text.
    """
    return webbrowser.open(TIKTOK_VIDEO_URL)
