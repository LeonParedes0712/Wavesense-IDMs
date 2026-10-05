"""Opt-in desktop alert, independent of OpenAI and browser navigation."""


def show_visual_alert(text: str) -> None:
    """Display plain text when explicitly called; requires Tk and a desktop."""
    from tkinter import messagebox

    messagebox.showinfo(title="Wavesense — ayuda", message=text)
