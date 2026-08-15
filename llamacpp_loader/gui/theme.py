"""Modern dark theme for llamacpp-loader GUI.

Applies a Material-ish dark palette + rounded card layout via ttk.Style.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# Palette
BG = "#1e1f24"          # window background
CARD = "#2a2b32"        # panel/card background
CARD_ALT = "#31323b"    # inner panel
FIELD = "#3a3b45"       # entry/spinbox bg
BORDER = "#3f404b"      # subtle border
TEXT = "#e6e6ea"        # primary text
TEXT_DIM = "#9a9ba6"    # secondary text
ACCENT = "#4f8cff"      # primary accent (blue)
ACCENT_HOVER = "#6ba1ff"
GREEN = "#3ddc84"
RED = "#ff5252"
AMBER = "#ffb74d"
CONSOLE_BG = "#17181c"
CONSOLE_FG = "#c8cbd4"
FONT = ("Microsoft YaHei UI", 10)
FONT_SMALL = ("Microsoft YaHei UI", 9)
FONT_MONO = ("Consolas", 10)


def apply(root: tk.Tk) -> None:
    """Configure the ttk style set on *root*."""
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")

    # base
    style.configure(".", background=BG, foreground=TEXT, font=FONT)

    # Card frames (raised look)
    style.configure("Card.TFrame", background=CARD)
    style.configure("Card.TLabelframe", background=CARD, foreground=TEXT,
                    bordercolor=BORDER, relief="flat", borderwidth=1)
    style.configure("Card.TLabelframe.Label", background=CARD, foreground=TEXT_DIM,
                    font=FONT_SMALL)

    # Labels
    style.configure("TLabel", background=BG, foreground=TEXT, font=FONT)
    style.configure("Dim.TLabel", background=BG, foreground=TEXT_DIM, font=FONT_SMALL)
    style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=FONT)
    style.configure("CardDim.TLabel", background=CARD, foreground=TEXT_DIM, font=FONT_SMALL)

    # Inputs
    style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT,
                    insertcolor=TEXT, bordercolor=BORDER, lightcolor=FIELD,
                    darkcolor=FIELD, padding=5)
    style.map("TEntry", bordercolor=[("focus", ACCENT)])
    style.configure("TSpinbox", fieldbackground=FIELD, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD,
                    arrowsize=14, padding=4)
    style.map("TSpinbox", bordercolor=[("focus", ACCENT)])
    style.configure("TCombobox", fieldbackground=FIELD, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD,
                    arrowcolor=TEXT_DIM, padding=4)
    style.map("TCombobox",
              fieldbackground=[("readonly", FIELD), ("disabled", CARD_ALT)],
              foreground=[("readonly", TEXT), ("disabled", TEXT_DIM)],
              selectbackground=[("readonly", FIELD)],
              selectforeground=[("readonly", TEXT)],
              bordercolor=[("focus", ACCENT)])
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)

    # Buttons
    style.configure("TButton", background=CARD_ALT, foreground=TEXT,
                    bordercolor=BORDER, focuscolor=CARD_ALT, padding=(14, 7),
                    relief="flat", font=FONT)
    style.map("TButton",
              background=[("active", ACCENT_HOVER), ("pressed", ACCENT)],
              foreground=[("active", "#ffffff"), ("disabled", TEXT_DIM)])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    bordercolor=ACCENT, focuscolor=ACCENT, padding=(16, 8),
                    relief="flat", font=FONT)
    style.map("Accent.TButton",
              background=[("active", ACCENT_HOVER), ("pressed", "#3a6fd8")],
              foreground=[("disabled", "#8a8d99")])

    # PanedWindow / sash
    style.configure("TPanedwindow", background=BG)

    # Model table (Treeview) — zebra striping via row tags in app.py
    style.configure("Table.Treeview", background=CARD, fieldbackground=CARD,
                    foreground=TEXT, bordercolor=BORDER, relief="flat",
                    rowheight=24, font=FONT_SMALL)
    style.map("Table.Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
    style.configure("Table.Treeview.Heading", background=CARD_ALT,
                    foreground=TEXT_DIM, relief="flat", font=FONT_SMALL,
                    padding=4)
    # Row zebra striping is configured in app.py via tag_configure on the
    # Treeview directly (tag names "row_even"/"row_odd").

    # Console text widget handled directly (tk.Text, not ttk)
    return style


def console_text(root: tk.Tk, parent) -> tk.Text:
    """Create the console Text widget with dark editor styling."""
    return tk.Text(
        parent,
        wrap=tk.WORD,
        state=tk.DISABLED,
        bg=CONSOLE_BG,
        fg=CONSOLE_FG,
        insertbackground=TEXT,
        selectbackground=ACCENT,
        font=FONT_MONO,
        padx=10,
        pady=8,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        borderwidth=0,
    )
