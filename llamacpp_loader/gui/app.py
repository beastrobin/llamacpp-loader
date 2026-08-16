"""Main Tkinter window and all GUI widgets.

Layout::
    +---------------------------------------------------------------+
    |  llamacpp-loader                                              |
    +---------------------------+-----------------------------------+
    | [Model Selector ▼]        | [Console Output (read-only)]      |
    |                           |                                   |
    | -- Inference Params --    | > Server log lines stream here   |
    | ctx_size:  [4096     ]    |                                   |
    | gpu_layers:[35       ]    +-----------------------------------+
    | n_threads: [8        ]    |                                     |
    |                           | -- Sampling Params --               |
    | -- Sampling Params --     | temperature:  [0.7      ]         |
    | temp:      [0.7      ]    | top_k:       [40        ]         |
    | top_p:     [0.95     ]    | top_p (server):[0.95     ]        |
    |                           +-------------------------------------+
    | [Browse Model] [Start Server] [Stop Server] [Restart Server] Status: idle|
    +--------------------------------------------------------------------------+

Control flow::
    1. User clicks "Browse" -> file dialog selects GGUF model directory
    2. Profile list auto-refreshes from ConfigStore
    3. User selects a profile, params populate the fields
    4. User clicks "Start Server":
       a. Build ModelProfile from widget state (or use selected profile)
       b. ProcessManager.start(profile) -> launches llama-server subprocess
       c. SmokeTestRunner.wait_until_ready() polls health endpoint
       d. On PASS: webbrowser.open(http://localhost:<port>)
       e. ConsolePanel starts streaming server output
    5. User clicks "Stop Server" or process exits
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from llamacpp_loader import __version__ as APP_VERSION
from llamacpp_loader.config import recommend
from llamacpp_loader.gui import theme

logger = logging.getLogger(__name__)

# APP_VERSION is imported from llamacpp_loader.__version__ (single source of truth).


# Cache the nvidia-smi lookup — it never moves at runtime, and re-scanning
# the filesystem every 2s poll is wasted work (and, under pythonw.exe, each
# shutil.which / os.listdir call was part of the per-poll cost).
_NVSMI_CACHE: Optional[str] = None
_NVSMI_CACHE_READY: bool = False


def _find_nvidia_smi() -> Optional[str]:
    """Best-effort locate the ``nvidia-smi`` binary (cached after first call).

    It is frequently *not* on ``PATH`` (especially the ``.exe`` on Windows), which
    made the VRAM readout fall back to "N/A".  We first try ``shutil.which``, then
    a set of well-known Windows install locations.  Returns ``None`` if not found.
    """
    global _NVSMI_CACHE, _NVSMI_CACHE_READY
    if _NVSMI_CACHE_READY:
        return _NVSMI_CACHE

    import shutil

    found = shutil.which("nvidia-smi")
    if found:
        _NVSMI_CACHE, _NVSMI_CACHE_READY = found, True
        return found
    candidates = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Program Files (x86)\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            _NVSMI_CACHE, _NVSMI_CACHE_READY = c, True
            return c
    # Fallback: scan the NVIDIA NVSMI directory loosely.
    nvdir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
    try:
        if os.path.isdir(nvdir):
            for name in os.listdir(nvdir):
                if name.lower() == "nvidia-smi.exe":
                    found = os.path.join(nvdir, name)
                    _NVSMI_CACHE, _NVSMI_CACHE_READY = found, True
                    return found
    except Exception:
        pass
    _NVSMI_CACHE_READY = True
    return None


# --------------------------------------------------------------------------- MainWindow


class MainWindow:
    """Single-window main application class."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"llamacpp-loader V{APP_VERSION}")
        theme.apply(root)  # modern dark theme

        # Config store (thread-safe)
        from llamacpp_loader.config.store import ConfigStore
        self.store = ConfigStore()

        # Process manager
        from llamacpp_loader.process_manager.manager import ProcessManager, ServerConfig
        self.proc_mgr = ProcessManager(log_callback=self._on_server_log,
                                  config_store=self.store)

        # State
        self._current_profile_name: Optional[str] = None  # type: ignore[assignment]
        self._selected_profile_name: str = ""  # currently selected row in table
        self._active_port: int = 8080
        # (row, col, event_time) of the last click on a green "changed" overlay
        # cell — used for manual double-click detection (see _refresh_changed_overlays).
        self._overlay_last_click = None

        self._build_ui()
        # One-time backfill: seed the locked "default" preset (community
        # sampling) for every already-registered model, and adopt the
        # recommendation as the live sampling for untouched models.
        self._migrate_presets()
        self._restore_window_state()
        self._restore_sort_state()
        self._refresh_profile_list()

    def _on_server_log(self, line: str) -> None:
        """Callback invoked by ProcessManager when a new log line arrives.

        Thread-safe: uses root.after() to schedule on the main thread.
        """
        self.root.after(0, lambda: self._console_panel.append_line(line))

    def _build_ui(self) -> None:
        """Build all widgets and lay them out."""
        # Main frame with padding
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # First row: llama.cpp install path (user must pick it before first run)
        self._build_path_row(main)

        # Top row: model selector + buttons
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        self._build_toolbar(toolbar)

        # Bottom: status bar. This is packed BEFORE the expandable content area
        # so it reserves its natural height first; the PanedWindow above then
        # only claims the remaining space. Otherwise content.pack(expand=True)
        # hogs the whole frame and the status bar gets clipped off the bottom
        # when the window is resized to the minimum height.
        bottom = ttk.Frame(main)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        self._status_bar = StatusBar(bottom)

        # Main content area: vertical PanedWindow so the user can drag the sash
        # between the model table and the console panes.
        content = tk.PanedWindow(main, orient=tk.VERTICAL, sashrelief=tk.RAISED,
                                 sashwidth=4, bg=theme.BORDER)
        content.pack(fill=tk.BOTH, expand=True)
        # opaqueresize=False: during sash drag only the sash line moves, panes
        # reflow on release. Avoids the severe smear/ghost trail that live
        # (opaque) resizing causes on Windows when the Treeview + console Text
        # widgets must be repainted every mouse-motion frame.
        content.configure(opaqueresize=False)

        # Model list table (full width) with custom header row for Selected label
        table_frame = ttk.Frame(content)

        # Header row: title + Selected label (kept tight to the table below)
        header_row = ttk.Frame(table_frame)
        header_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        title_label = ttk.Label(header_row, text="Model List",
                                 background=theme.CARD, foreground=theme.TEXT,
                                 font=("Microsoft YaHei UI", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=(8, 12), pady=(0, 0))

        self._model_list_selected_label = ttk.Label(header_row, text="Selected: ",
                                                    style="Dim.TLabel")
        self._model_list_selected_label.pack(side=tk.LEFT, padx=(0, 0), pady=(0, 0))

        # Treeview lives in the remaining area. A plain Frame (not a LabelFrame
        # with an empty title) avoids the extra blank row between the "Model
        # List" header and the table.
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Collapse/expand state for the two column groups. The toggle control
        # now lives inside the coloured table header (the first column of each
        # group acts as the group button) instead of a separate banner row.
        self._inference_expanded = True
        self._sampling_expanded = True

        self._build_model_table(tree_container)

        # Add table and console areas to the vertical splitter.
        content.add(table_frame, stretch="always", minsize=200)

        # Console area split: Server Output (left, ~2/3) + Test Results (right, ~1/3)
        # Keep ttk.PanedWindow here: the stock Windows Tk in this Python build does
        # not expose opaqueresize on ttk.PanedWindow, and tk.PanedWindow.add() uses
        # "stretch" instead of "weight". The reported ghost-trail was on the
        # vertical list/console splitter (handled above), so the horizontal pane
        # can stay on ttk.
        console_pane = ttk.PanedWindow(content, orient=tk.HORIZONTAL)

        server_frame = ttk.LabelFrame(console_pane, text="Server Output",
                                      padding=6, style="Card.TLabelframe")
        self._console_panel = ConsolePanel(server_frame)
        self._console_panel.pack(fill=tk.BOTH, expand=True)
        console_pane.add(server_frame, weight=2)

        test_frame = ttk.LabelFrame(console_pane, text="Test Results",
                                    padding=6, style="Card.TLabelframe")
        self._test_panel = ConsolePanel(test_frame, title="Test Results")
        self._test_panel.pack(fill=tk.BOTH, expand=True)
        console_pane.add(test_frame, weight=1)
        content.add(console_pane, stretch="always", minsize=120)

        # Friendly placeholder text so empty panels don't look broken
        self._console_panel.append_line(
            "▶ Click Start to launch the model; llama-server logs stream here…")
        self._test_panel.append_line(
            "Click \"Smoke Test\" to run a health check and speed test "
            "(requires Start first)…")
        # Start the background resource monitor (RAM / VRAM) loop.
        self._update_resources()

    # Column layout (order = Treeview column order).
    # Order note: "speed" sits directly LEFT of "kv"; "size" is left of "quant".
    TABLE_COLUMNS = ("model", "size", "quant", "speed", "inf_group", "kv", "ctx",
                     "reasoning", "gpu", "threads", "mtp", "vision", "sam_group",
                     "temp", "topk", "topp", "rp", "presets")
    TABLE_HEADINGS = {
        "model": "Model",
        "size": "Size",
        "quant": "Quant",
        "kv": "KV Cache",
        "ctx": "Ctx",
        "speed": "Speed",
        "reasoning": "Thinking",
        "gpu": "GPU Layers",
        "threads": "Threads",
        "temp": "Temp",
        "topk": "Top-K",
        "topp": "Top-P",
        "rp": "Repeat",
        "vision": "Vision",
        "mtp": "MTP",
        "presets": "Presets",
        # Group toggle columns — blank data columns that fold their group.
        "inf_group": "Inference",
        "sam_group": "Sampling",
    }
    # Inference group = server-adjacent execution knobs; Sampling group = generation knobs
    INFERENCE_COLS = {"kv", "ctx", "reasoning", "gpu", "threads", "mtp", "vision"}
    SAMPLING_COLS = {"temp", "topk", "topp", "rp"}
    # Column widths: Model gets a generous width; sampling cols are narrow numbers
    TABLE_WIDTHS = {
        "model": 300, "size": 90, "quant": 78, "speed": 66, "inf_group": 70,
        "kv": 70, "ctx": 64, "reasoning": 68, "gpu": 58, "threads": 58,
        "sam_group": 70, "temp": 52, "topk": 52, "topp": 52, "rp": 56,
        "vision": 52, "mtp": 56, "presets": 96,
    }
    # Per-column header background/foreground for the custom (non-ttk) header row.
    # Inference columns get a blue tint, Sampling columns an amber tint, the rest
    # a neutral card tint — mirrors the group banners but at column granularity.
    HEADER_COLORS = {
        # group toggle columns (match their group tint)
        "inf_group": ("#2f3e52", "#cfe3ff"),
        "sam_group": ("#4a3d28", "#ffe6b8"),
        # inference group (blue)
        "kv": ("#2f3e52", "#cfe3ff"),
        "ctx": ("#2f3e52", "#cfe3ff"),
        "reasoning": ("#2f3e52", "#cfe3ff"),
        "gpu": ("#2f3e52", "#cfe3ff"),
        "threads": ("#2f3e52", "#cfe3ff"),
        # sampling group (amber)
        "temp": ("#4a3d28", "#ffe6b8"),
        "topk": ("#4a3d28", "#ffe6b8"),
        "topp": ("#4a3d28", "#ffe6b8"),
        "rp": ("#4a3d28", "#ffe6b8"),
        # neutral (model/quant/speed/presets)
        "model": (theme.CARD, theme.TEXT_DIM),
        "size": (theme.CARD, theme.TEXT_DIM),
        "quant": (theme.CARD, theme.TEXT_DIM),
        "speed": (theme.CARD, theme.TEXT_DIM),
        "mtp": ("#2f3e52", "#cfe3ff"),
        "vision": ("#2f3e52", "#cfe3ff"),
        "presets": (theme.CARD, theme.TEXT_DIM),
    }
    # Grouped heading labels: keep them clean — grouping is shown via banner + cell tint
    HEADING_LABELS = {
        "model": "Model",
        "size": "Size",
        "quant": "Quant",
        "kv": "KV Cache",
        "ctx": "Ctx",
        "speed": "Speed",
        "reasoning": "Thinking",
        "gpu": "GPU Layers",
        "threads": "Threads",
        "temp": "Temp",
        "topk": "Top-K",
        "topp": "Top-P",
        "rp": "Repeat",
        "vision": "Vision",
        "mtp": "MTP",
        "presets": "Presets",
        # Group toggle columns — blank, used only for folding.
        "inf_group": "Inference",
        "sam_group": "Sampling",
    }
    # Which columns get a numeric inline-edit on double click
    EDITABLE_COLS = {"quant", "kv", "ctx", "gpu", "threads", "temp", "topk", "topp", "rp"}
    # Only the *sampling* parameters are governed by the preset and subject to
    # the "default is locked" rule. Model-level inference knobs (kv / ctx /
    # gpu / threads) are per-model hardware/config settings that must stay
    # editable even on the locked default preset — otherwise the green "changed"
    # overlay would silently block editing q8_0 / q4_0 forever.
    SAMPLING_COLS = {"temp", "topk", "topp", "rp"}
    KV_OPTIONS = ["f16", "f32", "q8_0", "q6_k", "q5_k", "q4_k", "q4_0", "q3_k",
                  "iq4_nl", "iq4_xs"]
    # KV-cache compression severity: larger number = more compressed. Used so the
    # KV column sorts by compression level (f16 least, then q8_0, q6_k, q5_k,
    # q4_k/q4_0, q3_k, ...). Unknown types sort last.
    KV_RANK = {
        "f16": 0, "f32": 0,
        "q8_0": 1, "q8_k": 1,
        "q6_k": 2, "q6_0": 2,
        "q5_k": 3, "q5_0": 3, "q5_1": 3,
        "q4_k": 4, "q4_0": 4, "q4_1": 4,
        "iq4_nl": 4, "iq4_xs": 4, "iq4_0": 4,
        "q3_k": 5, "q3_0": 5,
        "iq3_xxs": 5, "iq3_xs": 5,
        "q2_k": 6, "q2_0": 6, "iq2_xxs": 6,
    }
    CTX_OPTIONS = ["4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"]
    # Sort state per column: None / "asc" / "desc"
    SORT_STATE: dict[str, Optional[str]] = {}  # type: ignore[assignment]

    def _group_of(self, col: str) -> Optional[str]:
        """Return the group a column belongs to, or None for ungrouped columns."""
        if col in self.INFERENCE_COLS:
            return "inference"
        if col in self.SAMPLING_COLS:
            return "sampling"
        return None

    def _group_anchor_col(self, group: str) -> Optional[str]:
        """The dedicated blank toggle column that folds a group."""
        return "inf_group" if group == "inference" else "sam_group"

    def _update_group_label(self, group: str, expanded: bool) -> None:
        """Refresh the anchor header cell's arrow to reflect collapsed state."""
        anchor = self._anchor_cols.get(group)
        lbl = self._header_labels.get(anchor) if anchor else None
        if lbl is not None:
            arrow = "▼" if expanded else "▶"
            lbl.config(text=f"  {arrow} {group.capitalize()}  ")

    def _toggle_group(self, group: str) -> None:
        """Collapse/expand a column group (inference or sampling).

        The group's anchor column stays visible at all times so its header can
        act as the re-expand button; the remaining columns are hidden (width 0).
        """
        if group == "inference":
            self._inference_expanded = not self._inference_expanded
            expanded = self._inference_expanded
            cols = self.INFERENCE_COLS
        else:
            self._sampling_expanded = not self._sampling_expanded
            expanded = self._sampling_expanded
            cols = self.SAMPLING_COLS

        anchor = self._group_anchor_col(group)
        for col in cols:
            if col == anchor:
                # Keep the toggle column visible so the group can be re-expanded.
                self._tree.column(col, width=self.TABLE_WIDTHS.get(col, 60),
                                  minwidth=40, stretch=False)
            elif expanded:
                self._tree.column(col, width=self.TABLE_WIDTHS.get(col, 60),
                                  minwidth=40, stretch=False)
            else:
                self._tree.column(col, width=0, minwidth=0, stretch=False)

        self._update_group_label(group, expanded)
        # Keep the custom coloured header aligned with the (possibly collapsed) tree.
        self._relayout_header()

    def _relayout_header(self) -> None:
        """Re-position the custom header labels to match current column widths.

        Called after a group is collapsed/expanded (which zeroes the width of
        hidden columns) so the coloured header stays aligned with the tree.
        """
        if not hasattr(self, "_header_labels"):
            return
        x = 0
        for col in self.TABLE_COLUMNS:
            w = self._tree.column(col, "width") or 0
            lbl = self._header_labels.get(col)
            rsz = self._header_resizers.get(col)
            if lbl is None:
                continue
            if w <= 0:
                lbl.place_forget()
                if rsz is not None:
                    rsz.place_forget()
            else:
                lbl.place(x=x, y=0, width=w, height=26)
                if rsz is not None:
                    rsz.place(x=x + w - 2, y=0)
                x += w
        self._header_canvas.config(scrollregion=(0, 0, x, 26))

    def _start_col_resize(self, col: str, event: tk.Event) -> None:
        """Begin dragging a column border to resize the column."""
        self._resize_col = col
        self._resize_start_x = event.x_root
        self._resize_start_w = self._tree.column(col, "width") or 0

    def _do_col_resize(self, event: tk.Event) -> None:
        """Resize the active column while the sash is being dragged."""
        col = getattr(self, "_resize_col", None)
        if col is None:
            return
        delta = event.x_root - self._resize_start_x
        new_w = max(40, self._resize_start_w + delta)
        self._tree.column(col, width=new_w)
        self._relayout_header()
        self._refresh_changed_overlays()

    def _end_col_resize(self, _event: tk.Event | None = None) -> None:
        """Finish a column resize and clear the drag state."""
        self._resize_col = None
        self._resize_start_x = 0
        self._resize_start_w = 0

    def _build_model_table(self, parent: ttk.Frame) -> None:
        """Build the Treeview model table.

        Columns: model | size | quant | speed | [Inference: kv ctx thinking
                 gpu threads mtp vision] | [Sampling: temp topk topp repeat] |
                 presets. Vision sits inside the Inference group (right of MTP).

        - Double-click a numeric/quant/kv cell to edit inline.
        - Single-click the Thinking column toggles reasoning (greyed when forced).
        - Single-click the Vision column attaches/swaps a mmproj file; attached
          vision models show as a child row beneath the model.
        - Single-click the Presets column opens the parameter archive menu.
        """
        columns = list(self.TABLE_COLUMNS)

        # Custom header row: ttk.Treeview does not support per-column heading
        # background colours, so we render our own coloured header on a Canvas
        # and hide the native heading via a stripped-down style layout.
        style = ttk.Style(self.root)
        try:
            style.layout("NoHead.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        except tk.TclError:
            pass
        style.configure("NoHead.Treeview", background=theme.CARD,
                        fieldbackground=theme.CARD, foreground=theme.TEXT,
                        bordercolor=theme.BORDER, relief="flat",
                        rowheight=24, font=theme.FONT_SMALL)
        # The selection background is supplied by the selected_row.Treeview
        # row tag, which paints one continuous bar. Do NOT also set it via
        # style.map: on Windows the native "selected" background is drawn
        # cell-by-cell and overlaps the tag background, producing gaps / a
        # disconnected highlight bar across columns.
        style.map("NoHead.Treeview",
                  foreground=[("selected", "#ffffff")])

        self._tree = ttk.Treeview(
            parent, columns=columns, show="tree", height=8,
            style="NoHead.Treeview")
        # We render our own coloured header on a Canvas above the tree. Using
        # show="tree" hides the native heading row; the tree column itself is
        # collapsed to zero width so only the data columns remain visible.
        self._tree.column("#0", width=0, minwidth=0, stretch=False)
        for col in columns:
            # Heading command is kept for keyboard/screen-reader activation,
            # but the native heading row itself is hidden.
            self._tree.heading(col,
                               command=lambda c=col: self._sort_by_column(c))
            anchor = tk.W if col == "model" else tk.CENTER
            self._tree.column(col, width=self.TABLE_WIDTHS[col], anchor=anchor,
                              stretch=False, minwidth=40)

        # Keep group sets referenced for downstream consumers (future styling)
        _ = self.INFERENCE_COLS, self.SAMPLING_COLS  # noqa: F841

        # Row zebra striping via tags. The selected tag is appended AFTER the
        # zebra tag on the active row so its background wins — a ttk row-tag
        # background overrides the built-in "selected" style state, which is why
        # selection looked un-highlighted before.
        self._tree.tag_configure("row_even.Treeview", background=theme.CARD)
        self._tree.tag_configure("row_odd.Treeview", background=theme.CARD_ALT)
        self._tree.tag_configure("selected_row.Treeview",
                                 background=theme.ACCENT, foreground="#ffffff")
        # Running model row: green tint so it's obvious which one is live.
        self._tree.tag_configure("running_row.Treeview",
                                 background="#1e3a2a", foreground="#9affc4")

        # Per-cell "changed value" overlays (green text) and running-model state.
        self._changed_overlays: list[tk.Widget] = []
        self._running_profile_name: str = ""

        # --- custom coloured header (Canvas, scrolls in sync with the tree) ---
        self._header_canvas = tk.Canvas(parent, height=26, bg=theme.CARD,
                                        highlightthickness=0)
        self._header_labels: dict[str, tk.Label] = {}
        self._header_resizers: dict[str, tk.Frame] = {}
        # Track an active column resize so the mouse can stray past the handle.
        self._resize_col: str | None = None
        self._resize_start_x = 0
        self._resize_start_w = 0
        # Map each group to its dedicated blank toggle column. The toggle cell
        # stays visible even when the rest of the group is collapsed, so the
        # group can always be re-expanded from the header.
        self._anchor_cols = {
            "inference": self._group_anchor_col("inference"),
            "sampling": self._group_anchor_col("sampling"),
        }
        self._anchor_to_group = {v: k for k, v in self._anchor_cols.items()}
        anchor_set = set(self._anchor_cols.values())
        total_w = 0
        for col in columns:
            bg, fg = self.HEADER_COLORS.get(col, (theme.CARD, theme.TEXT_DIM))
            if col in anchor_set:
                grp = self._anchor_to_group[col]
                text = f"  ▼ {grp.capitalize()}  "
                font = ("Microsoft YaHei UI", 9, "bold")
                cmd = lambda e, g=grp: self._toggle_group(g)
            else:
                text = self.HEADING_LABELS[col]
                font = theme.FONT_SMALL
                cmd = lambda e, c=col: self._sort_by_column(c)
            lbl = tk.Label(self._header_canvas, text=text, bg=bg, fg=fg,
                           font=font, anchor=tk.CENTER, cursor="hand2")
            w = self.TABLE_WIDTHS[col]
            lbl.place(x=total_w, y=0, width=w, height=26)
            lbl.bind("<Button-1>", cmd)
            self._header_labels[col] = lbl

            # Thin resize handle on the right edge of each header cell.
            rsz = tk.Frame(self._header_canvas, width=4, height=26, bg=bg,
                           cursor="sb_h_double_arrow")
            rsz.place(x=total_w + w - 2, y=0)
            rsz.bind("<Button-1>", lambda e, c=col: self._start_col_resize(c, e))
            rsz.bind("<B1-Motion>", self._do_col_resize)
            rsz.bind("<ButtonRelease-1>", self._end_col_resize)
            self._header_resizers[col] = rsz
            total_w += w
        self._header_canvas.config(scrollregion=(0, 0, total_w, 26))

        self._vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(
            yscrollcommand=lambda *a: (self._vsb.set(*a), self._refresh_changed_overlays()))
        self._vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # IMPORTANT: pack the header canvas AFTER the vertical scrollbar so its
        # viewport width matches the tree's (both are parent_width minus the vsb
        # width). If the header were packed first it would span the full parent
        # width and stop scrolling horizontally once columns only slightly exceed
        # the tree width — leaving the header out of sync with the body.
        self._header_canvas.pack(side=tk.TOP, fill=tk.X)

        # Horizontal scrollbar: columns can exceed window width. Dragging it
        # scrolls the tree AND the custom header canvas in lockstep.
        self._hsb = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(
            xscrollcommand=lambda *a: (self._hsb.set(*a),
                                       self._header_canvas.xview_moveto(
                                           self._tree.xview()[0]),
                                       self._refresh_changed_overlays()))
        self._hsb.pack(side=tk.BOTTOM, fill=tk.X)

        self._tree.pack(fill=tk.BOTH, expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-1>", self._on_table_double_click)
        self._tree.bind("<Button-1>", self._on_table_click)
        self._tree.bind("<Motion>", self._on_table_motion)
        self._tree.bind("<Leave>", lambda _e: self._hide_tooltip())
        self._tooltip: tk.Toplevel | None = None
        self._tooltip_text = ""
        self._tree.bind("<Button-3>", self._on_table_click)  # right-click (lock)
        # Recompute green "changed" overlays whenever the tree is (re)laid out.
        self._tree.bind("<Configure>", lambda e: self._refresh_changed_overlays())

        # Inline edit widget (created lazily on first edit)
        self._edit_entry: Optional[tk.Widget] = None
        self._edit_row: Optional[str] = None
        self._edit_col: Optional[str] = None

    # ---------------------------------------------------------------- clicks
    def _col_index(self, col_id: str) -> int:
        """Convert '#N' Treeview column id to 0-based index (0 = tree column)."""
        try:
            return int(col_id.lstrip("#")) - 1
        except ValueError:
            return 0

    def _col_name(self, col_id: str) -> Optional[str]:
        idx = self._col_index(col_id)
        if 0 <= idx < len(self.TABLE_COLUMNS):
            return self.TABLE_COLUMNS[idx]
        return None

    def _on_table_click(self, event) -> None:
        """Single click handling: thinking toggle / vision attach / presets menu.

        Right-click (Button-3) on the Thinking column toggles the forced-ON lock.
        """
        region = self._tree.identify("region", event.x, event.y)
        if region not in ("cell", "tree"):
            return
        row_id = self._tree.identify_row(event.y)
        if not row_id or row_id.startswith("_vision_"):
            return
        col = self._col_name(self._tree.identify_column(event.x))

        if col == "reasoning":
            if event.num == 3:  # right-click => lock/unlock Thinking
                self._toggle_reasoning_forced(row_id)
            else:
                self._toggle_reasoning(row_id)
        elif col == "vision":
            self._attach_vision_model(row_id)
        elif col == "mtp":
            self._open_mtp_menu(row_id)
        elif col == "presets":
            self._open_presets_menu(row_id, event.x_root, event.y_root)

    def _on_table_motion(self, event: tk.Event) -> None:
        """Show a tooltip with the full model name when hovering the model cell."""
        region = self._tree.identify("region", event.x, event.y)
        if region != "cell":
            self._hide_tooltip()
            return
        col = self._tree.identify_column(event.x)
        if col != "#1":  # only the model column
            self._hide_tooltip()
            return
        row_id = self._tree.identify_row(event.y)
        if not row_id or row_id.startswith("_vision_"):
            self._hide_tooltip()
            return
        values = self._tree.item(row_id, "values")
        if not values:
            self._hide_tooltip()
            return
        text = str(values[0])
        if text == self._tooltip_text:
            return
        self._show_tooltip(event.x_root + 12, event.y_root + 12, text)

    def _show_tooltip(self, x: int, y: int, text: str) -> None:
        self._hide_tooltip()
        self._tooltip_text = text
        self._tooltip = tk.Toplevel(self.root)
        self._tooltip.wm_overrideredirect(True)
        self._tooltip.wm_attributes("-topmost", True)
        self._tooltip.configure(bg=theme.BORDER)
        lbl = tk.Label(self._tooltip, text=text, bg=theme.CARD_ALT,
                       fg=theme.TEXT, font=theme.FONT_SMALL, padx=6, pady=3)
        lbl.pack()
        self._tooltip.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self) -> None:
        if getattr(self, "_tooltip", None):
            self._tooltip.destroy()
            self._tooltip = None
        self._tooltip_text = ""

    def _toggle_reasoning(self, name: str) -> None:
        """Toggle the Thinking switch. Forced-ON models stay locked."""
        profile = self.store.load(name)
        if not profile:
            return
        # Locked models ("on*") cannot disable Thinking.
        if profile.reasoning_forced:
            return
        self.store.update(name, {"reasoning": not profile.reasoning})
        self._refresh_model_table(self.store.list_profiles())

    def _toggle_reasoning_forced(self, name: str) -> None:
        """Right-click the Thinking cell to lock/unlock Thinking (forced ON).

        Locking forces ``reasoning=True`` (a locked model must think); unlocking
        restores user control. Persisted to the profile.
        """
        profile = self.store.load(name)
        if not profile:
            return
        new_val = not profile.reasoning_forced
        updates = {"reasoning_forced": new_val}
        if new_val:
            updates["reasoning"] = True  # locking implies Thinking ON
        self.store.update(name, updates)
        self._refresh_model_table(self.store.list_profiles())

    def _attach_vision_model(self, name: str) -> None:
        """Popup menu: attach / replace / detach vision (mmproj) model."""
        profile = self.store.load(name)
        if not profile:
            return
        current_vision = [f for f in profile.extra_files
                          if "mmproj" in f.lower() or "clip" in f.lower()]
        current_text = current_vision[0] if current_vision else "(none)"

        menu = tk.Menu(self._tree, tearoff=0)
        menu.add_command(label=f"Current: {current_text}", state=tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="Attach / Replace...", command=lambda: self._pick_vision_file(name))
        if current_vision:
            menu.add_command(label="Detach", command=lambda: self._detach_vision_model(name))

        # Popup at cursor
        x = self._tree.winfo_pointerx()
        y = self._tree.winfo_pointery()
        menu.tk_popup(x, y)

    def _pick_vision_file(self, name: str) -> None:
        """Open file picker and attach the chosen mmproj file."""
        from tkinter import filedialog as fd
        path = fd.askopenfilename(
            title=f"Attach Vision Model for {name}",
            filetypes=[("GGUF mmproj", "*.gguf"), ("All files", "*.*")],
            initialdir=self.store.get_ui_state().last_browse_dir,
        )
        if not path:
            return
        p = Path(path)
        self.store.set_ui_state(last_browse_dir=str(p.parent))
        profile = self.store.load(name)
        if not profile:
            return
        # Replace first extra file with the chosen vision model.
        # Store the FULL absolute path so the mmproj can live anywhere
        # (e.g. a shared mmproj dir) instead of being force-copied next to
        # every main model — os.path.join(model_path, abs_path) keeps the
        # absolute path, so launch still resolves it correctly.
        extra = [f for f in profile.extra_files if "mmproj" not in f.lower()]
        extra.insert(0, str(p))
        self.store.update(name, {"extra_files": extra})
        self._refresh_model_table(self.store.list_profiles())

    def _detach_vision_model(self, name: str) -> None:
        """Remove any vision (mmproj) file association from the profile."""
        profile = self.store.load(name)
        if not profile:
            return
        extra = [f for f in profile.extra_files
                 if "mmproj" not in f.lower() and "clip" not in f.lower()]
        self.store.update(name, {"extra_files": extra})
        self._refresh_model_table(self.store.list_profiles())

    # ----------------------------------------------------------- MTP (spec decoding)
    def _open_mtp_menu(self, name: str) -> None:
        """Popup menu: enable/disable MTP and attach / auto-detect / detach draft."""
        profile = self.store.load(name)
        if not profile:
            return
        menu = tk.Menu(self._tree, tearoff=0)
        menu.add_command(
            label=f"Draft: {profile.mtp_model or '(none)'}",
            state=tk.DISABLED)
        menu.add_separator()
        label = "Disable MTP" if profile.mtp_enabled else "Enable MTP"
        menu.add_command(label=label, command=lambda: self._toggle_mtp(name))
        menu.add_command(label="Attach / Replace Draft...",
                         command=lambda: self._pick_mtp_file(name))
        menu.add_command(label="Auto-detect Draft",
                         command=lambda: self._auto_detect_mtp(name))
        if profile.mtp_model:
            menu.add_command(label="Detach Draft",
                             command=lambda: self._detach_mtp_model(name))
        x = self._tree.winfo_pointerx()
        y = self._tree.winfo_pointery()
        menu.tk_popup(x, y)

    def _toggle_mtp(self, name: str) -> None:
        """Enable / disable MTP speculative decoding for this profile."""
        profile = self.store.load(name)
        if not profile:
            return
        # Enabling without a draft model is allowed but will be skipped at launch
        # (and surfaced as "on?" in the cell) until one is attached.
        self.store.update(name, {"mtp_enabled": not profile.mtp_enabled})
        self._refresh_model_table(self.store.list_profiles())

    def _pick_mtp_file(self, name: str) -> None:
        """Open file picker and attach an external MTP draft GGUF."""
        from tkinter import filedialog as fd
        path = fd.askopenfilename(
            title=f"Attach MTP Draft Model for {name}",
            filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")],
            initialdir=self.store.get_ui_state().last_browse_dir,
        )
        if not path:
            return
        p = Path(path)
        self.store.set_ui_state(last_browse_dir=str(p.parent))
        # Store the FULL absolute path so the MTP draft can live anywhere
        # instead of being force-copied next to every main model.
        updates = {"mtp_model": str(p), "mtp_enabled": True}
        self.store.update(name, updates)
        self._refresh_model_table(self.store.list_profiles())

    def _auto_detect_mtp(self, name: str) -> None:
        """Auto-pair a sibling '<base>-mtp.gguf' draft in the model directory."""
        profile = self.store.load(name)
        if not profile:
            return
        base_stem = Path(profile.gguf_file).stem.replace(" ", "-").lower()
        draft = None
        try:
            from llamacpp_loader.config.metadata import find_mtp_draft
            draft = find_mtp_draft(profile.model_path, base_stem)
        except Exception:  # noqa: BLE001
            draft = None
        if not draft:
            self._status_bar.set_state("error", f"No MTP draft found for {name}")
            return
        self.store.update(name, {"mtp_model": draft, "mtp_enabled": True})
        self._refresh_model_table(self.store.list_profiles())

    def _detach_mtp_model(self, name: str) -> None:
        """Remove the MTP draft association and disable MTP."""
        profile = self.store.load(name)
        if not profile:
            return
        self.store.update(name, {"mtp_model": "", "mtp_enabled": False})
        self._refresh_model_table(self.store.list_profiles())

    def _open_presets_menu(self, name: str, x: int, y: int) -> None:
        """Popup menu for the parameter archive (presets).

        The built-in ``default`` preset is community-recommended and locked (it
        cannot be overwritten).  User-customizable slots are Preset 1/2/3 — they
        can be applied (if they exist) and saved into (create / overwrite).
        """
        profile = self.store.load(name)
        if not profile:
            return
        menu = tk.Menu(self._tree, tearoff=0)
        active = profile.active_preset or recommend.PRESET_DEFAULT

        # --- Apply section ---
        # The locked community default is always available to apply back to.
        dlabel = "default (community, locked)"
        if recommend.PRESET_DEFAULT == active:
            dlabel += "  ✓"
        menu.add_command(
            label=f"Apply: {dlabel}",
            command=lambda: self._apply_preset(name, recommend.PRESET_DEFAULT))
        for pname in recommend.USER_PRESETS:
            if pname in profile.presets:
                mark = "  ✓" if pname == active else ""
                menu.add_command(
                    label=f"Apply: {pname}{mark}",
                    command=lambda n=pname: self._apply_preset(name, n))
        menu.add_separator()

        # --- Save section (custom slots only; default is locked) ---
        menu.add_command(label="Save as: Preset 1",
                         command=lambda: self._save_preset(name, "Preset 1"))
        menu.add_command(label="Save as: Preset 2",
                         command=lambda: self._save_preset(name, "Preset 2"))
        menu.add_command(label="Save as: Preset 3",
                         command=lambda: self._save_preset(name, "Preset 3"))
        menu.tk_popup(x, y)

    def _apply_preset(self, name: str, preset: str) -> None:
        profile = self.store.load(name)
        if not profile:
            return
        if profile.load_preset(preset):
            self.store.add(profile)  # persist
        self._refresh_model_table(self.store.list_profiles())

    def _save_preset(self, name: str, preset: str) -> None:
        profile = self.store.load(name)
        if not profile:
            return
        if recommend.is_locked_preset(preset):
            self._status_bar.set_state(
                "idle", "default is locked — save your tweaks into Preset 1/2/3")
            return
        profile.save_preset(preset)
        self.store.add(profile)  # persist
        self._refresh_model_table(self.store.list_profiles())

    def _preset_is_locked(self, name: str) -> bool:
        """True if *name*'s active preset is the locked community default."""
        profile = self.store.load(name)
        if not profile:
            return False
        return recommend.is_locked_preset(profile.active_preset or recommend.PRESET_DEFAULT)

    def _sync_active_preset(self, name: str) -> None:
        """Snapshot current params INTO the active *custom* preset (if any).

        Called after an edit so user tweaks to a custom Preset 1/2/3 are saved
        automatically.  No-op while the locked default is active (edits there are
        blocked before they reach this point).
        """
        profile = self.store.load(name)
        if not profile:
            return
        active = profile.active_preset or recommend.PRESET_DEFAULT
        if recommend.is_locked_preset(active):
            return
        profile.save_preset(active)
        self.store.add(profile)

    def _migrate_presets(self) -> None:
        """Backfill the locked ``default`` preset for already-registered models.

        Runs once at startup.  For every profile lacking a ``default`` preset, it
        seeds one from the community recommendation; models still on the
        untouched global defaults also adopt the community sampling live so the
        "default" they see is actually the good baseline.
        """
        changed = False
        for name in self.store.list_profiles():
            profile = self.store.load(name)
            if profile is None:
                continue
            if profile.ensure_default_preset():
                self.store.add(profile)
                changed = True
        if changed:
            logger.info("Migrated community 'default' presets for existing models")

    # ------------------------------------------------------------ inline edit
    def _on_table_double_click(self, event) -> None:
        """Start inline editing for editable columns (direct hit on tree)."""
        region = self._tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self._tree.identify_row(event.y)
        if not row_id or row_id.startswith("_vision_"):
            return
        col = self._col_name(self._tree.identify_column(event.x))
        self._maybe_edit_cell(row_id, col)

    def _maybe_edit_cell(self, row_id: str, col: str) -> None:
        """Open the inline editor for a cell if it is editable.

        Shared by a direct double-click on the Treeview and by a double-click on
        a green "changed" overlay (which would otherwise swallow the event or
        mis-forward coordinates and silently fail to open the editor). The
        overlay passes the row/col it already knows, so no fragile coordinate
        math is needed.
        """
        if not row_id or row_id.startswith("_vision_"):
            return
        if col not in self.EDITABLE_COLS:
            return
        # The read-only "default" preset is community-recommended and locked —
        # block editing of its *sampling* parameter columns (switch to Preset
        # 1/2/3 first). Model-level knobs (kv/ctx/gpu/threads) are always
        # editable, even on the locked default.
        if col in self.SAMPLING_COLS and self._preset_is_locked(row_id):
            self._status_bar.set_state(
                "idle", "default is locked — switch to Preset 1/2/3 to customize")
            return
        self._start_cell_edit(row_id, col, 0, 0)

    def _hide_changed_overlays(self) -> None:
        """Destroy every green "changed cell" overlay label.

        Overlays are children of the tree's parent and stack ABOVE the tree,
        so they would cover an inline edit widget and make the cell look
        uneditable. Call this before popping an editor; repaint afterwards.
        """
        for w in self._changed_overlays:
            try:
                w.destroy()
            except Exception:
                pass
        self._changed_overlays = []

    def _is_overlay_double_click(self, row: str, col: str, ev_time: int) -> bool:
        """Detect the second half of a double-click on a green "changed" cell.

        Tk's native ``<Double-1>`` synthesis can be defeated when a
        ``<Button-1>`` handler calls ``event_generate`` (which is what the
        overlay does to forward single clicks). So the overlay measures the gap
        between consecutive clicks on the same cell and reports a double-click
        here. Returns True for the triggering second click and resets state.
        """
        prev = getattr(self, "_overlay_last_click", None)
        is_double = (prev is not None and prev[0] == row and prev[1] == col
                     and (ev_time - prev[2]) < 400)
        if is_double:
            self._overlay_last_click = None
        else:
            self._overlay_last_click = (row, col, ev_time)
        return is_double

    def _start_cell_edit(self, row_id: str, col: str, x: int, y: int) -> None:
        """Place an Entry/Combobox widget over the clicked cell."""
        if self._edit_entry is not None:
            self._edit_entry.destroy()
            self._edit_entry = None
        # Hide overlays so the editor is not obscured by the green labels.
        self._hide_changed_overlays()
        bbox = self._tree.bbox(row_id, col)
        if not bbox:
            return
        current = self._tree.set(row_id, col)
        if col == "kv":
            entry = ttk.Combobox(self._tree, values=self.KV_OPTIONS, state="readonly")
            entry.set(current if current in self.KV_OPTIONS else "f16")
        elif col == "ctx":
            # Ctx: Combobox with suggestions but free-form typing allowed
            entry = ttk.Combobox(self._tree, values=self.CTX_OPTIONS, state="normal")
            entry.insert(0, current)
        else:
            entry = tk.Entry(self._tree)
            entry.insert(0, current)
        entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        entry.focus_set()
        if hasattr(entry, "select_range"):
            entry.select_range(0, tk.END)

        def commit(event=None) -> None:
            # Guard against a second call (e.g. FocusOut firing right after the
            # widget is destroyed) — a destroyed widget raises on .get().
            if entry is None or not entry.winfo_exists():
                return
            try:
                raw = entry.get()
            except Exception:
                raw = ""
            self._commit_cell_edit(row_id, col, raw)
            if entry.winfo_exists():
                try:
                    entry.unbind("<FocusOut>")
                except Exception:
                    pass
                entry.destroy()
            self._edit_entry = None
            # Repaint green "changed" overlays now that editing is finished.
            self._refresh_changed_overlays()

        def cancel(event=None) -> None:
            if entry is not None and entry.winfo_exists():
                try:
                    entry.unbind("<FocusOut>")
                except Exception:
                    pass
                entry.destroy()
            self._edit_entry = None
            # Repaint green "changed" overlays now that editing is finished.
            self._refresh_changed_overlays()

        entry.bind("<Return>", commit)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", commit)
        if isinstance(entry, ttk.Combobox):
            entry.bind("<<ComboboxSelected>>", commit)
        self._edit_entry = entry

    def _commit_cell_edit(self, row_id: str, col: str, raw: str) -> None:
        """Validate and save an edited cell value back to ConfigStore."""
        profile = self.store.load(row_id)
        if not profile:
            return
        try:
            if col == "ctx":
                v = raw.strip().upper()
                if v.endswith("K"):
                    val = int(v[:-1]) * 1024
                elif v.endswith("M"):
                    val = int(v[:-1]) * 1024 * 1024
                else:
                    val = int(v)
                self.store.update(row_id, {"inference.ctx_size": val})
            elif col == "quant":
                self.store.update(row_id, {"quant": raw.strip()})
            elif col == "kv":
                self.store.update(row_id, {"kv_cache": raw.strip().lower()})
            elif col == "gpu":
                self.store.update(row_id, {"inference.gpu_layers": int(raw)})
            elif col == "threads":
                self.store.update(row_id, {"inference.n_threads": int(raw)})
            elif col == "temp":
                self.store.update(row_id, {"sampling.temperature": float(raw)})
            elif col == "topk":
                self.store.update(row_id, {"sampling.top_k": int(raw)})
            elif col == "topp":
                self.store.update(row_id, {"sampling.top_p": float(raw)})
            elif col == "rp":
                self.store.update(row_id, {"sampling.repeat_penalty": float(raw)})
        except (ValueError, TypeError):
            logger.warning("Invalid value for %s: %r", col, raw)
        else:
            # Persist the edit into the active custom preset (if not locked).
            if col in self.SAMPLING_COLS:
                self._sync_active_preset(row_id)
        self._refresh_model_table(self.store.list_profiles())

    def _on_tree_select(self, event=None) -> None:
        """Load the selected table row into the parameter panel."""
        sel = self._tree.selection()
        if not sel:
            return
        name = sel[0]  # iid is profile_name
        self._selected_profile_name = name
        self._update_toolbar_profile_label(name)
        self._highlight_selected_row(name)
        self._on_profile_selected(None)

    def _highlight_selected_row(self, name: str) -> None:
        """Re-apply zebra striping and the selection highlight to the table.

        The selected row gets an explicit ``selected_row.Treeview`` tag appended
        AFTER its zebra tag, so the tag's ``background`` wins. ttk row-tag
        backgrounds take precedence over the built-in ``selected`` style state,
        which is why selection looked un-highlighted when the zebra tags were the
        only thing driving row colour.
        """
        if not hasattr(self, "_tree"):
            return
        running = getattr(self, "_running_profile_name", "")
        for row in self._tree.get_children():
            tags = [t for t in self._tree.item(row, "tags")
                    if t not in ("selected_row.Treeview", "running_row.Treeview")]
            if row == name:
                tags.append("selected_row.Treeview")
            if row == running:
                # Running tag appended last => its background wins when a row is
                # both selected and running.
                tags.append("running_row.Treeview")
            self._tree.item(row, tags=tuple(tags))
        # Row backgrounds changed (selection/running) => recolour changed cells.
        self._refresh_changed_overlays()

    def _build_path_row(self, parent: ttk.Frame) -> None:
        """First row: llama.cpp install folder selector.

        Picks the folder that contains llama-server(.exe); persisted to
        UiState.llama_server_path so the process manager resolves the binary
        from there. New users must set this before the first launch.
        """
        path_frame = ttk.Frame(parent)
        path_frame.pack(fill=tk.X, pady=(0, 6))

        # Button on the LEFT, labelled "llamacpp path" (user-facing).
        sel_btn = ttk.Button(path_frame, text="llamacpp path",
                             command=self._select_llamacpp_path)
        sel_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._llamacpp_path_var = tk.StringVar(
            value=self.store.get_ui_state().llama_server_path or "")
        self._llamacpp_path_label = ttk.Label(
            path_frame, textvariable=self._llamacpp_path_var,
            style="Dim.TLabel", anchor=tk.W)
        self._llamacpp_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                        padx=(0, 6))

        self._refresh_llamacpp_path_display()

    def _refresh_llamacpp_path_display(self) -> None:
        """Update the path label; highlight in amber when unset."""
        p = self.store.get_ui_state().llama_server_path
        if p:
            self._llamacpp_path_var.set(p)
            self._llamacpp_path_label.config(foreground=theme.TEXT_DIM)
        else:
            self._llamacpp_path_var.set(
                "Not set - click \"llamacpp path\" to choose the folder "
                "containing llama-server(.exe)")
            self._llamacpp_path_label.config(foreground=theme.AMBER)

    def _select_llamacpp_path(self) -> None:
        """Open a directory picker and persist the chosen llama.cpp folder."""
        from tkinter import filedialog as fd
        init = self.store.get_ui_state().llama_server_path or ""
        d = fd.askdirectory(
            title="Select the llama.cpp install folder (must contain llama-server.exe)",
            initialdir=init if os.path.isdir(init) else "")
        if not d:
            return
        self.store.set_ui_state(llama_server_path=d)
        self._refresh_llamacpp_path_display()
        messagebox.showinfo(
            "llamacpp path",
            f"Saved:\n{d}\n\nThe model will use llama-server.exe from this "
            f"folder when launching.")

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        """Build the top toolbar: all controls on a single row.

        Buttons distributed evenly across the window width; each picks up a
        share of the available horizontal space (sticky=EW).
        """
        # 6 equal columns so every toolbar button is exactly the same width:
        # Add model · Remove model · Stop · Restart · Smoke Test · Start
        for i in range(6):
            parent.columnconfigure(i, weight=1)

        browse_btn = ttk.Button(parent, text="Add model",
                                command=self._browse_model)
        browse_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))

        remove_btn = ttk.Button(parent, text="Remove model",
                                command=self._remove_selected_model)
        remove_btn.grid(row=0, column=1, sticky=tk.EW, padx=(4, 4))

        self._toolbar_stop_btn = ttk.Button(
            parent, text="Stop Server", command=self._on_stop, state=tk.DISABLED)
        self._toolbar_stop_btn.grid(row=0, column=2, sticky=tk.EW, padx=(4, 4))

        self._toolbar_restart_btn = ttk.Button(
            parent, text="Restart Server", command=self._on_restart, state=tk.DISABLED)
        self._toolbar_restart_btn.grid(row=0, column=3, sticky=tk.EW, padx=(4, 4))

        self._toolbar_smoke_btn = ttk.Button(
            parent, text="Smoke Test", command=self._on_smoke_test)
        self._toolbar_smoke_btn.grid(row=0, column=4, sticky=tk.EW, padx=(4, 4))

        self._toolbar_start_btn = ttk.Button(parent, text="Start Server", command=self._on_start,
                                             style="Accent.TButton")
        self._toolbar_start_btn.grid(row=0, column=5, sticky=tk.EW, padx=(4, 0))

    def _set_toolbar_running(self, running: bool) -> None:
        """Toggle Start/Stop/Restart button states in the toolbar."""
        if running:
            self._toolbar_start_btn.config(state=tk.DISABLED)
            self._toolbar_stop_btn.config(state=tk.NORMAL)
            self._toolbar_restart_btn.config(state=tk.NORMAL)
        else:
            self._toolbar_start_btn.config(state=tk.NORMAL)
            self._toolbar_stop_btn.config(state=tk.DISABLED)
            self._toolbar_restart_btn.config(state=tk.DISABLED)

    def _update_toolbar_profile_label(self, name: str) -> None:
        """Update the model-selection label on the Model List header."""
        if hasattr(self, "_model_list_selected_label"):
            self._model_list_selected_label.config(text=f"Selected: {name}")

    def _remove_selected_model(self) -> None:
        """Delete the currently selected model profile from the store."""
        sel = self._tree.selection()
        name = None
        if sel:
            cand = sel[0]
            if not cand.startswith("_vision_"):
                name = cand  # iid is profile_name
        if not name:
            # Fall back to selected state
            name = self._selected_profile_name or ""
        if not name:
            messagebox.showinfo("Remove Model", "Please select a model in the list first.")
            return

        # Resolve display_name for the confirmation prompt
        profile = self.store.load(name)
        display = profile.display_name if profile else name
        if not messagebox.askyesno("Remove Model", f"Remove '{display}' from the list?"):
            return

        profiles = self.store.list_profiles()
        if len(profiles) <= 1:
            messagebox.showwarning("Remove Model", "Cannot remove the last profile.")
            return

        self.store.delete(name)
        self._current_profile_name = None
        self._refresh_profile_list()

    def _profile_has_model(self, name: str) -> bool:
        """Return True if the profile has both a model path and a GGUF file."""
        p = self.store.load(name)
        return bool(p and p.gguf_file and p.model_path)

    def _refresh_profile_list(self) -> None:
        """Reload the profile list from ConfigStore and update the table."""
        # Drop any stale placeholder stored under the reserved defaults key
        # (older versions persisted it as a selectable "New Model" profile,
        # which then became the default selection and broke Start).
        self.store.delete("__global_defaults__")

        profiles = self.store.list_profiles()
        if not profiles:
            # First run: seed a clearly-named placeholder (never the reserved key).
            profile = self.store.create_default_profile(
                display_name="New Model",
                model_path="",
                gguf_file="",
            )
            object.__setattr__(profile, "profile_name", "__new_model__")
            self.store.add(profile)
            profiles = self.store.list_profiles()

        self._refresh_model_table(profiles)

        # Default-select the first profile that actually has a model configured,
        # so Start works out of the box instead of pointing at an empty placeholder.
        if not self._selected_profile_name or self._selected_profile_name not in profiles:
            default_name = next(
                (p for p in profiles if self._profile_has_model(p)),
                profiles[0],
            )
            self._selected_profile_name = default_name
            self._update_toolbar_profile_label(self._selected_profile_name)
            self._on_profile_selected(None)

    def _refresh_model_table(self, profiles: list[str]) -> None:
        """Populate the model Treeview from the profile list.

        Each model row carries the 14 columns; attached vision models appear as
        child rows (tree branch) beneath the model row.
        """
        # Preserve the active column sort across refreshes. Any rebuild of the
        # table (toggling Thinking, attaching vision, editing a preset, ...) used
        # to reset the order back to the store's natural order. Now we re-apply
        # the current SORT_STATE so the user's sort survives those actions.
        if self.SORT_STATE:
            sort_col, sort_dir = next(iter(self.SORT_STATE.items()))
            profiles = sorted(
                profiles,
                key=lambda n: self._sort_key(sort_col, n),
                reverse=(sort_dir == "desc"),
            )

        for item in self._tree.get_children():
            self._tree.delete(item)

        row_index = 0
        for name in profiles:
            profile = self.store.load(name)
            if not profile:
                continue
            inf = profile.inference
            sam = profile.sampling

            # Format ctx as "256K" / "64K" instead of raw 262144 / 65536
            ctx_kb = inf.ctx_size // 1024
            ctx_str = f"{ctx_kb // 1024}M" if ctx_kb >= 1024 else f"{ctx_kb}K"

            # Thinking column: locked "on*" if forced, else on/off toggle text
            forced = profile.reasoning_forced
            if forced:
                think_str = "on*"  # locked ON — cannot be turned off
            else:
                think_str = "on" if profile.reasoning else "off"

            vision_files = [f for f in profile.extra_files
                          if "mmproj" in f.lower() or "clip" in f.lower()]
            vision_str = "👁" if vision_files else ""
            # MTP cell: enabled with a draft -> "on", enabled without one -> "on?",
            # disabled -> "off" (turns green via the changed-overlay when "on").
            if profile.mtp_enabled:
                mtp_str = "on" if profile.mtp_model else "on?"
            else:
                mtp_str = "off"
            presets_str = profile.active_preset or recommend.PRESET_DEFAULT
            # The community "default" preset is locked — flag it so the user
            # knows edits must go into a custom Preset 1/2/3.
            if (profile.active_preset or recommend.PRESET_DEFAULT) == recommend.PRESET_DEFAULT:
                presets_str = "default 🔒"
            size_str = self._model_size_str(profile)

            # MoE badge — autonomous detection from GGUF expert_count.
            model_disp = profile.display_name or name
            if profile.is_moe:
                model_disp = f"{model_disp} [MoE]"

            self._tree.insert(
                "", tk.END, iid=name, values=(
                    model_disp,
                    size_str,
                    profile.quant or "",
                    f"{profile.speed:.1f}" if profile.speed > 0 else "",
                    "",                              # inf_group: blank fold toggle
                    profile.kv_cache or "f16",
                    ctx_str,
                    think_str,
                    inf.gpu_layers,
                    inf.n_threads,
                    mtp_str,
                    vision_str,
                    "",                              # sam_group: blank fold toggle
                    f"{sam.temperature:.2f}",
                    sam.top_k,
                    f"{sam.top_p:.2f}",
                    f"{sam.repeat_penalty:.2f}",
                    presets_str,
                ),
                tags=(f"row_{'even' if row_index % 2 == 0 else 'odd'}.Treeview",),
            )
            row_index += 1

        # Re-apply the selection highlight. The table is rebuilt on edits/sorts,
        # which clears the native Treeview selection, so we re-stamp an explicit
        # "selected" tag on the active row here (and in _on_tree_select).
        self._highlight_selected_row(self._selected_profile_name)

    def _model_size_str(self, profile) -> str:
        """Return the on-disk size of the model GGUF, e.g. '5.9 GB' / '6249 MB'."""
        if not profile.model_path or not profile.gguf_file:
            return ""
        p = os.path.join(profile.model_path, profile.gguf_file)
        try:
            sz = os.path.getsize(p)
        except OSError:
            return ""
        gb = sz / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{sz / (1024 ** 2):.0f} MB"

    def _model_size_bytes(self, profile) -> int:
        """On-disk GGUF size in bytes (0 if unavailable) — used for sorting."""
        if not profile.model_path or not profile.gguf_file:
            return 0
        p = os.path.join(profile.model_path, profile.gguf_file)
        try:
            return os.path.getsize(p)
        except OSError:
            return 0

    def _kv_rank(self, kv: str) -> int:
        """Return compression-level rank of a KV-cache type.

        Higher number = more compressed. Unknown types sort last (99).
        """
        return self.KV_RANK.get((kv or "").lower(), 99)

    # Columns whose cell turns green when the value differs from the default.
    CHANGED_COLS = ("kv", "ctx", "reasoning", "gpu", "threads", "mtp",
                    "temp", "topk", "topp", "rp")

    def _default_param_values(self) -> dict:
        """Baseline parameter values used to detect user changes."""
        return {
            "kv": "f16",
            "ctx": 4096,
            "reasoning": False,
            "mtp": False,
            "gpu": -1,
            "threads": 4,
            "temp": 0.7,
            "topk": 40,
            "topp": 0.95,
            "rp": 1.1,
        }

    def _baseline(self, profile) -> dict:
        """Baseline a cell is compared against for the green 'changed' flag.

        Uses the active preset's stored snapshot when present, so the indicator
        means "differs from the current preset" rather than the bare global
        defaults.  Falls back to :meth:`_default_param_values` for fields not
        captured by the snapshot (notably ``kv_cache``, which lives outside
        ``params_snapshot``).
        """
        g = self._default_param_values()
        ap = profile.active_preset or recommend.PRESET_DEFAULT
        snap = profile.presets.get(ap) or {}
        inf = snap.get("inference", {}) or {}
        sam = snap.get("sampling", {}) or {}
        return {
            "kv": g["kv"],
            "ctx": inf.get("ctx_size", g["ctx"]),
            "gpu": inf.get("gpu_layers", g["gpu"]),
            "threads": inf.get("n_threads", g["threads"]),
            "temp": sam.get("temperature", g["temp"]),
            "topk": sam.get("top_k", g["topk"]),
            "topp": sam.get("top_p", g["topp"]),
            "rp": sam.get("repeat_penalty", g["rp"]),
            "reasoning": g["reasoning"],
            "mtp": g["mtp"],
        }

    def _is_changed_cell(self, col: str, profile, d: dict) -> bool:
        """True if *col*'s value differs from its default (so it should be green)."""
        if col == "kv":
            return (profile.kv_cache or "f16") != d["kv"]
        if col == "ctx":
            return profile.inference.ctx_size != d["ctx"]
        if col == "reasoning":
            return profile.reasoning != d["reasoning"]
        if col == "mtp":
            return profile.mtp_enabled != d["mtp"]
        if col == "gpu":
            return profile.inference.gpu_layers != d["gpu"]
        if col == "threads":
            return profile.inference.n_threads != d["threads"]
        if col == "temp":
            return abs(profile.sampling.temperature - d["temp"]) > 1e-6
        if col == "topk":
            return profile.sampling.top_k != d["topk"]
        if col == "topp":
            return abs(profile.sampling.top_p - d["topp"]) > 1e-6
        if col == "rp":
            return abs(profile.sampling.repeat_penalty - d["rp"]) > 1e-6
        return False

    def _row_bg(self, tags: tuple) -> str:
        """Background colour for a row given its current tags."""
        if "running_row.Treeview" in tags:
            return "#1e3a2a"
        if "selected_row.Treeview" in tags:
            return theme.ACCENT
        if "row_even.Treeview" in tags:
            return theme.CARD
        if "row_odd.Treeview" in tags:
            return theme.CARD_ALT
        return theme.CARD

    def _read_resources(self):
        """Return (ram_used, ram_total, vram_used, vram_total) in GB.

        RAM: try psutil, fall back to ctypes GlobalMemoryStatusEx (Windows) or
        /proc/meminfo (Linux). VRAM: nvidia-smi if available, else 0. Any
        failure degrades gracefully to 0 (rendered as N/A in the UI).
        """
        ram_used = ram_total = vram_used = vram_total = 0.0
        # --- RAM ---
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_used = vm.used / (1024 ** 3)
            ram_total = vm.total / (1024 ** 3)
        except Exception:
            try:
                if os.name == "nt":
                    import ctypes
                    class _MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    ms = _MEMORYSTATUSEX()
                    ms.dwLength = ctypes.sizeof(ms)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                        ram_total = ms.ullTotalPhys / (1024 ** 3)
                        ram_used = (ms.ullTotalPhys - ms.ullAvailPhys) / (1024 ** 3)
                elif os.path.exists("/proc/meminfo"):
                    info = {}
                    with open("/proc/meminfo") as fh:
                        for line in fh:
                            parts = line.split(":")
                            if len(parts) == 2:
                                info[parts[0].strip()] = parts[1].strip()
                    total_kb = float(info.get("MemTotal", "0").split()[0])
                    avail_kb = float(
                        info.get("MemAvailable", info.get("MemFree", "0")).split()[0])
                    ram_total = total_kb / (1024 ** 2)
                    ram_used = (total_kb - avail_kb) / (1024 ** 2)
            except Exception:
                pass
        # --- VRAM ---
        try:
            import subprocess
            nvsmi = _find_nvidia_smi()
            if nvsmi:
                run_kwargs = {}
                # CREATE_NO_WINDOW (0x08000000) stops a black console window from
                # flashing on every poll when the app is launched via pythonw.exe
                # (which has no owning console). This was the cause of the
                # recurring black-window flicker + sluggishness on launch.
                if os.name == "nt":
                    run_kwargs["creationflags"] = 0x08000000
                out = subprocess.run(
                    [nvsmi, "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    **run_kwargs,
                )
                if out.returncode == 0:
                    first = out.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in first.split(",")]
                    if len(parts) == 2 and parts[0] and parts[1]:
                        vram_used = float(parts[0]) / 1024
                        vram_total = float(parts[1]) / 1024
        except Exception:
            pass
        return ram_used, ram_total, vram_used, vram_total

    def _update_resources(self) -> None:
        """Poll system RAM/VRAM usage, update the status bar, reschedule."""
        ram_u, ram_t, vram_u, vram_t = self._read_resources()
        try:
            self._status_bar.set_resources(ram_u, ram_t, vram_u, vram_t)
        except Exception:
            pass
        # Reschedule every 2s; no-op in headless test environments (no mainloop).
        if getattr(self, "root", None) is not None:
            self.root.after(2000, self._update_resources)

    def _refresh_changed_overlays(self) -> None:
        """Paint a green Label over every cell whose value differs from default.

        ttk.Treeview has no per-cell colouring, so we overlay small Labels
        positioned exactly on the changed cells. They are rebuilt on scroll,
        resize, selection and run-state changes (see _build_model_table).
        """
        # Don't repaint while an inline editor is open — the editor must stay
        # unobstructed, and we repaint explicitly on commit/cancel.
        if getattr(self, "_edit_entry", None) is not None:
            return
        if not hasattr(self, "_tree") or not hasattr(self, "_changed_overlays"):
            return
        for w in self._changed_overlays:
            try:
                w.destroy()
            except Exception:
                pass
        self._changed_overlays = []

        # No geometry yet (tree not realised) — skip until <Configure> fires.
        if not self._tree.winfo_ismapped():
            return

        master = self._tree.master
        base_x = self._tree.winfo_x()
        base_y = self._tree.winfo_y()
        for row in self._tree.get_children():
            profile = self.store.load(row)
            if profile is None:
                continue
            # Baseline is per-profile: compare against the active preset's
            # stored snapshot so the green flag means "edited vs this preset".
            d = self._baseline(profile)
            tags = self._tree.item(row, "tags")
            # Changed-cell overlays sit on top of the row and reuse the row's
            # current background (zebra, selected-blue, running-green) so the
            # cell does not look visually detached from its row.  The green
            # foreground text remains the only "value differs from preset" cue.
            bg = self._row_bg(tags)
            for col in self.CHANGED_COLS:
                if not self._is_changed_cell(col, profile, d):
                    continue
                bbox = self._tree.bbox(row, col)
                if not bbox:
                    continue  # scrolled out / hidden column
                x, y, w, h = bbox
                # Clip vertically: a green "changed" flag on the last visible
                # row must never spill down onto the horizontal scrollbar (which
                # sits just below the tree). Skip painting it instead of letting
                # it overlap the scrollbar thumb.
                if y + h > self._tree.winfo_height():
                    continue
                val = self._tree.set(row, col)
                anchor = tk.W if col == "model" else tk.CENTER
                # Overlay a green label on the changed cell. Tk does NOT let
                # events fall through to a widget underneath, so bindtags=()
                # would just swallow clicks. Instead we explicitly forward
                # click/double-click to the Treeview below (same parent
                # coordinate space) so sorting, inline editing and the
                # thinking/vision/presets cell actions keep working on changed
                # cells.
                lbl = tk.Label(master, text=val, bg=bg, fg=theme.GREEN,
                               font=theme.FONT_SMALL, anchor=anchor)
                # Forward single clicks to the Treeview below (so row selection
                # and column actions still work on a green "changed" cell), and
                # detect double-clicks ourselves. Tk's native <Double-1> synthesis
                # can be defeated when a <Button-1> handler calls event_generate,
                # which is exactly why changed cells used to be uneditable — the
                # double-click landed but never opened the editor. We measure the
                # gap between consecutive clicks on the same cell and open the
                # inline editor through the same path a normal cell uses.
                def _on_overlay_click(ev, r=row, c=col, cx=x, cy=y):
                    self._tree.event_generate(
                        "<Button-1>", x=cx + ev.x, y=cy + ev.y)
                    if self._is_overlay_double_click(r, c, getattr(ev, "time", 0)):
                        self._maybe_edit_cell(r, c)
                lbl.bind("<Button-1>", _on_overlay_click)
                # Belt-and-suspenders: keep the native double-click binding too.
                lbl.bind("<Double-1>",
                         lambda e, r=row, c=col: self._maybe_edit_cell(r, c))
                lbl.place(x=base_x + x, y=base_y + y, width=w, height=h)
                self._changed_overlays.append(lbl)

        # Keep scrollbars painted above the green "changed" overlays. The
        # overlays are rebuilt on every scroll/resize and would otherwise be
        # stacked on top of the scrollbar thumbs, hiding/clobbering them.
        if getattr(self, "_vsb", None) is not None:
            self._vsb.lift()
        if getattr(self, "_hsb", None) is not None:
            self._hsb.lift()

    def _sort_by_column(self, col: str) -> None:
        """Click handler: sort by column. Toggle asc/desc on repeated clicks.

        Clicking the active column flips direction (desc -> asc in one click);
        clicking a new column starts at ascending. There is no "clear" state —
        the previous sort is always replaced, which matches the expected
        header-toggle UX.
        """
        cur = self.SORT_STATE.get(col)
        # Toggle between ascending and descending.
        if cur == "desc":
            new_dir = "asc"
        else:  # None (new column) or "asc" -> flip to desc
            new_dir = "desc" if cur == "asc" else "asc"
        # Clear all sort state and set the new one
        self.SORT_STATE.clear()
        self.SORT_STATE[col] = new_dir

        # Persist the new sort for next launch.
        self.store.set_ui_state(sort_column=col, sort_dir=new_dir)

        # Update custom header labels to show the sort arrow.
        self._update_sort_headers()

        sorted_names = sorted(
            self.store.list_profiles(),
            key=lambda n: self._sort_key(col, n),
            reverse=(new_dir == "desc"),
        )
        self._refresh_model_table(sorted_names)

    def _update_sort_headers(self) -> None:
        """Repaint the ▲/▼ arrows on column headers from the current SORT_STATE.

        Group anchor cells (which show the "▼ Group" toggle) are left untouched.
        """
        anchor_cols = set(self._anchor_cols.values()) if hasattr(self, "_anchor_cols") else set()
        for c in self.TABLE_COLUMNS:
            if c in anchor_cols:
                continue
            base = self.HEADING_LABELS[c]
            suffix = ""
            if self.SORT_STATE.get(c) == "asc":
                suffix = " \u25B2"
            elif self.SORT_STATE.get(c) == "desc":
                suffix = " \u25BC"
            lbl = self._header_labels.get(c)
            if lbl is not None:
                lbl.config(text=base + suffix)

    def _restore_sort_state(self) -> None:
        """Restore the last-used column sort from disk before the first paint.

        ``_refresh_model_table`` already re-applies ``SORT_STATE`` on every
        rebuild, so we only need to seed it here; the initial
        ``_refresh_profile_list`` call will then sort the table accordingly.
        """
        ui = self.store.get_ui_state()
        col, dir_ = ui.sort_column, ui.sort_dir
        if col in self.TABLE_COLUMNS and dir_ in ("asc", "desc"):
            self.SORT_STATE = {col: dir_}
        self._update_sort_headers()

    def _sort_key(self, col: str, name: str):
        """Return the sort key for a profile.

        Shared by column-header sorting and by ``_refresh_model_table`` so that
        any table rebuild (e.g. toggling Thinking) keeps the active sort.
        """
        profile = self.store.load(name)
        if profile is None:
            return ""
        NUMERIC_COLS = {"gpu", "threads", "temp", "topk", "topp", "rp"}
        if col == "ctx":
            return profile.inference.ctx_size
        if col == "speed":
            return profile.speed
        if col == "size":
            # Sort by real on-disk size (bytes), not the formatted string.
            return self._model_size_bytes(profile)
        if col == "kv":
            # Sort by KV-cache compression level (f16 = least compressed).
            return self._kv_rank(profile.kv_cache or "f16")
        if col in NUMERIC_COLS:
            return getattr(profile.inference if col in ("gpu", "threads") else profile.sampling,
                           col)
        return str(getattr(profile, col, "")) or profile.display_name

    def _browse_model(self) -> None:
        """Open a file picker for one or more GGUF files.

        Supports multi-file models (e.g. vision models with a separate mmproj):
        the first selected file becomes the main model; any others are kept in
        ``profile.extra_files`` and the first one with ``mmproj`` in the filename
        is auto-flagged for ``--mmproj`` at launch.
        """
        from tkinter import filedialog as fd
        paths = fd.askopenfilenames(
            title="Select GGUF Model File(s)",
            filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")],
            initialdir=self.store.get_ui_state().last_browse_dir,
        )
        if not paths:
            return  # User cancelled

        # Normalize: keep first as main, rest as extras
        main_path = Path(paths[0])
        extra_paths = [Path(p) for p in paths[1:]]

        # Remember directory for next browse
        self.store.set_ui_state(last_browse_dir=str(main_path.parent))

        profile_name = main_path.stem.replace(" ", "-").lower()
        display_name = main_path.stem

        existing_profile = self.store.load(profile_name)
        # Store absolute paths for extra files (e.g. mmproj) so they resolve
        # correctly wherever they live, not only beside the main model.
        extra_files = [str(p) for p in extra_paths]

        if existing_profile:
            existing_profile.model_path = str(main_path.parent)
            existing_profile.gguf_file = main_path.name
            existing_profile.extra_files = extra_files
            prof = existing_profile
        else:
            from llamacpp_loader.config.store import ModelProfile  # local import
            prof = ModelProfile(
                profile_name=profile_name,
                display_name=display_name,
                model_path=str(main_path.parent),
                gguf_file=main_path.name,
                extra_files=extra_files,
            )
        # Seed the locked "default" preset with community-recommended sampling.
        prof.ensure_default_preset()
        self.store.add(prof)

        # Autonomous capability detection: read GGUF for MoE / MTP support and
        # auto-pair a sibling "<base>-mtp.gguf" draft model if present.
        try:
            from llamacpp_loader.config.store import _enrich_profile_from_gguf
            _enrich_profile_from_gguf(prof, main_path, None)
            self.store.add(prof)
        except Exception as exc:  # noqa: BLE001
            logger.debug("GGUF enrichment skipped for %s: %s", profile_name, exc)

        # Refresh UI
        self._refresh_profile_list()
        self._load_profile_to_ui(profile_name)

    def _on_profile_selected(self, event=None) -> None:
        """Handler for profile selection change."""
        name = getattr(self, "_selected_profile_name", "") or ""
        if not name:
            return
        self._current_profile_name = name
        self._load_profile_to_ui(name)

    def _load_profile_to_ui(self, name: str) -> None:
        """Load a profile's parameters into the UI widgets.

        In the table-based layout, parameter values are shown directly in the
        Model List Treeview; selecting a row updates the port indicator.
        """
        profile = self.store.load(name)
        if not profile:
            logger.warning("Profile %s not found", name)
            return

        # Update port indicator to match selected profile
        self._active_port = profile.server.port
        self._status_bar.set_port(profile.server.port)
        # NOTE: do NOT rebuild the table here — rebuilding clears the native
        # Treeview selection and makes the active row look un-highlighted.
        # Just re-stamp the selection highlight.
        self._highlight_selected_row(name)

    def _restore_window_state(self) -> None:
        """Restore saved window dimensions and position.

        The saved geometry is clamped to the window minimum so a previously
        shrunk-then-closed window can never reopen shorter than the status bar
        (the bottom-most row) needs to stay visible.
        """
        MIN_W, MIN_H = 960, 640
        ui = self.store.get_ui_state()
        w = max(int(ui.window_width), MIN_W)
        h = max(int(ui.window_height), MIN_H)
        self.root.minsize(MIN_W, MIN_H)
        self.root.geometry(f"{w}x{h}")

    def save_window_state(self) -> None:
        """Save current window state to ConfigStore before exit."""
        try:
            geom = self.root.geometry()  # "960x680+100+200"
            size_part = geom.split("+")[0]
            w, h = map(int, size_part.split("x"))
            self.store.set_ui_state(window_width=w, window_height=h)
        except (ValueError, IndexError):
            pass  # Ignore parse errors on exit

    def _get_selected_profile(self) -> Optional[ModelProfile]:
        """Return the currently selected profile if it is valid and exists on disk.

        Shows a message box on error and returns None.
        """
        name = self._selected_profile_name or ""
        if not name:
            messagebox.showwarning("No Profile", "Please select or create a model profile first.")
            return None

        profile = self.store.load(name)
        if not profile:
            messagebox.showerror("Error", f"Profile '{name}' not found in config store.")
            return None

        # Resolve and verify the model file exists on disk before launching.
        if profile.model_path and profile.gguf_file:
            model_file = os.path.join(profile.model_path, profile.gguf_file)
        elif profile.gguf_file:
            model_file = profile.gguf_file
        else:
            model_file = ""
        if not model_file or not os.path.isfile(model_file):
            messagebox.showerror(
                "Model File Missing",
                f"Model file not found:\n{model_file}\n\n"
                "Click 'Browse...' to select the GGUF model again.",
            )
            return None

        return profile

    @staticmethod
    def _port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
        """Return True if a TCP listener answers on (host, port)."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _do_start_profile(self, profile: ModelProfile) -> bool:
        """Launch the server for *profile* and update the UI.

        Returns True on successful process launch. Does NOT run a smoke test.
        """
        # Guard against launching a duplicate server. If the target port is
        # already occupied (typically an orphaned llama-server left running by a
        # previous loader session that was closed without stopping), refuse
        # instead of blindly spawning a second process and doubling VRAM/RAM.
        host = profile.server.host
        probe_host = "127.0.0.1" if (not host or host in ("0.0.0.0",)) else host
        if self._port_in_use(probe_host, profile.server.port):
            self._status_bar.set_state(
                "error", f"Port {profile.server.port} already in use")
            messagebox.showerror(
                "Port Already In Use",
                f"Something is already listening on port {profile.server.port}.\n\n"
                "This is usually a leftover llama-server from a previous loader "
                "session that was closed without stopping it.\n\n"
                "Stop that process first (e.g. end 'llama-server.exe' in Task "
                "Manager, or run:  taskkill /f /im llama-server.exe), then start "
                "again — or pick a different port for this profile.")
            return False

        success = self.proc_mgr.start(profile)
        if not success:
            self._status_bar.set_state("error", "Failed to start server")
            messagebox.showerror(
                "Start Failed",
                "Could not launch llama-server.\n\n"
                "Check the 'Server Output' panel for the exact error "
                "(executable not found, bad flags, or model load failure).",
            )
            return False

        # Update UI state
        self._active_port = profile.server.port
        self._status_bar.set_state("running", f"Running on port {profile.server.port}")
        self._status_bar.set_port(profile.server.port)
        # Clear the server-output panel and show a launch banner; real logs
        # stream in live from the subprocess reader thread.
        self._console_panel.clear()
        self._console_panel.append_line(
            f"▶ Starting {profile.display_name} on port {profile.server.port}…")
        self._set_toolbar_running(True)

        # Highlight the running row (green tint) and ensure it stays selected.
        name = profile.profile_name
        self._running_profile_name = name
        self._tree.selection_set(name)
        self._highlight_selected_row(name)
        return True

    def _on_start(self) -> None:
        """Start the server with the currently selected profile."""
        profile = self._get_selected_profile()
        if profile is None:
            return
        if not self._do_start_profile(profile):
            return

        # Run smoke test in background thread, then open browser on success
        threading.Thread(
            target=self._smoke_test_and_open_browser,
            args=(profile,),
            daemon=True,
        ).start()

    def _smoke_test_and_open_browser(self, profile: ModelProfile) -> None:
        """Run smoke test and open browser if server is healthy.

        Runs in a background thread to avoid blocking the GUI.
        """
        from llamacpp_loader.smoke_test.runner import (
            SmokeTestRunner, SmokeResult, SmokeTestResult)

        try:
            runner = SmokeTestRunner(
                host=profile.server.host,
                port=profile.server.port,
                timeout=120.0,
            )
            poll_interval = 0.5
            start_time = time.monotonic()
            last_status_update = 0.0

            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= runner._timeout:
                    break
                # Show "loading" feedback every few seconds
                if elapsed - last_status_update >= 3.0:
                    self.root.after(
                        0,
                        lambda e=elapsed: self._status_bar.set_state(
                            "running", f"Server loading... {int(e)}s"),
                    )
                    last_status_update = elapsed
                result = runner.run()
                if result.status == SmokeResult.PASS:
                    break
                # If the server process died early, fail fast instead of waiting full timeout
                if self.proc_mgr.state.value in {"error", "idle"} and self.proc_mgr.get_pid() is None:
                    result = SmokeTestResult(
                        status=SmokeResult.CONNECTION_ERROR,
                        detail="Server process exited before becoming ready",
                    )
                    break
                time.sleep(poll_interval)

            if result.status == SmokeResult.PASS:
                # Success — open browser on the main thread
                self.root.after(
                    0,
                    lambda: self._open_browser(profile.server.port),
                )
                self._status_bar.set_state("running", f"Server ready (latency: {result.latency_ms}ms)")
            else:
                # Failed — update UI on main thread
                detail = result.detail or "Unknown error"
                self.root.after(
                    0,
                    lambda d=detail: self._status_bar.set_state("error", f"Smoke test failed: {d}"),
                )

        except Exception as exc:
            logger.exception("Smoke test exception")
            self.root.after(
                0,
                lambda e=str(exc): self._status_bar.set_state("error", f"Error: {e}"),
            )

    def _open_browser(self, port: int) -> None:
        """Open the default browser to the server\'s web UI."""
        import webbrowser
        url = f"http://localhost:{port}"
        logger.info("Opening browser: %s", url)
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Failed to open browser: %s", exc)

    def _on_stop(self) -> None:
        """Stop the running server."""
        self.proc_mgr.stop()
        self._status_bar.set_state("idle", "Server stopped")
        self._set_toolbar_running(False)
        # Drop the running highlight.
        self._running_profile_name = ""
        self._highlight_selected_row(self._selected_profile_name)

    def _on_restart(self) -> None:
        """Restart the server with current profile settings."""
        name = self._selected_profile_name or ""
        if not name:
            messagebox.showwarning("No Profile", "Please select a model profile first.")
            return

        profile = self.store.load(name)
        if not profile:
            return

        success = self.proc_mgr.restart(profile)
        if success:
            self._status_bar.set_state("running", f"Restarting on port {profile.server.port}")
            self._set_toolbar_running(True)

    def _on_smoke_test(self) -> None:
        """Run smoke test against the active server.

        If no server is currently running, this starts the selected profile first,
        then runs the test. Progress is printed into the Test Results panel.
        """
        from llamacpp_loader.smoke_test.runner import SmokeTestRunner, SmokeResult

        profile = self._get_selected_profile()
        if profile is None:
            return
        name = profile.profile_name

        # Decide whether the selected model is already the one serving.
        running = self.proc_mgr.is_running()
        same_model = bool(self._running_profile_name) and self._running_profile_name == name

        if not running:
            self._test_panel.append_line(
                "WARNING: no running llama-server detected - auto-starting the "
                "selected model first...")
            if not self._do_start_profile(profile):
                self._test_panel.append_line("[FAIL] model failed to start, Smoke Test aborted")
                return
        elif not same_model:
            # A different model is still serving — stop it (synchronous, releases
            # the port) before starting the newly selected model.
            self._test_panel.append_line(
                f"WARNING: {self._running_profile_name} is currently running, "
                f"which differs from the selected {name} - stopping the old "
                f"service before starting the selected model...")
            self._on_stop()
            if not self._do_start_profile(profile):
                self._test_panel.append_line("[FAIL] model failed to start, Smoke Test aborted")
                return
        else:
            self._test_panel.append_line(
                f"{name} is already running - running Smoke Test directly...")

        port = profile.server.port
        self._test_panel.append_line(f"--- Smoke test on port {port} ---")

        def worker() -> None:
            try:
                runner = SmokeTestRunner(host="127.0.0.1", port=port, timeout=60.0)
                self.root.after(0, lambda: self._test_panel.append_line("[1/2] Checking server health..."))
                result = runner.wait_until_ready(poll_interval=0.5)

                if result.status != SmokeResult.PASS:
                    self.root.after(0, lambda: self._test_panel.append_line(
                        f"[FAIL] Health check: {result.status.value} - {result.detail}"))
                    self.root.after(0, lambda: self._status_bar.set_state("error", "Smoke test failed"))
                    return

                self.root.after(0, lambda: self._test_panel.append_line(
                    f"[OK] Server healthy (HTTP {result.http_code}, latency {result.latency_ms}ms)"))
                self.root.after(0, lambda: self._test_panel.append_line(
                    "[2/2] Measuring generation speed..."))

                # Speed test: try OpenAI-compatible generation endpoints first,
                # fall back to the native /tokenize endpoint for builds / states
                # that do not expose generation endpoints (some builds return 404
                # on /v1/completions while slots are still loading).
                import time as _time
                import urllib.request
                import urllib.error
                import json as _json
                import re as _re

                def _extract_speed(msg: str):
                    """Pull the tokens/s number out of a speed result message."""
                    m = _re.search(r"([\d.]+)\s*tokens?/s", msg)
                    return float(m.group(1)) if m else None

                def _pure_decode_speed(data):
                    """Pure generation (decode) speed from llama.cpp timings.

                    Uses the server's own measured generation phase (predicted_*
                    or eval_*), which EXCLUDES prompt-processing time. This is the
                    exact number the llama-server logs per request and is the
                    standard llama.cpp benchmark metric, so the GUI Speed column
                    now matches the terminal instead of undercounting by mixing in
                    prompt processing. Falls back to None if timings are absent.
                    """
                    if not isinstance(data, dict):
                        return None
                    timings = data.get("timings") or {}
                    if not timings:
                        return None
                    # Newer llama.cpp server keys
                    pred_ms = timings.get("predicted_ms")
                    pred_n = timings.get("predicted_n")
                    if pred_ms and pred_n and pred_ms > 0:
                        return pred_n / (pred_ms / 1000.0)
                    # Older keys
                    eval_ms = timings.get("eval_time_ms")
                    eval_n = timings.get("eval_n_tokens")
                    if eval_ms and eval_n and eval_ms > 0:
                        return eval_n / (eval_ms / 1000.0)
                    # Direct per-second field if present
                    ps = timings.get("predicted_per_second") or timings.get("eval_per_second")
                    if ps:
                        return float(ps)
                    return None

                def _try_speed(url: str, payload: str, parse):
                    t0 = _time.time()
                    req = urllib.request.Request(
                        url,
                        data=payload.encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        body = resp.read().decode()
                    elapsed = _time.time() - t0
                    return parse(_json.loads(body), elapsed)

                def _parse_completion(data, elapsed):
                    n = (data.get("usage", {}) or {}).get("completion_tokens") or 0
                    pure = _pure_decode_speed(data)
                    if pure is not None:
                        return f"[OK] Generated {n} tokens -> {pure:.1f} tokens/s (decode, excl. prompt)"
                    if elapsed > 0:
                        return f"[OK] Generated {n} tokens in {elapsed:.1f}s -> {n / elapsed:.1f} tokens/s"
                    return "[OK] Generated 0 tokens"

                def _parse_chat(data, elapsed):
                    n = (data.get("usage", {}) or {}).get("completion_tokens") or 0
                    pure = _pure_decode_speed(data)
                    if pure is not None:
                        return f"[OK] Chat generated {n} tokens -> {pure:.1f} tokens/s (decode, excl. prompt)"
                    if elapsed > 0:
                        return f"[OK] Chat generated {n} tokens in {elapsed:.1f}s -> {n / elapsed:.1f} tokens/s"
                    return "[OK] Chat generated 0 tokens"

                def _parse_tokenize(data, elapsed):
                    n = len(data.get("tokens", []))
                    if elapsed > 0:
                        return f"[OK] Tokenized {n} tokens in {elapsed:.1f}s -> {n / elapsed:.1f} tok/s (tokenize only)"
                    return "[OK] Tokenized 0 tokens"

                endpoints = [
                    (
                        f"http://127.0.0.1:{port}/v1/completions",
                        '{"model": "model", "prompt": "Introduce yourself in 50 words", "max_tokens": 100, "temperature": 0.7}',
                        _parse_completion,
                    ),
                    (
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        '{"model": "model", "messages": [{"role": "user", "content": "Introduce yourself in 50 words"}], "max_tokens": 100, "temperature": 0.7}',
                        _parse_chat,
                    ),
                    (
                        f"http://127.0.0.1:{port}/tokenize",
                        '{"content": "Introduce yourself in 50 words"}',
                        _parse_tokenize,
                    ),
                ]

                speed_msg = "[WARN] No compatible generation endpoint available for speed test"
                last_error = None
                for url, payload, parse in endpoints:
                    try:
                        speed_msg = _try_speed(url, payload, parse)
                        break
                    except urllib.error.HTTPError as exc:
                        last_error = exc
                        # 404/405 means the endpoint is not exposed; try the next one.
                        if exc.code not in (404, 405):
                            raise
                else:
                    if last_error:
                        raise last_error

                self.root.after(0, lambda: self._test_panel.append_line(speed_msg))
                status_text = "Smoke passed"
                if "tokenize" in speed_msg:
                    status_text = "Smoke passed (tokenize only)"

                # Persist the measured speed back to the profile so the Speed
                # column shows a real value after a passed smoke test.
                speed_val = _extract_speed(speed_msg)
                if speed_val is not None and self._selected_profile_name:
                    self.store.update(self._selected_profile_name, {"speed": speed_val})
                    self.root.after(
                        0, lambda: self._refresh_model_table(self.store.list_profiles()))

                self.root.after(0, lambda: self._status_bar.set_state("running", status_text))

            except Exception as exc:
                logger.exception("Smoke test error")
                error_text = f"[ERROR] {exc}"
                self.root.after(0, lambda: self._test_panel.append_line(error_text))

        threading.Thread(target=worker, daemon=True).start()


# --------------------------------------------------------------------------- ParameterPanel


class ParameterPanel(ttk.Frame):
    """Left-side parameter input panel.

    .. deprecated::
        The table-based UI edits parameters inline in the Treeview, so this
        standalone panel is no longer wired into the main window. Kept only
        because ``tests/test_gui.py`` still exercises it; do not use in new code.
    """

    def __init__(self, parent: ttk.Frame, store=None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._store = store  # Optional[ConfigStore] for defaults
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create all input widgets in this panel."""
        # Inference section
        inf_frame = ttk.LabelFrame(self, text="Inference", padding=10, style="Card.TLabelframe")
        inf_frame.pack(fill=tk.X, pady=(0, 8))

        # ctx_size
        ttk.Label(inf_frame, text="Context Size:", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._ctx_var = tk.IntVar(value=4096)
        self._ctx_spin = ttk.Spinbox(
            inf_frame, from_=64, to=131072, increment=512,
            textvariable=self._ctx_var, width=12, command=self._on_param_change)
        self._ctx_spin.grid(row=0, column=1, sticky=tk.W, pady=3)

        # gpu_layers
        ttk.Label(inf_frame, text="GPU Layers:", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._gpu_var = tk.IntVar(value=-1)
        self._gpu_spin = ttk.Spinbox(
            inf_frame, from_=-1, to=999, textvariable=self._gpu_var, width=12,
            command=self._on_param_change)
        self._gpu_spin.grid(row=1, column=1, sticky=tk.W, pady=3)

        # n_threads
        ttk.Label(inf_frame, text="Threads:", style="Card.TLabel").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._threads_var = tk.IntVar(value=4)
        self._threads_spin = ttk.Spinbox(
            inf_frame, from_=1, to=128, textvariable=self._threads_var, width=12,
            command=self._on_param_change)
        self._threads_spin.grid(row=2, column=1, sticky=tk.W, pady=3)

        # Sampling section
        sam_frame = ttk.LabelFrame(self, text="Sampling", padding=10, style="Card.TLabelframe")
        sam_frame.pack(fill=tk.X, pady=(0, 8))

        # temperature
        ttk.Label(sam_frame, text="Temperature:", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._temp_var = tk.DoubleVar(value=0.7)
        self._temp_spin = ttk.Spinbox(
            sam_frame, from_=0.0, to=2.0, increment=0.05,
            textvariable=self._temp_var, width=12, command=self._on_param_change)
        self._temp_spin.grid(row=0, column=1, sticky=tk.W, pady=3)

        # top_k
        ttk.Label(sam_frame, text="Top-K:", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._topk_var = tk.IntVar(value=40)
        self._topk_spin = ttk.Spinbox(
            sam_frame, from_=1, to=256, textvariable=self._topk_var, width=12,
            command=self._on_param_change)
        self._topk_spin.grid(row=1, column=1, sticky=tk.W, pady=3)

        # top_p (server param)
        ttk.Label(sam_frame, text="Top-P:", style="Card.TLabel").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._topp_var = tk.DoubleVar(value=0.95)
        self._topp_spin = ttk.Spinbox(
            sam_frame, from_=0.0, to=1.0, increment=0.01,
            textvariable=self._topp_var, width=12, command=self._on_param_change)
        self._topp_spin.grid(row=2, column=1, sticky=tk.W, pady=3)

    def _on_param_change(self) -> None:
        """Auto-save current parameter values to ConfigStore.

        Edits are blocked while the active preset is the locked ``default``
        (the widgets are disabled in :meth:`set_from_profile`, but we guard here
        too).  When a custom Preset 1/2/3 is active, the tweak is also snapshotted
        into that preset so it is saved automatically.
        """
        if not self._store or not self._current_profile_name:
            return  # type: ignore[attr-defined]
        name = str(self._current_profile_name)  # type: ignore[arg-type]
        profile = self._store.load(name)
        if profile is None:
            return
        active = profile.active_preset or recommend.PRESET_DEFAULT
        if recommend.is_locked_preset(active):
            return  # default is locked — nothing to save

        try:
            self._store.update(
                name,
                {
                    "inference.ctx_size": int(self._ctx_var.get()),
                    "inference.gpu_layers": int(self._gpu_var.get()),
                    "inference.n_threads": max(1, int(self._threads_var.get())),
                    "sampling.temperature": round(float(self._temp_var.get()), 2),
                    "sampling.top_k": max(1, int(self._topk_var.get())),
                    "sampling.top_p": round(float(self._topp_var.get()), 2),
                },
            )
        except Exception as exc:
            logger.warning("Failed to auto-save parameters: %s", exc)
            return

        # Snapshot into the active custom preset so the tweak is persisted.
        profile = self._store.load(name)
        if profile is not None and not recommend.is_locked_preset(
                profile.active_preset or recommend.PRESET_DEFAULT):
            profile.save_preset(profile.active_preset)
            self._store.add(profile)

    def set_from_profile(self, profile) -> None:
        """Load a ModelProfile\'s values into the UI widgets.

        Args:
            profile: A ModelProfile from ConfigStore.load()
        """
        self._current_profile_name = profile.profile_name  # type: ignore[attr-defined]
        self._ctx_var.set(profile.inference.ctx_size)
        self._gpu_var.set(profile.inference.gpu_layers)
        self._threads_var.set(max(1, profile.inference.n_threads))
        self._temp_var.set(profile.sampling.temperature)
        self._topk_var.set(profile.sampling.top_k)
        self._topp_var.set(profile.sampling.top_p)

        # Disable the controls when the active preset is the read-only default
        # (community recommendation cannot be edited in place).
        locked = recommend.is_locked_preset(
            profile.active_preset or recommend.PRESET_DEFAULT)
        state = tk.DISABLED if locked else tk.NORMAL
        for w in (self._ctx_spin, self._gpu_spin, self._threads_spin,
                  self._temp_spin, self._topk_spin, self._topp_spin):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def get_profile(self, display_name: str, model_path: str, gguf_file: str):
        """Build a ModelProfile from current widget values.

        Returns a fully-populated ModelProfile ready for ProcessManager.start().
        """
        from llamacpp_loader.config.store import (
            ModelProfile, ServerParams, InferenceParams, SamplingParams)
        return ModelProfile(
            profile_name=self._current_profile_name or display_name.lower().replace(" ", "-"),  # type: ignore[attr-defined]
            display_name=display_name,
            model_path=model_path,
            gguf_file=gguf_file,
            server=ServerParams(host="127.0.0.1", port=8080),
            inference=InferenceParams(
                ctx_size=int(self._ctx_var.get()),
                gpu_layers=int(self._gpu_var.get()),
                n_threads=max(1, int(self._threads_var.get())),
            ),
            sampling=SamplingParams(
                temperature=float(self._temp_var.get()),
                top_k=max(1, int(self._topk_var.get())),
                top_p=float(self._topp_var.get()),
            ),
        )


# --------------------------------------------------------------------------- ConsolePanel


class ConsolePanel(ttk.Frame):
    """Reusable text log panel (Server Output or Test Results)."""

    def __init__(self, parent: ttk.Frame, title: str = "Server Output", **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._title = title
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Create the read-only console output widget."""
        # Embed self inside a LabelFrame so the title shows; the outer parent
        # is already a LabelFrame so use plain Frame here to avoid double frame.
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        # Text widget with vertical scrollbar (themed dark editor)
        self._text = theme.console_text(self.winfo_toplevel(), frame)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._text.yview)
        self._text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(fill=tk.BOTH, expand=True)

    def append_line(self, line: str) -> None:
        """Append a single log line to the console output.

        Thread-safe via root.after(). Appends at end of text widget and
        auto-scrolls to show new content.
        """
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, line + "\n")
        # Auto-scroll to bottom
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        """Clear all console output."""
        self._text.configure(state=tk.NORMAL)
        self._text.delete(1.0, tk.END)
        self._text.configure(state=tk.DISABLED)


# --------------------------------------------------------------------------- StatusBar


class StatusBar(ttk.Frame):
    """Bottom status bar showing process state and server info."""

    def __init__(self, parent: ttk.Frame, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._build_widgets()

    def _build_widgets(self) -> None:
        # Port indicator on the LEFT (user wants it visible first)
        self._port_label = ttk.Label(self, text="Port: 8080", style="Dim.TLabel")
        self._port_label.pack(side=tk.LEFT, padx=(0, 8))
        # Status text with flat relief — no white border box.
        self._label = ttk.Label(
            self, text="Status: idle", relief=tk.FLAT, anchor=tk.W)
        self._label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Resource monitor (RAM / VRAM) on the RIGHT — bottom-right corner.
        # Keeps the numeric GB readout AND adds two horizontal usage bars with a
        # live percentage, so load is visible at a glance.
        # Red "blood-bar" style for the meters (Windows themed progressbars
        # honour the style background as the bar fill colour).
        try:
            ttk.Style().configure("Blood.Horizontal.TProgressbar",
                                  background="#e53935", troughcolor="#2a1414")
        except Exception:
            pass

        self._resource_frame = ttk.Frame(self)
        self._resource_frame.pack(side=tk.RIGHT, padx=(8, 0))

        # One horizontal row holding both RAM and VRAM meters side by side.
        row = ttk.Frame(self._resource_frame)
        row.pack(side=tk.TOP)

        def _make_meter(name: str):
            cell = ttk.Frame(row)
            cell.pack(side=tk.LEFT, padx=(0, 12))
            num = ttk.Label(cell, text=f"{name} -/-", width=17,
                            style="Dim.TLabel", anchor=tk.W)
            num.pack(side=tk.LEFT, padx=(0, 4))
            bar = ttk.Progressbar(cell, orient=tk.HORIZONTAL, length=120,
                                  mode="determinate", maximum=100,
                                  style="Blood.Horizontal.TProgressbar")
            bar.pack(side=tk.LEFT)
            pct = ttk.Label(cell, text="--%", width=6,
                            style="Dim.TLabel", anchor=tk.E)
            pct.pack(side=tk.LEFT, padx=(4, 0))
            return num, bar, pct

        self._ram_num, self._ram_bar, self._ram_pct = _make_meter("RAM")
        self._vram_num, self._vram_bar, self._vram_pct = _make_meter("VRAM")
        self.pack(fill=tk.X)

    def set_state(self, state: str, message: str = "") -> None:
        """Update the status bar with a new state and optional detail message.

        Args:
            state: One of "idle", "running", "error" — used for color coding.
            message: Human-readable detail shown after the state label.
        """
        colors = {"idle": "#888888", "running": "#4caf50", "error": "#f44336"}
        self._label.config(text=f"Status: {state}  |  {message}", foreground=colors.get(state, "#888888"))

    def set_port(self, port: int) -> None:
        """Update the port indicator (right side of the status bar)."""
        self._port_label.config(text=f"Port: {port}")

    def set_resources(self, ram_used: float, ram_total: float,
                      vram_used: float, vram_total: float) -> None:
        """Update the bottom-right RAM/VRAM usage indicator.

        Shows the numeric GB readout, a horizontal usage bar, and a live
        percentage for both RAM and VRAM.
        """
        if ram_total > 0:
            ram_pct = int(round(ram_used / ram_total * 100))
            self._ram_num.config(text=f"RAM {ram_used:.1f}/{ram_total:.1f}G")
            self._ram_bar.config(value=ram_pct)
            self._ram_pct.config(text=f"{ram_pct}%")
        else:
            self._ram_num.config(text="RAM N/A")
            self._ram_bar.config(value=0)
            self._ram_pct.config(text="N/A")
        if vram_total > 0:
            vram_pct = int(round(vram_used / vram_total * 100))
            self._vram_num.config(text=f"VRAM {vram_used:.1f}/{vram_total:.1f}G")
            self._vram_bar.config(value=vram_pct)
            self._vram_pct.config(text=f"{vram_pct}%")
        else:
            self._vram_num.config(text="VRAM N/A")
            self._vram_bar.config(value=0)
            self._vram_pct.config(text="N/A")


# --------------------------------------------------------------------------- Entry point (used by main.py and tests)

def create_app(root=None):
    """Factory function to create a MainWindow instance.

    Used both by main.py (with real tk.Tk()) and by test_gui for unit testing.
    """
    if root is None:
        root = tk.Tk()
        mw = MainWindow(root)

        # Clean exit on window close: gracefully stop the managed llama-server
        # so it does not linger as an orphan still holding VRAM/RAM after the GUI
        # exits. This is what prevented re-opening the loader from spawning a
        # second server on top of the first (double VRAM).
        def on_closing():
            logger.info("App closing — stopping managed server if running")
            try:
                mw.proc_mgr.stop()
            except Exception as exc:  # never block window close on stop failure
                logger.warning("Error stopping server on close: %s", exc)
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        return mw
    return MainWindow(root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app(tk.Tk())
    tk.mainloop()

