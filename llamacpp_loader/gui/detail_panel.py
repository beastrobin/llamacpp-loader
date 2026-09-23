"""Compact, scrollable editor for the selected model profile."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from llamacpp_loader.config.budget import estimate_vram
from llamacpp_loader.config.store import (
    DEFAULT_NGRAM_TYPE,
    NGRAM_LABELS,
    REASONING_EFFORTS,
    REASONING_MODES,
    normalize_ngram_type,
    normalize_reasoning,
    normalize_reasoning_effort,
    ngram_label,
)
from llamacpp_loader.gui import theme


class ModelDetailPanel(ttk.Frame):
    def __init__(self, parent, *, on_change, on_preset, on_action, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_change, self._on_preset, self._on_action = on_change, on_preset, on_action
        self._profile_name = ""
        self._loading = False
        self._vars = {}
        self._inputs = {}
        self._capability_status = {}
        self._scroll_sync_after = None
        self._last_canvas_window_width = None
        self._last_scrollregion = None
        self._build_widgets()

    def _build_widgets(self):
        ttk.Style(self).configure("Detail.TFrame", background=theme.BG)
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(body, highlightthickness=0, bg=theme.BG)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._vscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self._canvas.yview)
        self._vscroll.grid(row=0, column=1, sticky="ns")
        self._hscroll = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=self._canvas.xview)
        self._hscroll.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=self._vscroll.set, xscrollcommand=self._hscroll.set)
        body.rowconfigure(0, weight=1); body.columnconfigure(0, weight=1)
        inner = ttk.Frame(self._canvas, padding=(2, 0, 2, 4), style="Detail.TFrame")
        self._inner_window = self._canvas.create_window((0, 0), window=inner, anchor="nw")
        # A horizontal sash drag emits a Configure event for every pixel.
        # Reflowing this canvas synchronously on every event makes the form
        # visibly jump on Windows, so coalesce the work until the pointer
        # pauses (or the drag ends).
        inner.bind("<Configure>", self._request_scroll_sync)
        self._canvas.bind("<Configure>", self._request_scroll_sync)
        self._pages = {}
        # Header lives inside the same canvas as the settings pages, so it
        # follows horizontal scrolling and cannot be clipped independently.
        title_row = ttk.Frame(inner, style="Detail.TFrame")
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        title_row.columnconfigure(1, weight=1)
        self._title = ttk.Label(title_row, text="Select a model", width=28,
                                anchor=tk.W,
                                font=("Microsoft YaHei UI", 12, "bold"))
        self._title.grid(row=0, column=0, sticky="w")
        self._tabbar = tk.Frame(title_row, bg=theme.BG)
        self._tabbar.grid(row=0, column=2, sticky="e", padx=(18, 0))
        self._tab_buttons = {}
        self._active_tab = "Quick"
        for tab_name in ("Quick", "Generation", "Advanced"):
            button = tk.Button(self._tabbar, text=tab_name, relief=tk.FLAT,
                               bd=0, width=11, padx=4, pady=3, cursor="hand2",
                               command=lambda n=tab_name: self._select_tab(n))
            button.pack(side=tk.LEFT, padx=(0, 3))
            self._tab_buttons[tab_name] = button
        self._summary = ttk.Label(inner, text="", style="Dim.TLabel",
                                  wraplength=560, justify=tk.LEFT)
        self._summary.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._vram = self._summary
        quick, generation, advanced = (ttk.Frame(inner, padding=6, style="Detail.TFrame") for _ in range(3))
        self._pages = {"Quick": quick, "Generation": generation, "Advanced": advanced}
        for page in self._pages.values():
            page.grid(row=2, column=0, sticky="nsew")
        inner.grid_rowconfigure(2, weight=1)
        inner.grid_columnconfigure(0, weight=1)
        self._select_tab("Quick")
        # Keep the two setting groups compact and left-aligned.  Giving both
        # columns a weight spreads them across the whole canvas, creating a
        # large, unhelpful gap on wide windows.
        q_left = ttk.Frame(quick, style="Detail.TFrame"); q_left.grid(row=0, column=0, sticky="nw")
        q_right = ttk.Frame(quick, style="Detail.TFrame"); q_right.grid(row=0, column=1, sticky="nw", padx=(48, 0))
        self._quick_columns = (q_left, q_right)
        self._add_combo(q_left, 0, "Context (K)", "ctx", ("32", "64", "128", "256", "512"), "32")
        self._add_gpu_row(q_left, 1)
        self._add_cpu_moe_row(q_left, 2)
        self._add_combo(q_left, 3, "KV cache", "kv", ("F16", "Q8", "Q4"), "F16")
        self._add_field(q_left, 4, "Batch size", "batch", "512", "")
        # Physical micro-batch (-ub).  Hermes-style prefill recipes pair a
        # large -b with -ub; 0 keeps llama.cpp's default (512).  The manager
        # refuses to emit -ub when it exceeds -b (llama.cpp would not start).
        self._add_field(q_left, 5, "Ubatch", "ubatch", "0", "0 = auto")
        self._add_field(q_right, 0, "CPU threads", "threads", "4", "")
        self._add_combo(q_right, 1, "Flash Attention", "flash", ("auto", "on", "off"), "auto")
        # Reasoning is tri-state.  "auto" leaves the decision to llama.cpp (it
        # detects it from the chat template) and emits no launch flag at all.
        self._add_combo(q_right, 2, "Reasoning", "reasoning", REASONING_MODES, "auto")
        self._add_field(q_right, 3, "Parallel", "parallel", "1", "")
        # Max tokens caps --n-predict so a long thinking trace cannot swallow
        # the whole generation budget; 0 keeps llama.cpp's default (unbounded).
        self._add_field(q_right, 4, "Max tokens", "max_tokens", "0", "0 = auto")
        g_left = ttk.Frame(generation, style="Detail.TFrame"); g_left.grid(row=0, column=0, sticky="nw")
        g_right = ttk.Frame(generation, style="Detail.TFrame"); g_right.grid(row=0, column=1, sticky="nw", padx=(48, 0))
        gen_fields = (("Temperature", "temp", "0.7"), ("Top-K", "topk", "40"), ("Top-P", "topp", "0.95"), ("Min-P", "minp", ""), ("Repeat penalty", "repeat", "1.1"), ("Seed", "seed", "-1"), ("Frequency penalty", "frequency", "0.0"), ("Presence penalty", "presence", "0.0"))
        for row, item in enumerate(gen_fields[:4]): self._add_field(g_left, row, *item, "")
        for row, item in enumerate(gen_fields[4:]): self._add_field(g_right, row, *item, "")
        # Thinking budget: caps the reasoning trace on its own, so the final
        # answer still fits inside Max tokens (which caps thinking+answer
        # together).  Every control here is optional -- llama.cpp exits on
        # arguments it does not recognise, so a blank box must emit no flag.
        think = ttk.Frame(generation, style="Detail.TFrame")
        think.grid(row=1, column=0, columnspan=2, sticky="w", pady=(16, 0))
        ttk.Label(think, text="Thinking budget").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 3))
        self._add_field(think, 1, "Reasoning budget", "reasoning_budget", "",
                        "blank=off, -1=unlimited, 0=stop, N=tokens")
        self._add_field(think, 2, "Budget message", "reasoning_budget_msg", "",
                        "injected when the budget runs out")
        self._add_combo(think, 3, "Reasoning effort", "reasoning_effort",
                        REASONING_EFFORTS, "default")
        ttk.Label(think, text="template-side hint; default = no flag",
                  style="Dim.TLabel").grid(row=3, column=2, sticky=tk.W,
                                           padx=(6, 0), pady=2)
        advanced.columnconfigure(0, weight=1)
        self._add_vision_row(advanced, 0)
        self._add_draft_row(advanced, 1, "MTP", "mtp")
        self._add_ngram_row(advanced, 2)
        self._add_draft_row(advanced, 3, "DFlash", "dflash")
        self._add_backend_sampling_row(advanced, 4)
        self._add_metrics_row(advanced, 5)
        # Depth/batch guidance that does not fit on the rows themselves.
        ttk.Label(advanced,
                  text="MTP n-max is model-dependent: too deep can nullify the "
                       "gain (Qwen3.x vendor recipe: 2).  Ubatch requires "
                       "Batch >= Ubatch.",
                  style="Dim.TLabel", wraplength=560, justify=tk.LEFT
                  ).grid(row=6, column=0, sticky="w", pady=(8, 0))
        self._hint = None
        self._bind_mousewheel(self)
        self.after_idle(self._sync_scrollregion)

    def _request_scroll_sync(self, _event=None):
        """Coalesce resize work so the detail form stays stable while dragging."""
        if self._scroll_sync_after is not None:
            try:
                self.after_cancel(self._scroll_sync_after)
            except tk.TclError:
                pass
        self._scroll_sync_after = self.after(75, self._sync_scrollregion)

    def _sync_scrollregion(self):
        """Fit the settings surface to the viewport unless content is wider."""
        self._scroll_sync_after = None
        if not hasattr(self, "_inner_window"):
            return
        viewport = self._canvas.winfo_width()
        inner = self._canvas.nametowidget(self._canvas.itemcget(self._inner_window, "window"))
        requested = inner.winfo_reqwidth()
        width = max(viewport, requested)
        if width != self._last_canvas_window_width:
            self._canvas.itemconfigure(self._inner_window, width=width)
            self._last_canvas_window_width = width
        scrollregion = (0, 0, width, self._canvas.bbox("all")[3] if self._canvas.bbox("all") else 0)
        if scrollregion != self._last_scrollregion:
            self._canvas.configure(scrollregion=scrollregion)
            self._last_scrollregion = scrollregion
        if viewport > 0 and width <= viewport + 1:
            self._canvas.xview_moveto(0)
            self._hscroll.state(["disabled"])
        else:
            self._hscroll.state(["!disabled"])

    def _select_tab(self, name):
        """Show one parameter page and give its tab a clear active state."""
        self._active_tab = name
        for tab_name, page in self._pages.items():
            if tab_name == name:
                page.tkraise()
            button = self._tab_buttons[tab_name]
            active = tab_name == name
            button.configure(bg=theme.TAB_ACTIVE if active else theme.CARD_ALT,
                             fg="#ffffff" if active else theme.TEXT_DIM,
                             activebackground=theme.TAB_HOVER,
                             activeforeground="#ffffff")

    def _add_field(self, parent, row, label, key, default, help_text):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.StringVar(value=default); self._vars[key] = var
        entry = ttk.Entry(parent, textvariable=var, width=9); entry.grid(row=row, column=1, sticky=tk.W, padx=(8, 0), pady=2); self._inputs[key] = entry
        if help_text: ttk.Label(parent, text=help_text, style="Dim.TLabel").grid(row=row, column=2, sticky=tk.W, padx=(6, 0), pady=2)
        entry.bind("<Return>", lambda _e, k=key: self._commit(k)); entry.bind("<FocusOut>", lambda _e, k=key: self._commit(k))

    def _add_combo(self, parent, row, label, key, values, default, preset=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.StringVar(value=default); self._vars[key] = var
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=9); combo.grid(row=row, column=1, sticky=tk.W, padx=(8, 0), pady=2); self._inputs[key] = combo
        combo.bind("<<ComboboxSelected>>", lambda _e, k=key, p=preset: self._commit(k, preset=p))

    def _add_check(self, parent, row, label, key):
        var = tk.BooleanVar(value=False); self._vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var, command=lambda: self._commit(key)).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

    def _add_gpu_row(self, parent, row):
        ttk.Label(parent, text="GPU layers").grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.StringVar(value="-1"); self._vars["gpu"] = var
        entry = ttk.Entry(parent, textvariable=var, width=9)
        entry.grid(row=row, column=1, sticky="w", padx=(8, 2), pady=2); self._inputs["gpu"] = entry
        ttk.Button(parent, text="Auto", width=9, style="DetailAction.TButton", command=self._set_gpu_auto).grid(row=row, column=2, padx=2, pady=2)
        ttk.Button(parent, text="Recommend", width=9, style="DetailAction.TButton", command=lambda: self._action("recommend")).grid(row=row, column=3, padx=(2, 0), pady=2)
        entry.bind("<Return>", lambda _e: self._commit("gpu"))
        entry.bind("<FocusOut>", lambda _e: self._commit("gpu"))

    def _advanced_row(self, parent, row, stretch_column):
        """Give each Advanced setting its own grid so mixed controls align."""
        frame = ttk.Frame(parent, style="Detail.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(stretch_column, weight=1)
        return frame

    def _add_status_entry(self, parent, row, key, *, column=1):
        var = tk.StringVar(value="Not attached")
        self._vars[key] = var
        entry = ttk.Entry(parent, textvariable=var, state="readonly", width=22)
        entry.grid(row=row, column=column, sticky="ew", padx=(8, 4), pady=2)
        self._capability_status[key] = entry
        return entry

    def _add_vision_row(self, parent, row):
        line = self._advanced_row(parent, row, 4)
        ttk.Label(line, text="Vision", width=10).grid(row=0, column=0, sticky=tk.W, pady=2)
        mode = tk.StringVar(value="Off")
        self._vars["vision_enabled"] = mode
        combo = ttk.Combobox(line, textvariable=mode, values=("Off", "On"), state="readonly", width=7)
        combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 4), pady=2)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._commit_vision())
        ttk.Button(line, text="Browse", width=9, style="DetailAction.TButton",
                   command=lambda: self._action("vision_pick")).grid(row=0, column=2, padx=2, pady=2, ipady=3)
        ttk.Button(line, text="Clear", width=9, style="DetailAction.TButton",
                   command=lambda: self._action("vision_clear")).grid(row=0, column=3, padx=(2, 0), pady=2, ipady=3)
        self._add_status_entry(line, 0, "vision_status", column=4)

    def _add_draft_row(self, parent, row, label, prefix):
        line = self._advanced_row(parent, row, 4)
        ttk.Label(line, text=label, width=10).grid(row=0, column=0, sticky=tk.W, pady=2)
        mode_key = f"{prefix}_enabled"
        mode = tk.StringVar(value="Off")
        self._vars[mode_key] = mode
        combo = ttk.Combobox(line, textvariable=mode, values=("Off", "On"), state="readonly", width=7)
        combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 4), pady=2)
        combo.bind("<<ComboboxSelected>>", lambda _e, p=prefix: self._commit_capability(p))
        ttk.Button(line, text="Browse", width=9, style="DetailAction.TButton",
                   command=lambda p=prefix: self._action(f"{p}_pick")).grid(row=0, column=2, padx=2, pady=2, ipady=3)
        ttk.Button(line, text="Clear", width=9, style="DetailAction.TButton",
                   command=lambda p=prefix: self._action(f"{p}_clear")).grid(row=0, column=3, padx=(2, 0), pady=2, ipady=3)
        if prefix == "mtp":
            # Keep the speculative draft budget next to MTP so it is visible
            # and editable without the model-list context menu.
            ttk.Label(line, text="n-max").grid(row=0, column=4, sticky=tk.W,
                                                padx=(8, 2), pady=2)
            n_max = tk.StringVar(value="7")
            self._vars["mtp_n_max"] = n_max
            entry = ttk.Spinbox(line, from_=1, to=16, textvariable=n_max,
                                width=5, state="normal")
            entry.grid(row=0, column=5, sticky=tk.W, padx=(2, 4), pady=2)
            self._inputs["mtp_n_max"] = entry
            entry.bind("<Return>", lambda _e: self._commit_mtp_n_max())
            entry.bind("<FocusOut>", lambda _e: self._commit_mtp_n_max())
            self._add_status_entry(line, 0, "mtp_status", column=6)
            line.columnconfigure(6, weight=1)
        else:
            self._add_status_entry(line, 0, f"{prefix}_status", column=4)

    def _add_cpu_moe_row(self, parent, row):
        # This lives beside GPU layers in Quick, so use its parent grid
        # directly: mode/input line up with GPU / Auto / Recommend.
        ttk.Label(parent, text="CPU-MoE").grid(row=row, column=0, sticky=tk.W, pady=2)
        mode = tk.StringVar(value="GPU all")
        self._vars["cpu_moe_mode"] = mode
        combo = ttk.Combobox(parent, textvariable=mode, values=("GPU all", "CPU all", "First N"), state="readonly", width=9)
        combo.grid(row=row, column=1, sticky=tk.W, padx=(8, 2), pady=2)
        self._inputs["cpu_moe_mode"] = combo
        combo.bind("<<ComboboxSelected>>", lambda _e: self._commit_cpu_moe())
        ttk.Label(parent, text="Layers").grid(row=row, column=2, sticky=tk.W, padx=(8, 2), pady=2)
        layers = tk.StringVar(value="0")
        self._vars["cpu_moe_layers"] = layers
        entry = ttk.Entry(parent, textvariable=layers, width=9)
        entry.grid(row=row, column=3, sticky=tk.W, padx=(2, 0), pady=2)
        self._inputs["cpu_moe_layers"] = entry
        entry.bind("<Return>", lambda _e: self._commit_cpu_moe())
        entry.bind("<FocusOut>", lambda _e: self._commit_cpu_moe())

    def _add_ngram_row(self, parent, row):
        """N-gram speculative decoding — no draft file, stacks with MTP/DFlash."""
        line = self._advanced_row(parent, row, 4)
        ttk.Label(line, text="N-gram", width=10).grid(row=0, column=0, sticky=tk.W, pady=2)
        mode = tk.StringVar(value="Off")
        self._vars["ngram_mode"] = mode
        combo = ttk.Combobox(line, textvariable=mode, values=tuple(NGRAM_LABELS),
                             state="readonly", width=7)
        combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 4), pady=2)
        self._inputs["ngram_mode"] = combo
        combo.bind("<<ComboboxSelected>>", lambda _e: self._commit_ngram())
        # Align with the status entry on the Vision / MTP / DFlash rows.
        self._add_status_entry(line, 0, "ngram_status", column=4)
        self._vars["ngram_status"].set("free speed-up, stacks with MTP")

    def _add_backend_sampling_row(self, parent, row):
        """Backend sampling — part of the vendor recipe for integrated-MTP
        models (Qwen3.x); samples on the GPU for both target and draft."""
        line = self._advanced_row(parent, row, 4)
        ttk.Label(line, text="Bknd sample", width=10).grid(row=0, column=0, sticky=tk.W, pady=2)
        mode = tk.StringVar(value="Off")
        self._vars["backend_sampling"] = mode
        combo = ttk.Combobox(line, textvariable=mode, values=("Off", "On"),
                             state="readonly", width=7)
        combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 4), pady=2)
        self._inputs["backend_sampling"] = combo
        combo.bind("<<ComboboxSelected>>", lambda _e: self._commit_backend_sampling())
        self._add_status_entry(line, 0, "backend_sampling_status", column=4)
        self._vars["backend_sampling_status"].set(
            "experimental; recommended with MTP (Qwen3.x)")

    def _add_metrics_row(self, parent, row):
        """Prometheus metrics endpoint -- what external monitors poll."""
        line = self._advanced_row(parent, row, 4)
        ttk.Label(line, text="Metrics", width=10).grid(
            row=0, column=0, sticky=tk.W, pady=2)
        mode = tk.StringVar(value="Off")
        self._vars["metrics"] = mode
        combo = ttk.Combobox(line, textvariable=mode, values=("Off", "On"),
                             state="readonly", width=7)
        combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 4), pady=2)
        self._inputs["metrics"] = combo
        combo.bind("<<ComboboxSelected>>", lambda _e: self._commit_metrics())
        self._add_status_entry(line, 0, "metrics_status", column=4)
        self._vars["metrics_status"].set(
            "Prometheus /metrics endpoint for external monitors")

    def _commit_metrics(self):
        if self._loading or not self._profile_name:
            return
        self._on_change(self._profile_name, {
            "server.metrics": self._vars["metrics"].get() == "On"})

    def _commit_backend_sampling(self):
        if self._loading or not self._profile_name:
            return
        self._on_change(self._profile_name, {
            "server.backend_sampling": self._vars["backend_sampling"].get() == "On"})

    def _commit_ngram(self):
        """Persist the n-gram choice; "Off" disables the track entirely."""
        if self._loading or not self._profile_name:
            return
        spec = NGRAM_LABELS.get(self._vars["ngram_mode"].get(), "")
        self._on_change(self._profile_name, {
            "ngram_enabled": bool(spec),
            "ngram_type": spec or DEFAULT_NGRAM_TYPE,
        })

    def _commit_capability(self, prefix):
        if self._loading or not self._profile_name:
            return
        self._on_change(self._profile_name, {f"{prefix}_enabled": self._vars[f"{prefix}_enabled"].get() == "On"})

    def _commit_mtp_n_max(self):
        if self._loading or not self._profile_name:
            return
        try:
            value = int(self._vars["mtp_n_max"].get())
        except (TypeError, ValueError):
            value = 0
        if not 1 <= value <= 16:
            if self._hint is not None:
                self._hint.config(text="MTP n-max must be an integer from 1 to 16.",
                                  foreground=theme.AMBER)
            return
        self._on_change(self._profile_name, {"mtp_n_max": value})

    def _commit_vision(self):
        if self._loading or not self._profile_name:
            return
        if self._vars["vision_enabled"].get() == "On":
            if self._vars["vision_status"].get() == "Not attached":
                self._action("vision_pick")
        else:
            self._action("vision_clear")

    def _commit_cpu_moe(self):
        if self._loading or not self._profile_name:
            return
        mode = self._vars["cpu_moe_mode"].get()
        try:
            layers = max(0, int(self._vars["cpu_moe_layers"].get()))
        except ValueError:
            return
        if mode == "CPU all":
            updates = {"cpu_moe": True, "n_cpu_moe": 0}
        elif mode == "First N":
            updates = {"cpu_moe": False, "n_cpu_moe": max(1, layers)}
        else:
            updates = {"cpu_moe": False, "n_cpu_moe": 0}
        self._on_change(self._profile_name, updates)

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        for child in widget.winfo_children(): self._bind_mousewheel(child)

    def _on_mousewheel(self, event):
        if not (self.winfo_rootx() <= event.x_root <= self.winfo_rootx()+self.winfo_width() and self.winfo_rooty() <= event.y_root <= self.winfo_rooty()+self.winfo_height()): return
        units = -int(event.delta / 120) if event.delta else 0
        if getattr(event, "state", 0) & 0x0001: self._canvas.xview_scroll(units, "units")
        else: self._canvas.yview_scroll(units, "units")
        return "break"

    def _commit(self, key, preset=False):
        if self._loading or not self._profile_name: return
        if preset: return self._on_preset(self._profile_name, self._vars[key].get())
        paths = {"ctx": ("inference.ctx_size", lambda v: int(float(v)*1024)), "gpu": ("inference.gpu_layers", int), "threads": ("inference.n_threads", int), "batch": ("inference.n_batch", int), "ubatch": ("inference.n_ubatch", lambda v: max(0, int(v))), "parallel": ("inference.n_parallel", int), "seed": ("inference.seed", int), "max_tokens": ("inference.n_predict", lambda v: max(0, int(v))), "reasoning_budget": ("inference.reasoning_budget", lambda v: None if not str(v).strip() else int(v)), "reasoning_budget_msg": ("inference.reasoning_budget_message", str), "reasoning_effort": ("inference.reasoning_effort", normalize_reasoning_effort), "temp": ("sampling.temperature", float), "topk": ("sampling.top_k", int), "topp": ("sampling.top_p", float), "minp": ("sampling.min_p", lambda v: None if not str(v).strip() else float(v)), "repeat": ("sampling.repeat_penalty", float), "frequency": ("sampling.frequency_penalty", float), "presence": ("sampling.presence_penalty", float), "kv": ("kv_cache", lambda v: {"F16":"f16", "Q8":"q8_0", "Q4":"q4_0"}.get(v, v.lower())), "flash": ("server.flash_attn", str), "reasoning": ("reasoning", normalize_reasoning)}
        try: path, conv = paths[key]; self._on_change(self._profile_name, {path: conv(self._vars[key].get())})
        except (KeyError, TypeError, ValueError):
            if self._hint is not None:
                self._hint.config(text=f"Invalid value for {key}.", foreground=theme.AMBER)

    def _action(self, action):
        if self._profile_name: self._on_action(self._profile_name, action)

    def _set_gpu_auto(self):
        self._vars["gpu"].set("-1")
        self._commit("gpu")

    def set_profile(self, profile):
        self._loading = True
        try:
            self._profile_name = profile.profile_name; self._title.config(text=profile.display_name or profile.profile_name)
            e = estimate_vram(profile)
            # The projector is a separate file, so showing it inside "weights"
            # would hide 600 MB of a vision profile's footprint.
            vision = (f"  •  vision {e.vision_mb/1024:.1f} GB"
                      if getattr(e, "vision_mb", 0) else "")
            self._summary.config(text=f"Estimated VRAM {e.total_mb/1024:.1f} GB  •  weights {e.weights_mb/1024:.1f} GB  •  KV {e.kv_mb/1024:.1f} GB  •  reserve {e.overhead_mb/1024:.1f} GB{vision}")
            i, s = profile.inference, profile.sampling
            ctx_k = str(i.ctx_size // 1024)
            if ctx_k not in ("32", "64", "128", "256", "512"):
                ctx_k = min(("32", "64", "128", "256", "512"), key=lambda x: abs(int(x) - i.ctx_size // 1024))
            vals = {"ctx": ctx_k, "gpu": i.gpu_layers, "kv": {"q8_0":"Q8", "q4_0":"Q4"}.get(profile.kv_cache, "F16"), "threads": i.n_threads, "flash": profile.server.flash_attn, "reasoning": normalize_reasoning(profile.reasoning), "temp": s.temperature, "topk": s.top_k, "topp": s.top_p, "minp": "" if getattr(s, "min_p", None) is None else s.min_p, "repeat": s.repeat_penalty, "seed": i.seed, "frequency": s.frequency_penalty, "presence": s.presence_penalty, "batch": i.n_batch, "ubatch": getattr(i, "n_ubatch", 0) or 0, "parallel": i.n_parallel, "max_tokens": i.n_predict or 0, "reasoning_budget": "" if i.reasoning_budget is None else i.reasoning_budget, "reasoning_budget_msg": i.reasoning_budget_message or "", "reasoning_effort": normalize_reasoning_effort(i.reasoning_effort)}
            for k, v in vals.items(): self._vars[k].set(v)
            self._vars["backend_sampling"].set(
                "On" if getattr(profile.server, "backend_sampling", False) else "Off")
            self._vars["metrics"].set(
                "On" if getattr(profile.server, "metrics", False) else "Off")
            vision = next((f for f in profile.extra_files if "mmproj" in f.lower() or "clip" in f.lower()), "")
            self._vars["vision_enabled"].set("On" if vision else "Off")
            self._vars["vision_status"].set(Path(vision).name if vision else "Not attached")
            self._vars["mtp_enabled"].set("On" if profile.mtp_enabled else "Off")
            self._vars["mtp_n_max"].set(str(max(1, min(16, int(getattr(profile, "mtp_n_max", 7) or 7)))))
            mtp_status = profile.mtp_model or ("Native (in-model)" if profile.mtp_native else "Not attached")
            self._vars["mtp_status"].set(Path(mtp_status).name if profile.mtp_model else mtp_status)
            self._vars["dflash_enabled"].set("On" if profile.dflash_enabled else "Off")
            dflash_status = profile.dflash_model or "Not attached"
            self._vars["dflash_status"].set(Path(dflash_status).name if profile.dflash_model else dflash_status)
            if profile.cpu_moe:
                cpu_moe_mode, cpu_moe_layers = "CPU all", 0
            elif profile.n_cpu_moe:
                cpu_moe_mode, cpu_moe_layers = "First N", profile.n_cpu_moe
            else:
                cpu_moe_mode, cpu_moe_layers = "GPU all", 0
            self._vars["cpu_moe_mode"].set(cpu_moe_mode)
            self._vars["cpu_moe_layers"].set(cpu_moe_layers)
            ngram_spec = normalize_ngram_type(getattr(profile, "ngram_type", "")) \
                if getattr(profile, "ngram_enabled", False) else ""
            self._vars["ngram_mode"].set(ngram_label(ngram_spec))
            self._vars["ngram_status"].set(
                ngram_spec if ngram_spec else "free speed-up, stacks with MTP")
        finally: self._loading = False
