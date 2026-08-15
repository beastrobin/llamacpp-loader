# llamacpp-loader

**Local LLM launcher** — a Tkinter-based graphical manager for running llama.cpp servers.

It folds the whole "pick a model → tune params → launch → test" workflow you used to do with hand-written `.bat` files and sticky notes into a single GUI:

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Tests](https://img.shields.io/badge/tests-70%20passed-brightgreen)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Model management** | Add GGUF models via a file picker; each gets its own auto-generated `ModelProfile` (independent parameters per model) |
| 🔎 **Auto-discovery** | `scan_models()` recursively scans a directory, detects every `.gguf` and builds a readable config |
| ⚙️ **Parameter tuning** | Launch params (ctx / GPU layers / threads) + sampling params (temperature / top_k / top_p) with a visual editor |
| 💾 **Parameter persistence** | Each model saves its own JSON config; switching models never overwrites another |
| 🚀 **One-click launch** | Select model → Start → auto-spawns llama-server → health check passes → **opens the Web UI automatically** |
| 🧪 **Smoke test** | `SmokeTestRunner` checks server health + `scripts/smoke_live.py` measures real throughput (tokens/s) |
| 🛑 **Graceful shutdown** | Stops the process, releases VRAM/RAM, and supports crash auto-restart (watchdog) |

## 📦 Installation

```bash
# Requires Python 3.10+ (official build, includes tkinter)
git clone https://github.com/beastrobin/llamacpp-loader
cd llamacpp-loader

# Core runtime needs NO third-party deps (pure stdlib: tkinter / subprocess / json / threading).
# Optional: `pip install gguf` enables automatic MoE / MTP metadata detection when scanning models.
python -m llamacpp_loader.main     # launch the GUI
```

> ⚠️ Note: `llama-server.exe` (the llama.cpp binary itself) is **not** included in this repo. Download it from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) page.

## 🚀 Usage

1. **Add a model**: click `Browse...` to pick a GGUF file, or point it at a directory to auto-scan
2. **Tune params**: adjust parameters in the Inference / Sampling panel (ctx_size, gpu_layers, temperature, etc.)
3. **Save**: parameters auto-attach to the current model profile; each model stays independent
4. **Launch**: click `Start` → the app spawns llama-server → health check → browser opens automatically
5. **Smoke test**: run `python scripts/smoke_live.py` to measure server status and generation speed
6. **Stop**: click `Stop` to shut down gracefully, or let the watchdog auto-restart on crash

## 🏗️ Architecture

```
llamacpp-loader/
├── llamacpp_loader/
│   ├── main.py                    # Entry point: launches the Tkinter app
│   ├── config/store.py            # ModelProfile collection store (per-model params, JSON persistence)
│   │   ├── ModelProfile           # model + ServerParams + InferenceParams + SamplingParams
│   │   ├── ConfigStore            # CRUD + validation + default template + scan_models auto-discovery
│   │   └── ServerParams/...       # host/port, ctx/gpu/threads, temp/top_k/top_p
│   ├── config/recommend.py        # Preset recommendations / baselines
│   ├── config/metadata.py         # GGUF metadata reader (MoE / MTP detection)
│   ├── process_manager/manager.py # Subprocess lifecycle management
│   │   ├── ProcessManager         # Popen wrapper + state machine (idle→starting→running→stopping)
│   │   ├── ServerConfig           # Builds llama-server CLI args from a Profile
│   │   └── ProcessWatcher         # Background thread watches for crashes and auto-restarts
│   ├── smoke_test/runner.py       # Post-launch health validation
│   │   ├── SmokeTestRunner        # /health first, falls back to /v1/models on 404
│   │   └── ServerHealthChecker    # Synchronous single-request check (for pytest)
│   ├── gui/app.py                 # Tkinter GUI (MainWindow/ParameterPanel/ConsolePanel/StatusBar/ControlBar)
│   └── gui/theme.py               # Dark theme styling (colors, fonts, ttk styles)
├── scripts/
│   ├── smoke_live.py              # Live smoke test: health check + throughput (tokens/s)
│   ├── register_all_models.py     # Batch-register GGUF models (MoE/MTP auto-detect)
│   └── e2e_smoke_check.py         # Headless end-to-end Start -> Smoke Test check
├── tests/                         # 70 pytest tests
│   ├── test_config_store.py       # CRUD/validation/persistence/scan_models
│   ├── test_process_manager.py    # lifecycle/command-build/browser
│   ├── test_smoke_test.py         # result construction/endpoint checks
│   ├── test_gui.py                # component creation/state (shared Tk root fixture)
│   ├── test_kv_edit_unlock.py     # KV/ctx/gpu/threads editing vs sampling lock
│   └── test_overlay_double_click.py # changed-cell overlay double-click detection
├── models/                        # Local GGUF model directory
└── logs/                          # Runtime logs
```

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v          # 70 tests, stable across consecutive runs
python scripts/smoke_live.py   # live smoke test (requires llama-server running on 8080)
```

## 📝 Design notes

- **One parameter set per model**: `ModelProfile` binds the model file + launch params + sampling params together, eliminating "switch models and lose my params"
- **Thread safety**: `ConfigStore` uses a Lock, `ProcessManager` locks all state changes, and GUI callbacks run through `root.after()`
- **Robust process management**: graceful SIGTERM + SIGKILL on timeout + background watchdog for crash self-healing
- **Core runtime is dependency-free**: pure Python standard library, works out of the box. An **optional** `gguf` package (`pip install gguf`) auto-detects MoE / MTP model metadata during `scan_models()` — without it the tool still launches and runs, just without that metadata enhancement.

## 📄 License

MIT
