# llamacpp-loader

**本地大模型加载器** —— 一个用 Tkinter 编写的 llama.cpp 服务器图形化管理工具。

把你以前靠手写 bat、记参数的"选模型 → 调参 → 启动 → 测试"流程，全部收进一个 GUI：

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Tests](https://img.shields.io/badge/tests-60%20passed-brightgreen)

---

## ✨ 功能

| 功能 | 说明 |
|---|---|
| 🎯 **模型管理** | 文件选择器添加 GGUF 模型，自动生成 `ModelProfile`（每模型一套独立参数） |
| 🔎 **自动发现** | `scan_models()` 递归扫描目录，自动识别所有 `.gguf` 并生成可读配置 |
| ⚙️ **参数调节** | 启动参数（ctx / GPU 层数 / 线程数）+ 采样参数（temperature / top_k / top_p）可视化调整 |
| 💾 **参数持久化** | 每个模型独立保存一套 JSON 配置，切换模型互不覆盖 |
| 🚀 **一键启动** | 选择模型 → Start → 自动拉起 llama-server → 健康检查通过 → **自动打开 Web UI** |
| 🧪 **冒烟测试** | `SmokeTestRunner` 检查 server 健康 + `scripts/smoke_live.py` 实测测速（tokens/s） |
| 🛑 **优雅关闭** | 停止进程、释放显存/内存，支持崩溃自动重启（watchdog） |

## 📦 安装

```bash
# 需要 Python 3.10+（官方版，含 tkinter）
git clone https://github.com/beastrobin/llamacpp-loader
cd llamacpp-loader

# 零第三方依赖（全 stdlib：tkinter / subprocess / json / threading）
python -m llamacpp_loader.main     # 启动 GUI
```

> ⚠️ 注意：`llama-server.exe`（llama.cpp 本体）不包含在本仓库，请从 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 自行获取。

## 🚀 使用流程

1. **添加模型**：点击 `Browse...` 选择 GGUF 模型文件，或输入目录让程序自动扫描
2. **调参**：在 Inference / Sampling 面板调整参数（ctx_size、gpu_layers、temperature 等）
3. **保存**：参数自动关联当前模型 profile，切换模型时各自独立
4. **启动**：点击 `Start` → 程序拉起 llama-server → 健康检查 → 浏览器自动打开
5. **冒烟测试**：运行 `python scripts/smoke_live.py` 实测服务器状态和生成速度
6. **停止**：点击 `Stop` 优雅关闭，或崩溃后自动重启

## 🏗️ 架构

```
llamacpp-loader/
├── llamacpp_loader/
│   ├── main.py                    # 入口：启动 Tkinter 应用
│   ├── config/store.py            # ModelProfile 集合存储（每模型一套参数，JSON 持久化）
│   │   ├── ModelProfile           # 模型 + ServerParams + InferenceParams + SamplingParams
│   │   ├── ConfigStore            # CRUD + 参数校验 + 默认模板 + scan_models 自动发现
│   │   └── ServerParams/...       # host/port、ctx/gpu/threads、temp/top_k/top_p
│   ├── process_manager/manager.py # 子进程生命周期管理
│   │   ├── ProcessManager         # Popen 封装 + 状态机（idle→starting→running→stopping）
│   │   ├── ServerConfig           # 从 Profile 构建 llama-server 命令行参数
│   │   └── ProcessWatcher         # 后台线程监控崩溃自动重启
│   ├── smoke_test/runner.py       # 启动后健康验证
│   │   ├── SmokeTestRunner        # /health 优先，404 回退 /v1/models
│   │   └── ServerHealthChecker    # 同步单请求检查（供 pytest 用）
│   └── gui/app.py                 # Tkinter GUI（MainWindow/ParameterPanel/ConsolePanel/StatusBar/ControlBar）
├── scripts/
│   └── smoke_live.py              # 真机冒烟测试：健康检查 + 测速（tokens/s）
├── tests/                         # 60 个 pytest 测试
│   ├── test_config_store.py       # 27 个：CRUD/校验/持久化/扫描
│   ├── test_process_manager.py    # 15 个：生命周期/命令构建/浏览器
│   ├── test_smoke_test.py         # 8 个：结果构造/端点
│   └── test_gui.py                # 6 个：组件创建/状态（共享 Tk root fixture）
├── models/                        # 本地 GGUF 模型目录
└── logs/                          # 运行时日志
```

## 🧪 测试

```bash
pip install pytest
pytest tests/ -v          # 60 tests，10 次连跑稳定
python scripts/smoke_live.py   # 真机冒烟测试（需 llama-server 在 8080 运行）
```

## 📝 设计要点

- **每模型一套参数**：ModelProfile 把模型文件 + 启动参数 + 采样参数绑定，杜绝"换个模型参数被覆盖"
- **线程安全**：ConfigStore 用 Lock、ProcessManager 状态变更全加锁、GUI 回调走 `root.after()`
- **健壮的进程管理**：优雅 SIGTERM + 超时 SIGKILL + 后台 watchdog 崩溃自愈
- **零依赖**：全 Python 标准库，开箱即用

## 🤖 关于本项目

本项目由 **本地大模型（Qwen3.6-35B-A3B / Nemotron-3.5-30B-A3B，跑在 RTX 5090 + llama.cpp）通过 Hermes Agent 自主开发**，人工负责需求评审与验收。从一个"验证本地模型写代码能力"的实验，长成了一个真正能用的工具。

## 📄 License

MIT
