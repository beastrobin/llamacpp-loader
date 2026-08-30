# llamacpp-loader × FreeToken 弹性显存设计草案

> Status: draft / proposal
> Author: research note for Robin (beastrobin)
> Date: 2026-08-28

## 0. 背景与结论速览

FreeToken 的 "弹性显存" 本质：
- 运行时在不重启的前提下，于 **专家缓存 ↔ KV cache** 之间动态再分配 VRAM；
- 后台进程抢显存时热缩 GPU 缓存、把计算交给 CPU，零停机、不 OOM。

llama.cpp 的硬约束（必须先讲清，避免误判）：
- 权重在 **启动时一次性载入 VRAM**，运行期改 `--gpu-layers` 必须重启进程。
- 因此 **无法 100% 复刻 FreeToken 的"真·零停机热缩"**。

可借鉴的正确姿势：**借其神而非其形** —— 做
**VRAM 感知的主动预算（启动前）+ 抢占安全降级（运行期）**，
用 loader 已有的重启链路实现"秒级停机重启"式的准零停机退化。

---

## 1. 现状盘点（代码已有什么）

| 能力 | 位置 | 现状 |
|------|------|------|
| 后台资源监控 loop | `gui/app.py` ~L364 | 已有，读 RAM/VRAM 显示到状态栏 |
| 资源读取 | `gui/app.py` `_read_resource()` ~L1778 | 读 `nvidia-smi`，**仅展示，不动作** |
| 注册期 VRAM 预算 | `scripts/register_all_models.py` L36/208 | 按 24GB 5090 算预算，但**只用于注册提示** |
| 启动层数与预算 | `process_manager/manager.py` `_build_command()` | 启动用 `gpu_layers=-1`（全 offload 自动），**未落预算** |
| 崩溃自动重启 | `process_manager/manager.py` `ProcessWatcher` | 现成，可复用做降级重启 |
| MoE / MTP 识别 | `config/store.py` / `metadata.py` | 已有 GGUF 元数据识别，可做 MoE 感知 |

**缺口一句话**：监控有了、重启有了、预算逻辑有了，但三者彼此独立 ——
监控不驱动动作，预算不进启动参数，MoE 识别没变成用户提示。

---

## 2. 设计目标

- **G1 启动不 OOM**：即便后台有游戏 / 渲染占着显存，也能算出安全 `--gpu-layers` 再启动。
- **G2 运行期抢占自愈**：后台进程抢显存逼近 OOM 时，自动降级（减层重启）而非崩溃。
- **G3 MoE 感知**：利用已有 MoE 识别，提示"激活参数小、可留余量换更长上下文"。

---

## 3. 方案（分阶段）

### Phase 0 — NVML 监控替代 nvidia-smi
- 引入 `pynvml`（optional extra `nvml`），子秒级轮询，免 shell 解析、免中文路径坑。
- 复用现有 monitor loop 槽位，把 `_read_resource()` 的 `nvidia-smi` 分支换掉。

### Phase 1 — 启动前 VRAM 预算（关 G1）
- 启动前 query **free VRAM** → 估算可 offload 层数 → 写显式 `--gpu-layers`（不再用 `-1`）。
- 公式（示意）：
  ```
  safe_layers = min(total_layers,
                    floor((free_vram_mb - kv_headroom_mb) / per_layer_vram_mb))
  ```
- 与 `register_all_models.py` 的预算逻辑对齐，避免两处各算各的。

### Phase 2 — 运行期抢占看门狗（核心借鉴，关 G2）
- monitor 检测 `free_vram < 阈值`（如 < 1.5 GB）时触发：
  1. 状态栏红色告警："显存被后台进程抢占"；
  2. 自动 `restart(减层配置)`（复用 `ProcessWatcher` 重启路径）；
  3. 可选：先尝试 `nvidia-smi` 调优先级再决定要不要重启。
- **说明**：这是"秒级停机重启"，不是 FreeToken 的真·零停机，但体验接近，且链路已现成。

### Phase 3 — MoE 感知提示（关 G3）
- 若模型为 MoE（已有识别）：提示每 token 激活参数小，可把部分层留给 CPU 换更长上下文 / 更大 KV。
- GUI 上加 "MoE 余量建议" 标签或启动前 toast。

### Phase 4（探索）— KV / 层 权衡滑块 + 跨卡 RPC
- 暴露 `ctx_size ↔ gpu-layers` 权衡 UI（显存有限时直观取舍）。
- 预留 llama.cpp `--rpc` 跨 GPU 接入（若未来上双卡；注：FreeToken 多卡 TP 也不支持，不必对标）。

---

## 4. 约束与风险

- **依赖**：`pynvml` 作为 optional extra 引入，保持核心零硬依赖（符合现有 `pyproject.toml` 风格）。
- **Windows 优先**：`pynvml` 在 Windows 可用，且比 `nvidia-smi` 解析更可靠。
- **真零停机不可达**：llama.cpp 架构限制，必须明确告知用户"降级 = 短暂重启"。
- **与 FreeToken 本质差异**：FreeToken 权重常驻 RAM 可热缩；llama.cpp 权重在 VRAM，重启才重排。本方案是"工程近似"，不是同机制。
- **代码规范**：新增 `*.py` 保持零 CJK、commit 全英文（现有项目铁律）。

---

## 5. 验收（done 标准）

- [ ] 启动前自动算 safe `--gpu-layers`，后台占显存也能启动不 OOM
- [ ] 运行中手动抢显存 → 自动降级重启且服务恢复
- [ ] MoE 模型在启动前显示"余量建议"
- [ ] NVML 监控在 Windows / Linux 均可用，无 shell 解析故障

---

## 附：与 FreeToken 的能力映射

| FreeToken 能力 | llamacpp-loader 对应做法 | 可达性 |
|----------------|--------------------------|--------|
| 运行时热缩 VRAM | 抢占看门狗 + 减层重启 | 近似（秒级停机） |
| 启动前显存预算 | Phase 1 显式 gpu-layers | 完全可达 |
| 弹性 KV / 层分配 | Phase 4 权衡滑块 | 部分可达 |
| 权重常驻 RAM 热换 | llama.cpp 不支持 | 不可达 |
| 后台抢占零停机 | 减层重启 | 近似 |
