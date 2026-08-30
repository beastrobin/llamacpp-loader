"""Conservative VRAM estimates used by the model detail panel."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VramEstimate:
    total_mb: float
    weights_mb: float
    kv_mb: float
    overhead_mb: float
    confidence: str = "rough"


def _file_size_mb(profile) -> float:
    import os
    try:
        return os.path.getsize(os.path.join(profile.model_path, profile.gguf_file)) / 1024**2
    except OSError:
        return 0.0


def estimate_vram(profile, *, gpu_layers=None) -> VramEstimate:
    """Estimate model VRAM from file size, context, KV type, and offload ratio."""
    size = _file_size_mb(profile)
    layers = max(0, int(getattr(profile.inference, "gpu_layers", -1) if gpu_layers is None else gpu_layers))
    total_layers = max(1, int(getattr(profile, "n_layers", 0) or layers or 1))
    ratio = 1.0 if getattr(profile.inference, "gpu_layers", -1) < 0 and gpu_layers is None else min(1.0, layers / total_layers)
    weights = size * ratio
    ctx = max(64, int(getattr(profile.inference, "ctx_size", 4096)))
    parallel = max(1, int(getattr(profile.inference, "n_parallel", 1)))
    kv_kind = str(getattr(profile, "kv_cache", "f16") or "f16").lower()
    bytes_per = 1 if "q4" in kv_kind else 1.5 if "q8" in kv_kind else 2
    kv = ctx * parallel * max(1, total_layers) * 2 * bytes_per / 1024**2
    overhead = max(256.0, weights * 0.08)
    return VramEstimate(weights + kv + overhead, weights, kv, overhead,
                        "metadata" if size else "rough")


def recommend_gpu_layers(profile, free_mb: float):
    """Return the largest conservative layer count that fits in free VRAM."""
    total = int(getattr(profile, "n_layers", 0) or 0)
    if total <= 0 or free_mb <= 0:
        return None
    for layers in range(total, -1, -1):
        if estimate_vram(profile, gpu_layers=layers).total_mb <= max(0, free_mb - 512):
            return layers
    return 0
