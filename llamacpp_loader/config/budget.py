"""Conservative VRAM estimates used by the model detail panel."""
from __future__ import annotations

from dataclasses import dataclass

#: llama.cpp reports its own worst case for a vision projector separately from
#: the weights, and for the measured one that came out at 873 MB against a
#: 600 MB file: the vision encoder needs activation space on top of the tensors
#: it loads.  Applies to the first (only) projector the launcher will pass.
_VISION_ENCODER_FACTOR = 1.45


@dataclass(frozen=True)
class VramEstimate:
    total_mb: float
    weights_mb: float
    kv_mb: float
    overhead_mb: float
    vision_mb: float = 0.0
    confidence: str = "rough"


def _file_size_mb(profile) -> float:
    import os
    try:
        return os.path.getsize(os.path.join(profile.model_path, profile.gguf_file)) / 1024**2
    except OSError:
        return 0.0


def _vision_mb(profile) -> float:
    """VRAM the vision projector will take, if the profile declares one.

    The launcher picks the projector out of ``extra_files`` by name and passes
    it as ``--mmproj``, so the estimate mirrors that rule (first match wins)
    instead of charging every extra file.  A 600 MB projector is far too big to
    leave out of a "will it fit" number.

    Scaled by :data:`_VISION_ENCODER_FACTOR` because the weights are not the
    whole story: llama.cpp reports its own worst case separately, and for the
    measured projector that came out 45% above the file size.
    """
    import os
    model_path = getattr(profile, "model_path", "") or ""
    for entry in getattr(profile, "extra_files", None) or []:
        if isinstance(entry, dict):
            continue
        name = str(entry)
        lowered = name.lower()
        if "mmproj" not in lowered and "clip" not in lowered:
            continue
        path = name if os.path.isabs(name) else os.path.join(model_path, name)
        try:
            size = os.path.getsize(path) / 1024**2
        except OSError:
            return 0.0
        return size * _VISION_ENCODER_FACTOR
    return 0.0


def estimate_vram(profile, *, gpu_layers=None) -> VramEstimate:
    """Estimate model VRAM from file size, context, KV type, and offload ratio.

    Hybrid (linear-attention) architectures only keep a KV cache on the blocks
    that attend, so the cache is charged to ``kv_layers`` -- not to every
    block.  Charging all of them overstates the cache by the attention ratio
    (4x on Ternary-Bonsai-2, 16x on a Nemotron-H style sparse mask) and made a
    configuration that fits look impossible.

    The fixed cost term covers the CUDA context, the compute buffers and the
    recurrent state of hybrid architectures; the vision projector, when the
    profile declares one, is charged separately as ``vision_mb``.
    """
    size = _file_size_mb(profile)
    vision = _vision_mb(profile)
    layers = max(0, int(getattr(profile.inference, "gpu_layers", -1) if gpu_layers is None else gpu_layers))
    total_layers = max(1, int(getattr(profile, "n_layers", 0) or layers or 1))
    ratio = 1.0 if getattr(profile.inference, "gpu_layers", -1) < 0 and gpu_layers is None else min(1.0, layers / total_layers)
    weights = size * ratio
    ctx = max(64, int(getattr(profile.inference, "ctx_size", 4096)))
    parallel = max(1, int(getattr(profile.inference, "n_parallel", 1)))
    kv_kind = str(getattr(profile, "kv_cache", "f16") or "f16").lower()
    # Include the actual KV tensor shape when GGUF metadata is available.
    # The fractional values include the per-block scale overhead of llama.cpp
    # cache formats (q4_0 ~= 0.5625 and q8_0 ~= 1.125 bytes/element).
    bytes_per = 0.5625 if "q4" in kv_kind else 1.125 if "q8" in kv_kind else 2
    kv_heads = max(0, int(getattr(profile, "n_kv_heads", 0) or 0))
    head_dim = max(0, int(getattr(profile, "head_dim", 0) or 0))
    kv_layers = max(0, int(getattr(profile, "kv_layers", 0) or 0))
    if kv_heads and head_dim:
        # No kv_layers recorded (older profile, or the GGUF could not be read)
        # -> assume every block attends, the historical behaviour.
        attn_layers = min(kv_layers, total_layers) if kv_layers else total_layers
        kv_elements = ctx * parallel * max(1, attn_layers) * kv_heads * head_dim * 2
        confidence = "metadata" if size else "rough"
    else:
        # Legacy profiles may not have the optional GGUF metadata. Preserve a
        # deliberately rough estimate rather than inventing a model shape.
        kv_elements = ctx * parallel * max(1, total_layers) * 2
        confidence = "rough"
    kv = kv_elements * bytes_per / 1024**2
    # Fixed (non-KV) VRAM cost: the CUDA context, compute buffers and the
    # recurrent state of hybrid architectures.  Calibrated against llama.cpp's
    # OWN buffer report ("projected to use N MiB of device memory"), not against
    # nvidia-smi peaks, so no desktop-baseline guesswork is baked in.  Measured
    # on an RTX 4080 Laptop, -b 1024, one slot, q4_0 KV, 27B model:
    #     131072 ctx -> 870 MB fixed    (5395 model + 2304 KV + 870 = 8569)
    #      65536 ctx -> 549 MB fixed    (6540 model + 1152 KV + 549 = 8241)
    # It is the compute buffer that dominates, and that grows with the context
    # window rather than with the weight count -- the previous
    # "weights * 8%, floor 256" produced 454 MB here, ~400 MB short, which is
    # what let recommend_gpu_layers() propose configurations that then died on
    # "failed to allocate" on a 12 GB card.  Large-context models cannot
    # realistically be packed tighter than this.
    overhead = 228.0 + ctx * 0.0049
    return VramEstimate(weights + kv + overhead + vision, weights, kv,
                        overhead, vision, confidence)


def recommend_gpu_layers(profile, free_mb: float):
    """Return the largest conservative layer count that fits in free VRAM."""
    total = int(getattr(profile, "n_layers", 0) or 0)
    if total <= 0 or free_mb <= 0:
        return None
    for layers in range(total, -1, -1):
        if estimate_vram(profile, gpu_layers=layers).total_mb <= max(0, free_mb - 512):
            return layers
    return 0
