"""Regression tests for GGUF-aware VRAM budgeting."""

from llamacpp_loader.config.budget import estimate_vram
from llamacpp_loader.config.store import ModelProfile


def test_kv_estimate_uses_gqa_shape_and_quant_bytes(tmp_path):
    model = tmp_path / "model-q4.gguf"
    model.write_bytes(b"0" * (1024 * 1024))
    profile = ModelProfile(
        profile_name="model",
        model_path=str(tmp_path),
        gguf_file=model.name,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        kv_cache="q4_0",
    )
    profile.inference.ctx_size = 4096
    profile.inference.n_parallel = 2

    estimate = estimate_vram(profile, gpu_layers=0)
    expected = 4096 * 2 * 32 * 8 * 128 * 2 * 0.5625 / 1024**2

    assert estimate.kv_mb == expected
    assert estimate.confidence == "metadata"


def test_legacy_profile_falls_back_to_rough_kv_estimate(tmp_path):
    model = tmp_path / "legacy.gguf"
    model.write_bytes(b"0")
    profile = ModelProfile(
        profile_name="legacy",
        model_path=str(tmp_path),
        gguf_file=model.name,
        n_layers=4,
        kv_cache="f16",
    )

    estimate = estimate_vram(profile, gpu_layers=0)

    assert estimate.kv_mb == 4096 * 1 * 4 * 2 * 2 / 1024**2
    assert estimate.confidence == "rough"


def test_kv_metadata_roundtrips_in_profile_dict():
    profile = ModelProfile(profile_name="meta", n_kv_heads=8, head_dim=128)

    restored = ModelProfile.from_dict(profile.to_dict())

    assert restored.n_kv_heads == 8
    assert restored.head_dim == 128
