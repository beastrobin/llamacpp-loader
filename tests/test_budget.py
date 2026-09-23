"""Regression tests for GGUF-aware VRAM budgeting."""

import pytest

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


def test_hybrid_model_charges_only_attending_layers(tmp_path):
    """Ternary-Bonsai-2-27B: 64 blocks but only 16 keep a KV cache.

    Charging all 64 overstated the cache 4x (9.0 GiB instead of 2.25 GiB at
    131072 ctx / q4_0) and made a configuration that fits read as impossible.
    """
    model = tmp_path / "bonsai.gguf"
    model.write_bytes(b"0" * (1024 * 1024))
    profile = ModelProfile(
        profile_name="ternary-bonsai-2-27b-pq2_0",
        model_path=str(tmp_path),
        gguf_file=model.name,
        n_layers=64,
        kv_layers=16,
        n_kv_heads=4,
        head_dim=256,
        kv_cache="q4_0",
    )
    profile.inference.ctx_size = 131072

    estimate = estimate_vram(profile, gpu_layers=0)

    elements = 131072 * 1 * 16 * 4 * 256 * 2
    assert estimate.kv_mb == elements * 0.5625 / 1024**2

    naive = 131072 * 1 * 64 * 4 * 256 * 2 * 0.5625 / 1024**2
    assert estimate.kv_mb == pytest.approx(naive / 4)


def test_missing_kv_layers_keeps_the_old_all_blocks_behaviour(tmp_path):
    """Profiles scanned before kv_layers existed must not change value."""
    model = tmp_path / "old.gguf"
    model.write_bytes(b"0" * (1024 * 1024))
    profile = ModelProfile(
        profile_name="old", model_path=str(tmp_path), gguf_file=model.name,
        n_layers=64, n_kv_heads=4, head_dim=256, kv_cache="q8_0",
    )
    profile.inference.ctx_size = 32768

    estimate = estimate_vram(profile, gpu_layers=0)

    assert estimate.kv_mb == 32768 * 1 * 64 * 4 * 256 * 2 * 1.125 / 1024**2


def test_kv_layers_is_clamped_to_the_block_count(tmp_path):
    """A corrupt kv_layers must never estimate more cache than blocks exist."""
    model = tmp_path / "corrupt.gguf"
    model.write_bytes(b"0" * (1024 * 1024))
    profile = ModelProfile(
        profile_name="corrupt", model_path=str(tmp_path), gguf_file=model.name,
        n_layers=8, kv_layers=9999, n_kv_heads=4, head_dim=256, kv_cache="f16",
    )
    profile.inference.ctx_size = 4096

    estimate = estimate_vram(profile, gpu_layers=0)

    assert estimate.kv_mb == 4096 * 1 * 8 * 4 * 256 * 2 * 2 / 1024**2


def test_kv_layers_roundtrips_in_profile_dict():
    profile = ModelProfile(profile_name="hybrid", n_layers=64, kv_layers=16)

    restored = ModelProfile.from_dict(profile.to_dict())

    assert restored.kv_layers == 16


class TestFixedCost:
    """The fixed reserve is dominated by the compute buffer (context-driven).

    Values come from llama.cpp's own buffer report on an RTX 4080 Laptop at
    -b 1024 / one slot, never from nvidia-smi peaks, so no desktop baseline is
    baked into them.
    """

    def _profile(self, ctx):
        profile = ModelProfile(profile_name="m", n_layers=64, kv_layers=16,
                               n_kv_heads=4, head_dim=256, kv_cache="q4_0")
        profile.inference.ctx_size = ctx
        return profile

    def test_reserve_follows_the_context_window(self, monkeypatch):
        monkeypatch.setattr("llamacpp_loader.config.budget._file_size_mb",
                            lambda profile: 5671.0)

        big = estimate_vram(self._profile(131072), gpu_layers=0)
        small = estimate_vram(self._profile(65536), gpu_layers=0)

        # Measured: 870 MB at 131072 ctx, 549 MB at 65536 ctx.
        assert big.overhead_mb == pytest.approx(870.0, abs=1.0)
        assert small.overhead_mb == pytest.approx(549.0, abs=1.0)

    def test_reserve_is_not_a_percentage_of_the_weights(self, monkeypatch):
        """A bigger model at the same context must not inflate the reserve.

        The old weights*8% rule grew with the model size, which is the wrong
        shape: the compute buffers scale with batch and context, not weights.
        """
        monkeypatch.setattr("llamacpp_loader.config.budget._file_size_mb",
                            lambda profile: 10000.0)

        estimate = estimate_vram(self._profile(131072), gpu_layers=0)

        assert estimate.overhead_mb == pytest.approx(870.0, abs=1.0)

    def test_estimate_covers_the_llama_cpp_measured_total(self, monkeypatch):
        """Regression: the old formula said 8429 MiB for a config llama.cpp
        measured at 8569 MiB, so a fitting setup could read as a leak and a
        non-fitting one could be recommended."""
        monkeypatch.setattr("llamacpp_loader.config.budget._file_size_mb",
                            lambda profile: 5671.0)

        # No gpu_layers override: the whole model is offloaded, which is what
        # llama.cpp's 8569 MiB measurement was taken with.
        estimate = estimate_vram(self._profile(131072))

        assert estimate.total_mb >= 8569.0


class TestVisionProjector:
    def test_projector_is_charged_with_its_activation_margin(self, tmp_path):
        """A 600 MB projector is 873 MB of VRAM, per llama.cpp's own report."""
        projector = tmp_path / "mmproj-model-f16.gguf"
        projector.write_bytes(b"0" * (1024 * 1024))
        model = tmp_path / "m.gguf"
        model.write_bytes(b"0" * (1024 * 1024))

        plain = ModelProfile(profile_name="plain", model_path=str(tmp_path),
                             gguf_file=model.name)
        vision = ModelProfile(profile_name="vision", model_path=str(tmp_path),
                              gguf_file=model.name,
                              extra_files=[projector.name])

        with_vision = estimate_vram(vision, gpu_layers=0)
        without = estimate_vram(plain, gpu_layers=0)

        assert with_vision.vision_mb == pytest.approx(1.45, abs=0.01)
        assert with_vision.total_mb - without.total_mb == pytest.approx(1.45,
                                                                       abs=0.01)

    def test_other_extra_files_are_not_charged(self, tmp_path):
        """Only the projector is passed to llama-server; spare files are not."""
        model = tmp_path / "m.gguf"
        model.write_bytes(b"0" * (1024 * 1024))
        spare = tmp_path / "something-else.gguf"
        spare.write_bytes(b"0" * (1024 * 1024))

        profile = ModelProfile(profile_name="spare", model_path=str(tmp_path),
                               gguf_file=model.name, extra_files=[spare.name])

        assert estimate_vram(profile, gpu_layers=0).vision_mb == 0.0

    def test_missing_projector_does_not_break_the_estimate(self, tmp_path):
        model = tmp_path / "m.gguf"
        model.write_bytes(b"0" * (1024 * 1024))
        profile = ModelProfile(profile_name="gone", model_path=str(tmp_path),
                               gguf_file=model.name,
                               extra_files=["mmproj-deleted.gguf"])

        estimate = estimate_vram(profile, gpu_layers=0)

        assert estimate.vision_mb == 0.0
        assert estimate.total_mb > 0
