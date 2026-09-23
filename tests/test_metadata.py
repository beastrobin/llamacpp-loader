"""Tests for the local GGUF metadata reader.

The reader has two paths: a dependency-free header scanner (primary) and the
optional ``gguf`` package (fallback).  These tests pin the behaviour that made
models like PrismML's Ternary-Bonsai-2-27B readable at all -- its tensors use
ggml type ids 142/143, which ``gguf`` rejects outright.
"""

import struct

import pytest

from llamacpp_loader.config.metadata import (
    _read_gguf_meta_raw,
    read_gguf_meta,
)

# ggml metadata value types used by the writer below.
T_UINT32 = 4
T_INT32 = 5
T_STRING = 8
T_ARRAY = 9


def _str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, vtype: int, value) -> bytes:
    """Encode one metadata entry.  Arrays are ``(inner_type, items)``."""
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == T_STRING:
        return out + _str(value)
    if vtype == T_UINT32:
        return out + struct.pack("<I", value)
    if vtype == T_INT32:
        return out + struct.pack("<i", value)
    if vtype == T_ARRAY:
        inner, items = value
        out += struct.pack("<I", inner) + struct.pack("<Q", len(items))
        for item in items:
            if inner == T_UINT32:
                out += struct.pack("<I", item)
            elif inner == T_STRING:
                out += _str(item)
            else:  # pragma: no cover - guards a typo in a test
                raise AssertionError(f"unsupported inner type {inner}")
        return out
    raise AssertionError(f"unsupported value type {vtype}")


def _tensor(name: str, dims, ggml_type: int, offset: int = 0) -> bytes:
    out = _str(name) + struct.pack("<I", len(dims))
    for dim in dims:
        out += struct.pack("<Q", dim)
    return out + struct.pack("<I", ggml_type) + struct.pack("<Q", offset)


def _write_gguf(path, kvs, tensors) -> None:
    """Write a GGUF v3 header carrying metadata and tensor infos only."""
    data = b"GGUF" + struct.pack("<I", 3)
    data += struct.pack("<Q", len(tensors))
    data += struct.pack("<Q", len(kvs))
    data += b"".join(kvs)
    data += b"".join(tensors)
    path.write_bytes(data)


def _base_kvs(arch: str = "qwen35") -> list:
    return [
        _kv("general.architecture", T_STRING, arch),
        _kv(f"{arch}.block_count", T_UINT32, 4),
        _kv(f"{arch}.context_length", T_UINT32, 32768),
    ]


class TestRawGgufScan:

    def test_reads_basic_scalar_metadata(self, tmp_path):
        path = tmp_path / "model.gguf"
        _write_gguf(
            path,
            _base_kvs() + [
                _kv("qwen35.attention.head_count_kv", T_UINT32, 4),
                _kv("qwen35.attention.key_length", T_UINT32, 256),
            ],
            [_tensor("blk.0.attn_q.weight", (256, 256), 2)],
        )

        meta = read_gguf_meta(path)

        assert meta["ok"] is True
        assert meta["arch"] == "qwen35"
        assert meta["n_layers"] == 4
        assert meta["context_length"] == 32768
        assert meta["n_kv_heads"] == 4
        assert meta["head_dim"] == 256
        assert meta["is_moe"] is False

    def test_expert_count_marks_moe(self, tmp_path):
        path = tmp_path / "moe.gguf"
        _write_gguf(
            path,
            _base_kvs() + [_kv("qwen35.expert_count", T_UINT32, 256)],
            [_tensor("blk.0.ffn_gate_exps.weight", (256, 256), 2)],
        )

        meta = read_gguf_meta(path)

        assert meta["expert_count"] == 256
        assert meta["is_moe"] is True

    @pytest.mark.parametrize("ggml_type", [142, 143])
    def test_unknown_tensor_type_still_yields_metadata(self, tmp_path, ggml_type):
        """Regression: PrismML's ternary ids 142/143 broke the whole read.

        ``gguf`` raises ``ValueError: np.uint32(142) is not a valid
        GGMLQuantizationType``, which left every field zeroed -- so the loader
        stored n_layers=0 / context_length=0 and mis-sized the model.
        """
        path = tmp_path / "ternary-bonsai.gguf"
        _write_gguf(
            path,
            _base_kvs() + [
                _kv("qwen35.attention.head_count_kv", T_UINT32, 4),
                _kv("qwen35.attention.key_length", T_UINT32, 256),
            ],
            [_tensor("output.weight", (2560, 5120), ggml_type)],
        )

        raw = _read_gguf_meta_raw(path)

        assert raw is not None
        assert raw["n_layers"] == 4
        assert raw["context_length"] == 32768
        assert raw["n_kv_heads"] == 4
        assert raw["head_dim"] == 256

    def test_per_layer_head_count_uses_largest_entry(self, tmp_path):
        """Hybrid models report one entry per layer; layer 0 is often 0.

        Nemotron-H ships ``[0, 0, 0, 0, 0, 2, 0, ...]`` -- only some layers
        attend.  Taking the first entry (0) or falling back to the *query*
        head count would mis-size the KV cache, so the largest entry wins.
        """
        path = tmp_path / "hybrid.gguf"
        _write_gguf(
            path,
            _base_kvs("nemotron_h_moe") + [
                _kv("nemotron_h_moe.attention.head_count", T_UINT32, 32),
                _kv("nemotron_h_moe.attention.head_count_kv", T_ARRAY,
                    (T_UINT32, [0, 0, 0, 0, 0, 2, 0, 0])),
                _kv("nemotron_h_moe.attention.key_length", T_UINT32, 128),
            ],
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )

        meta = read_gguf_meta(path)

        assert meta["n_kv_heads"] == 2

    def test_empty_head_count_kv_falls_back_to_head_count(self, tmp_path):
        path = tmp_path / "empty-array.gguf"
        _write_gguf(
            path,
            _base_kvs() + [
                _kv("qwen35.attention.head_count", T_UINT32, 8),
                _kv("qwen35.attention.head_count_kv", T_ARRAY, (T_UINT32, [])),
            ],
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )

        meta = read_gguf_meta(path)

        assert meta["n_kv_heads"] == 8

    def test_layer_types_flag_mtp_support(self, tmp_path):
        path = tmp_path / "mtp-layers.gguf"
        _write_gguf(
            path,
            _base_kvs() + [
                _kv("qwen35.attention.layer_types", T_ARRAY,
                    (T_STRING, ["attention", "mtp"])),
            ],
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )

        assert read_gguf_meta(path)["mtp_supported"] is True

    def test_nextn_tensor_marks_native_mtp(self, tmp_path):
        path = tmp_path / "native-mtp.gguf"
        _write_gguf(
            path,
            _base_kvs(),
            [
                _tensor("blk.0.attn_q.weight", (128, 128), 2),
                _tensor("blk.4.nextn.eh_proj.weight", (128, 128), 2),
            ],
        )

        assert read_gguf_meta(path)["mtp_native"] is True

    def test_no_native_mtp_without_block_count(self, tmp_path):
        """A missing block_count must not turn layer 0 into an MTP head.

        ``blk.{n_layers}.`` with n_layers=0 becomes ``blk.0.``, which matches
        the first layer of *every* model, so MTP used to be enabled spuriously
        whenever the block count could not be read.
        """
        path = tmp_path / "no-block-count.gguf"
        _write_gguf(
            path,
            [_kv("general.architecture", T_STRING, "qwen35")],
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )

        meta = read_gguf_meta(path)

        assert meta["ok"] is True
        assert meta["n_layers"] == 0
        assert meta["mtp_native"] is False

    def test_non_gguf_file_reports_not_ok(self, tmp_path):
        path = tmp_path / "not-a-model.gguf"
        path.write_bytes(b"this is definitely not a GGUF file, not at all!!")

        meta = read_gguf_meta(path)

        assert meta["ok"] is False
        assert meta["n_layers"] == 0

    def test_truncated_header_reports_not_ok(self, tmp_path):
        path = tmp_path / "truncated.gguf"
        # Valid magic and counts, but the payload is cut off.
        path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
                         + struct.pack("<Q", 4) + b"\x10\x00")

        assert read_gguf_meta(path)["ok"] is False

    def test_utf8_metadata_names_survive(self, tmp_path):
        path = tmp_path / "unicode.gguf"
        _write_gguf(
            path,
            _base_kvs() + [_kv("qwen35.name", T_STRING, "mod\u00e8le-\u4e2d\u6587")],
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )

        assert read_gguf_meta(path)["ok"] is True


class TestKvLayerCount:
    """``kv_layers`` -- how many blocks actually hold a KV cache.

    Hybrid architectures attend on a subset of blocks, so charging the cache to
    every block overstates VRAM by the attention ratio.  Ternary-Bonsai-2-27B is
    ``qwen35``: 64 blocks, full attention every 4th -> 16 blocks, i.e. 4x less
    KV than the naive count.
    """

    def _write(self, path, arch="qwen35", blocks=64, extra=()):
        _write_gguf(
            path,
            [
                _kv("general.architecture", T_STRING, arch),
                _kv(f"{arch}.block_count", T_UINT32, blocks),
                _kv(f"{arch}.context_length", T_UINT32, 262144),
            ] + list(extra),
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )
        return path

    def test_attention_interval_divides_the_layer_count(self, tmp_path):
        path = self._write(
            tmp_path / "qwen35-hybrid.gguf",
            extra=[
                _kv("qwen35.attention.head_count_kv", T_UINT32, 4),
                _kv("qwen35.attention.key_length", T_UINT32, 256),
                _kv("qwen35.full_attention_interval", T_UINT32, 4),
            ],
        )

        meta = read_gguf_meta(path)

        assert meta["n_layers"] == 64
        assert meta["kv_layers"] == 16

    def test_dense_model_counts_every_layer(self, tmp_path):
        path = self._write(
            tmp_path / "qwen35-dense.gguf",
            extra=[
                _kv("qwen35.attention.head_count_kv", T_UINT32, 4),
                _kv("qwen35.attention.key_length", T_UINT32, 256),
            ],
        )

        assert read_gguf_meta(path)["kv_layers"] == 64

    def test_interval_covering_every_layer_is_ignored(self, tmp_path):
        """interval 1 means "every block attends" -- same as no stride at all."""
        path = self._write(
            tmp_path / "interval-one.gguf",
            extra=[_kv("qwen35.full_attention_interval", T_UINT32, 1)],
        )

        assert read_gguf_meta(path)["kv_layers"] == 64

    def test_absurd_interval_cannot_zero_the_cache(self, tmp_path):
        """A stride larger than the model must not divide the cache away."""
        path = self._write(
            tmp_path / "interval-huge.gguf",
            extra=[_kv("qwen35.full_attention_interval", T_UINT32, 4096)],
        )

        assert read_gguf_meta(path)["kv_layers"] == 64

    def test_zero_heads_array_counts_attending_layers(self, tmp_path):
        """Nemotron-H: ``0`` marks a block with no KV cache at all."""
        path = self._write(
            tmp_path / "nemotron.gguf", arch="nemotron_h_moe", blocks=8,
            extra=[
                _kv("nemotron_h_moe.attention.head_count", T_UINT32, 32),
                _kv("nemotron_h_moe.attention.head_count_kv", T_ARRAY,
                    (T_UINT32, [0, 0, 0, 0, 0, 2, 0, 0])),
            ],
        )

        meta = read_gguf_meta(path)

        assert meta["n_kv_heads"] == 2
        assert meta["kv_layers"] == 1

    def test_zero_when_no_block_count(self, tmp_path):
        path = tmp_path / "no-blocks.gguf"
        _write_gguf(
            path,
            [_kv("general.architecture", T_STRING, "qwen35")],
            [_tensor("blk.0.attn_q.weight", (128, 128), 2)],
        )

        assert read_gguf_meta(path)["kv_layers"] == 0
