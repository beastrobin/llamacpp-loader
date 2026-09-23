"""Local GGUF metadata reader for autonomous model capability detection.

This module lets the loader recognise a model's properties directly from the
GGUF file -- no external agent, no network, no manual lookup:

* ``is_moe``       -- True when the model uses Mixture-of-Experts
                      (``<arch>.expert_count`` > 0).
* ``mtp_supported`` -- True when the base model was trained with Multi-Token
                      Prediction layers (``<arch>.attention.layer_types``
                      contains "mtp"), i.e. it can be sped up with an MTP
                      draft model.
* ``n_layers`` / ``context_length`` / ``n_kv_heads`` / ``head_dim`` /
  ``kv_layers`` -- used for context-window budgeting.  ``kv_layers`` counts the
  blocks that keep a KV cache, which is fewer than ``n_layers`` on hybrid
  (linear-attention) architectures.

It is deliberately dependency-light: ``gguf`` (and its ``numpy`` dependency)
are imported lazily so the GUI never crashes on a machine that does not have
them installed -- it simply skips enrichment.  Run ``pip install gguf`` to
enable full autonomous detection.
"""
from __future__ import annotations

import os
import re
import struct

from pathlib import Path
from typing import Any, Optional


def _field(reader: Any, name: str, default: Any = None) -> Any:
    """Read a decoded GGUF metadata value, tolerating missing / odd types."""
    f = reader.fields.get(name)
    if f is None:
        return default
    try:
        val = f.contents()
    except Exception:  # noqa: BLE001
        return default
    # Normalise numpy / bytes / list-of-bytes into plain Python types.
    if hasattr(val, "item"):  # numpy scalar
        try:
            return val.item()
        except Exception:  # noqa: BLE001
            return default
    if isinstance(val, bytes):
        return val.decode("utf-8", "ignore")
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", "ignore"))
            elif hasattr(x, "item"):
                try:
                    out.append(x.item())
                except Exception:  # noqa: BLE001
                    pass
            else:
                out.append(x)
        return out
    return val


# ----------------------------------------------------------------- raw GGUF scan

#: Metadata lives at the head of a GGUF.  64 MiB comfortably covers even a
#: 250k-token vocabulary; reading it in one shot keeps the parse to pure
#: offset arithmetic (no per-entry syscalls).
_GGUF_HEAD_BYTES = 64 << 20
#: Bounds for plausibility checks -- a truncated or corrupt header must be
#: rejected instead of allocating gigabytes.
_GGUF_MAX_STRING = 1 << 20
_GGUF_MAX_ELEMS = 1 << 26

#: Upper bound on how many entries of a *wanted* array are materialised.
_GGUF_MAX_KEEP_ELEMS = 1 << 12

#: ggml metadata value types 0-12 (the GGUF spec set).
_GGUF_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_GGUF_T_STRING = 8
_GGUF_T_ARRAY = 9

#: Metadata keys this module consumes.  Everything else -- above all the
#: 250k-entry tokenizer vocabularies and merge tables that dominate GGUF
#: metadata size -- is skipped without materialising a single object.
_GGUF_WANTED_EXACT = ("general.architecture",)
_GGUF_WANTED_SUFFIXES = (
    ".block_count",
    ".context_length",
    ".attention.head_count_kv",
    ".attention.head_count",
    ".attention.key_length",
    ".attention.value_length",
    ".full_attention_interval",
    ".expert_count",
    ".attention.layer_types",
)


class _GgufScanError(Exception):
    """Header truncated or malformed -- the caller should fall back."""


class _GgufCursor:
    """Offset-based reader over an in-memory GGUF header.

    Parsing from one ``bytes`` object rather than a file handle is what makes
    skipping cheap: advancing past a 250k-entry string array costs a Python
    integer bump instead of a seek + read per entry.
    """

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def skip(self, n: int) -> None:
        if n < 0 or self.pos + n > len(self.data):
            raise _GgufScanError("truncated header")
        self.pos += n

    def _take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise _GgufScanError("truncated header")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return int(struct.unpack("<I", self._take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self._take(8))[0])

    def string(self, *, keep: bool = True) -> str:
        n = self.u64()
        if n > _GGUF_MAX_STRING:
            raise _GgufScanError("implausible string length")
        if not keep:
            self.skip(n)
            return ""
        return self._take(n).decode("utf-8", "ignore")

    def value(self, vtype: int, *, keep: bool = True) -> Any:
        """Read one metadata value, or skip it when *keep* is False."""
        if vtype == _GGUF_T_STRING:
            return self.string(keep=keep)
        if vtype == _GGUF_T_ARRAY:
            inner = self.u32()
            count = self.u64()
            if count > _GGUF_MAX_ELEMS:
                raise _GgufScanError("implausible array count")
            if inner in _GGUF_SCALARS:
                # Fixed-width elements.  Wanted arrays are materialised in
                # full -- per-layer metadata such as ``attention.head_count_kv``
                # varies per layer and the caller needs the spread, not just
                # the first entry.  They are small (one entry per layer), but
                # the cap keeps a corrupt count from allocating unboundedly.
                fmt, size = _GGUF_SCALARS[inner]
                take = count if (keep and count <= _GGUF_MAX_KEEP_ELEMS) else 0
                items = [struct.unpack(fmt, self._take(size))[0]
                         for _ in range(take)]
                self.skip(size * (count - take))
                return items
            items: list[Any] = []
            for _ in range(count):
                val = self.value(inner, keep=keep)
                if keep:
                    items.append(val)
            return items
        if vtype in _GGUF_SCALARS:
            fmt, size = _GGUF_SCALARS[vtype]
            raw = self._take(size)
            if not keep:
                return None
            return struct.unpack(fmt, raw)[0]
        raise _GgufScanError(f"unknown metadata value type {vtype}")


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a decoded metadata value to ``int``, tolerating odd types."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _layer_metric(primary: Any, fallback: Any) -> int:
    """Pick a per-layer metric (KV-head count, head width) from GGUF metadata.

    ``attention.head_count_kv`` is a plain scalar on most models but an array
    with one entry per layer on hybrid architectures: Nemotron-H reports
    ``[0, 0, 0, 0, 0, 2, 0, ...]`` (only some layers attend at all) and the
    Gemma-4 sliding-window build reports ``[8, 8, 8, 8, 8, 1, ...]``.  The
    **largest** entry is what the KV cache must fit: the first entry is
    meaningless (Nemotron-H starts at 0), and falling back to the *query* head
    count on such a model overestimates the cache by 16x.

    A present but empty array carries no information, so *fallback* is used.
    """
    if isinstance(primary, (list, tuple)):
        if not primary:
            return _as_int(fallback)
        return max((_as_int(v) for v in primary), default=0)
    if primary is None:
        return _as_int(fallback)
    return _as_int(primary)


def _kv_layer_count(n_layers: int, kv_heads: Any, interval: Any = 0) -> int:
    """Number of blocks that actually keep a growing KV cache.

    Hybrid architectures -- now the norm rather than the exception -- do not
    attend on every layer, and charging the cache to every block overestimates
    VRAM by the attention ratio (4x on a 4:1 model such as Ternary-Bonsai-2):

    * ``<arch>.attention.head_count_kv`` becomes a per-layer array with ``0``
      for blocks that hold no cache at all (Nemotron-H reports
      ``[0, 0, 0, 0, 0, 2, 0, ...]``) -- count the non-zero entries.
    * Alternating full/linear attention declares its stride explicitly as
      ``<arch>.full_attention_interval`` (``Ternary-Bonsai-2-27B`` is
      ``qwen35`` with interval 4, so a 64-block model attends on 16 blocks).

    Anything else -- including a missing or nonsensical stride -- keeps a cache
    on every layer, which is the historical assumption.
    """
    if n_layers <= 0:
        return 0
    if isinstance(kv_heads, (list, tuple)) and kv_heads:
        used = sum(1 for v in kv_heads if _as_int(v) > 0)
        if used:
            return used
        return n_layers
    stride = _as_int(interval)
    if 1 < stride <= n_layers:
        # Full attention lands on the last block of every stride window.
        return len(range(stride - 1, n_layers, stride))
    return n_layers


def _read_gguf_meta_raw(path: str | Path) -> Optional[dict]:
    """Read capability metadata straight from the GGUF header bytes.

    Dependency-free and deliberately tolerant: tensor *types* are recorded but
    never looked up, so a model quantised with a ggml type the installed
    tooling does not know (e.g. PrismML's ternary ids 142/143, or any id added
    after this build) still yields arch / layer count / context length.  The
    ``gguf`` package raises on such files, which is why models like
    ``Ternary-Bonsai-2-27B`` used to come back with every field zeroed.

    Returns ``None`` when the file is not a readable GGUF v2/v3 -- callers
    should then fall back to the ``gguf`` package.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(min(size, _GGUF_HEAD_BYTES))
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"GGUF":
        return None

    cur = _GgufCursor(head)
    try:
        cur.skip(4)                       # magic
        version = cur.u32()
        if version not in (2, 3):
            return None                   # v1 laid the header out differently
        n_tensors = cur.u64()
        n_kv = cur.u64()

        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = cur.string()
            vtype = cur.u32()
            wanted = (key in _GGUF_WANTED_EXACT
                      or key.endswith(_GGUF_WANTED_SUFFIXES))
            meta[key] = cur.value(vtype, keep=wanted)

        tensor_names: list[str] = []
        for _ in range(n_tensors):
            tensor_names.append(cur.string())
            n_dims = cur.u32()
            cur.skip(8 * n_dims)          # dimensions
            cur.skip(4)                   # ggml type id (not validated here)
            cur.skip(8)                   # data offset
    except _GgufScanError:
        return None

    arch: Any = meta.get("general.architecture") or "unknown"
    if not isinstance(arch, str):
        arch = str(arch)

    n_layers = _as_int(meta.get(f"{arch}.block_count"))
    context_length = _as_int(meta.get(f"{arch}.context_length"))
    kv_heads_raw = meta.get(f"{arch}.attention.head_count_kv")
    n_kv_heads = _layer_metric(kv_heads_raw,
                               meta.get(f"{arch}.attention.head_count"))
    head_dim = _layer_metric(meta.get(f"{arch}.attention.key_length"),
                             meta.get(f"{arch}.attention.value_length"))
    expert_count = _as_int(meta.get(f"{arch}.expert_count"))
    kv_layers = _kv_layer_count(n_layers, kv_heads_raw,
                                meta.get(f"{arch}.full_attention_interval"))

    layer_types = meta.get(f"{arch}.attention.layer_types")
    mtp_supported = False
    if layer_types:
        if isinstance(layer_types, (list, tuple)):
            joined = " ".join(str(t) for t in layer_types)
        else:
            joined = str(layer_types)
        mtp_supported = "mtp" in joined.lower()

    # Native MTP head: an extra block at index block_count (blk.<n>.*) or a
    # "nextn" sub-block.  Without a known block count that test is meaningless
    # -- "blk.0." would match layer 0 of any model -- so restrict it to an
    # explicit "nextn" tensor in that case.
    if n_layers > 0:
        blk_prefix = f"blk.{n_layers}."
        blk_head_nextn = f"blk.{n_layers - 1}.nextn."
        mtp_native = any(
            name.startswith(blk_prefix)
            or name.startswith(blk_head_nextn)
            or "nextn" in name.lower()
            for name in tensor_names
        )
    else:
        mtp_native = any("nextn" in name.lower() for name in tensor_names)

    return {
        "arch": arch,
        "n_layers": n_layers,
        "context_length": context_length,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "kv_layers": kv_layers,
        "expert_count": expert_count,
        "is_moe": expert_count > 0,
        "mtp_supported": mtp_supported,
        "mtp_native": mtp_native,
    }


def read_gguf_meta(path: str | Path) -> dict:
    """Return capability metadata read directly from a GGUF file.

    Returns a dict with keys: arch, n_layers, context_length, n_kv_heads,
    head_dim, kv_layers, expert_count, is_moe, mtp_supported, mtp_native, ok.
    ``ok`` is False when the file could not be read (not a GGUF, or no reader
    could parse it) -- callers should treat missing capabilities gracefully.
    """
    result: dict[str, Any] = {
        "arch": "",
        "n_layers": 0,
        "context_length": 0,
        "n_kv_heads": 0,
        "head_dim": 0,
        "kv_layers": 0,
        "expert_count": 0,
        "is_moe": False,
        "mtp_supported": False,
        "mtp_native": False,
        "ok": False,
    }

    # Preferred path: dependency-free header scan.  Works without the optional
    # gguf package at all, tolerates tensor types it does not know, and skips
    # the multi-megabyte tokenizer arrays -- a 27B model is read in
    # milliseconds where gguf-python needs ~8 s.
    raw = _read_gguf_meta_raw(path)
    if raw is not None:
        result.update(raw)
        result["ok"] = True
        return result

    # Fallback: the optional gguf package, for layouts the raw scanner rejects.
    try:
        from gguf import GGUFReader  # lazy import -- optional dependency
    except Exception:  # noqa: BLE001
        return result

    try:
        reader = GGUFReader(str(path), "r")
    except Exception:  # noqa: BLE001
        return result

    try:
        arch = _field(reader, "general.architecture") or "unknown"
        if not isinstance(arch, str):
            arch = str(arch)
        result["arch"] = arch

        n_layers = _field(reader, f"{arch}.block_count") or 0
        ctx = _field(reader, f"{arch}.context_length") or 0
        # Per-layer arrays are common here too (see _layer_metric): passing a
        # list straight to int() raised, and because this whole block shares
        # one try/except, that single key silently zeroed every field after it
        # -- head_dim, expert_count, is_moe and all MTP detection.
        kv_heads_raw = _field(reader, f"{arch}.attention.head_count_kv")
        n_kv_heads = _layer_metric(
            kv_heads_raw,
            _field(reader, f"{arch}.attention.head_count"))
        head_dim = _layer_metric(
            _field(reader, f"{arch}.attention.key_length"),
            _field(reader, f"{arch}.attention.value_length"))
        experts = _field(reader, f"{arch}.expert_count") or 0

        result["n_layers"] = _as_int(n_layers)
        result["context_length"] = _as_int(ctx)
        result["n_kv_heads"] = n_kv_heads
        result["head_dim"] = head_dim
        result["expert_count"] = _as_int(experts)
        result["is_moe"] = result["expert_count"] > 0
        # Re-read from the normalised value: a per-layer array would otherwise
        # be used verbatim in the "blk.<n>." prefix test below.
        n_layers = result["n_layers"]
        result["kv_layers"] = _kv_layer_count(
            n_layers, kv_heads_raw,
            _field(reader, f"{arch}.full_attention_interval"))

        # MTP support: llama.cpp stores per-layer types; a "mtp" entry means
        # the model carries Multi-Token Prediction heads that an MTP draft can
        # accelerate.
        layer_types = _field(reader, f"{arch}.attention.layer_types")
        if layer_types:
            if isinstance(layer_types, (list, tuple)):
                joined = " ".join(str(t) for t in layer_types)
            else:
                joined = str(layer_types)
            result["mtp_supported"] = "mtp" in joined.lower()

        # Native MTP: some GGs bundle the MTP draft head *inside* the model
        # itself as an extra <arch>.<block_count>.* tensor block (e.g. the
        # empero-ai Qwen3.8-27B-Ridge GGUF ships blk.64.* / nextn tensors).
        # Such models run with ``--spec-type draft-mtp`` and NO external draft
        # file, so we must detect the in-model head to enable it.
        try:
            # A native MTP head appears as an extra block indexed by the
            # block_count (blk.<n>.*) or as a "nextn" sub-block
            # (blk.<n>.nextn.* / nextn.*).  Cover both the "block_count
            # excludes the head" and "block_count includes it" conventions
            # (e.g. empero-ai Ridge: block_count=65, head lives at blk.64 with
            # blk.64.nextn.* tensors).  A block count of 0 means the field was
            # unreadable, and "blk.0." would then match layer 0 of every model.
            blk_prefix = f"blk.{n_layers}." if n_layers else ""
            blk_head_nextn = f"blk.{n_layers - 1}.nextn." if n_layers else ""
            for t in reader.tensors:
                name = t.name
                if ((blk_prefix and name.startswith(blk_prefix))
                        or (blk_head_nextn and name.startswith(blk_head_nextn))
                        or "nextn" in name.lower()):
                    result["mtp_native"] = True
                    break
        except Exception:  # noqa: BLE001
            # Non-fatal: partial read still returns the other capabilities.
            pass
    except Exception:  # noqa: BLE001
        # Partial read is still useful (e.g. arch + is_moe may be set).
        pass

    result["ok"] = True
    return result


# Quantisation suffixes that may trail a model stem, e.g. '-q4_k_m', '-f16'.
_QUANT_RE = re.compile(
    r"-(?:iq?[0-9](?:_[a-z0-9]+)?|q[0-9](?:_[a-z0-9]+)?|f16|f32|bpw)$",
    re.IGNORECASE,
)


def _strip_quant(stem: str) -> str:
    """Drop a trailing quantisation token so two quantisations of the same
    model resolve to the same core name (e.g. '…-q4_k_m' and '…-q4_0')."""
    return _QUANT_RE.sub("", stem)


# Families that only ever ship as sparse MoE, matched as substrings.
_MOE_NAME_HINTS = (
    "mixtral", "-moe", "moe-", "deepseek-v3", "deepseek-r1",
    "glm-4.5", "glm-4.6", "glm-5", "glm-4.7",
)

# "<total>B-A<active>B" convention: 35b-a3b, 235b-a22b, 122b-a10b ...
_MOE_ACTIVE_RE = re.compile(r"(?:^|[-_])a\d+b(?![a-z0-9])", re.IGNORECASE)


def looks_moe_from_name(name: str | Path) -> bool:
    """Best-effort MoE detection straight from a model filename.

    Fallback used when the optional ``gguf`` package is missing (or the file
    could not be parsed), so ``read_gguf_meta`` cannot confirm ``expert_count``.
    Without this fallback every model is silently classified as dense, which
    disables the MoE-only features -- notably the --cpu-moe low-VRAM path.

    Recognises the ``<total>B-A<active>B`` convention (Qwen3.6-35B-A3B,
    DeepSeek-A22B, Qwen3.5-122B-A10B) plus family names that are only ever
    sparse.
    """
    s = Path(name).name if isinstance(name, Path) else (name or "")
    s = s.lower().replace(" ", "-")
    if not s:
        return False
    if any(h in s for h in _MOE_NAME_HINTS):
        return True
    return bool(_MOE_ACTIVE_RE.search(s))


def is_mtp_draft_filename(stem: str) -> bool:
    """True if *stem* is a speculative-decoding draft model (MTP / DFlash / EAGLE).

    Recognises the llama.cpp naming conventions in the wild:
      * prefix : 'mtp-Qwen3.6-35B-A3B-Q4_0', 'dflash-DeepSeek-...'
      * suffix : 'qwen3.6-27b-fable-mtp', 'xxx-dflash'
      * infix  : '...-mtp-...', '...-eagle3-...'

    Draft files are separate small GGUF files used as speculative decoders; they
    must NOT be registered as standalone base models.
    """
    s = stem.lower().replace(" ", "-")
    tokens = s.split("-")
    # Substring/prefix matching so "dflash2" (Inco AI's DFlash 2 draft) is
    # recognised in addition to exact "dflash", plus "mtp"/"eagle*" prefixes.
    for t in tokens:
        if t == "mtp" or t.startswith("mtp"):
            return True
        if "dflash" in t:
            return True
        if t.startswith("eagle"):
            return True
    return False


def base_stem_from_mtp(stem: str) -> str:
    """Recover the base-model stem from a speculative-decoding draft filename stem.

    Strips the leading/trailing draft-type token (mtp / dflash / eagle*).
    """
    s = stem.lower().replace(" ", "-")
    tokens = s.split("-")
    # Mirror is_mtp_draft_filename: drop any token that is/contains the
    # draft-type marker (mtp / dflash / dflash2 / eagle*), keeping the rest.
    cleaned = []
    for t in tokens:
        if t == "mtp" or t.startswith("mtp"):
            continue
        if "dflash" in t:
            continue
        if t.startswith("eagle"):
            continue
        cleaned.append(t)
    return "-".join(cleaned)


def find_mtp_draft(model_dir: str | Path, base_stem: str) -> Optional[str]:
    """Find a sibling MTP draft file for *base_stem* inside *model_dir*.

    Returns the draft filename if found, else None.  Handles both the
    ``mtp-<base>`` prefix and ``<base>-mtp`` suffix conventions, and tolerates
    differing quantisation between the base model and its draft.
    """
    d = Path(model_dir)
    if not d.is_dir():
        return None
    base = base_stem.lower().replace(" ", "-")
    base_core = _strip_quant(base)        # e.g. 'qwen3.6-35b-a3b'

    # Explicit candidates (both MTP and DFlash naming conventions).
    candidates = [
        f"{base}-mtp.gguf",               # suffix
        f"mtp-{base}.gguf",               # prefix (exact quant echo)
        f"mtp-{base_core}.gguf",          # prefix with quant stripped
        f"{base}-dflash.gguf",            # DFlash suffix
        f"dflash-{base}.gguf",            # DFlash prefix
        f"dflash-{base_core}.gguf",       # DFlash prefix, quant stripped
    ]
    for cand in candidates:
        if (d / cand).is_file():
            return cand

    # Fallback: scan every gguf; pair any MTP draft whose stripped base core
    # aligns with this model's core name (quantisation-agnostic).
    try:
        for p in d.glob("*.gguf"):
            s = p.stem.lower().replace(" ", "-")
            if not is_mtp_draft_filename(s):
                continue
            if s == base:
                # A model whose *own* filename carries an "mtp" token (e.g.
                # 'Qwen3.6-27B-...-NEO-MTP-IQ3_M.gguf') must never be paired
                # with itself as its draft. Skip self-matches outright.
                continue
            cand_base = base_stem_from_mtp(s)        # e.g. 'qwen3.6-35b-a3b-q4_0'
            cand_core = _strip_quant(cand_base)      # 'qwen3.6-35b-a3b'
            if (cand_core == base_core
                    or cand_core.startswith(base_core)
                    or base_core.startswith(cand_core)):
                return p.name
    except Exception:  # noqa: BLE001
        pass
    return None
