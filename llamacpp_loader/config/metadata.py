"""Local GGUF metadata reader for autonomous model capability detection.

This module lets the loader recognise a model's properties directly from the
GGUF file -- no external agent, no network, no manual lookup:

* ``is_moe``       -- True when the model uses Mixture-of-Experts
                      (``<arch>.expert_count`` > 0).
* ``mtp_supported`` -- True when the base model was trained with Multi-Token
                      Prediction layers (``<arch>.attention.layer_types``
                      contains "mtp"), i.e. it can be sped up with an MTP
                      draft model.
* ``n_layers`` / ``context_length`` -- used for context-window budgeting.

It is deliberately dependency-light: ``gguf`` (and its ``numpy`` dependency)
are imported lazily so the GUI never crashes on a machine that does not have
them installed -- it simply skips enrichment.  Run ``pip install gguf`` to
enable full autonomous detection.
"""
from __future__ import annotations

import re

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


def read_gguf_meta(path: str | Path) -> dict:
    """Return capability metadata read directly from a GGUF file.

    Returns a dict with keys: arch, n_layers, context_length, expert_count,
    is_moe, mtp_supported, ok.  ``ok`` is False when the file could not be
    read (e.g. ``gguf`` not installed or not a GGUF) -- callers should treat
    missing capabilities gracefully.
    """
    result: dict[str, Any] = {
        "arch": "",
        "n_layers": 0,
        "context_length": 0,
        "expert_count": 0,
        "is_moe": False,
        "mtp_supported": False,
        "mtp_native": False,
        "ok": False,
    }
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
        experts = _field(reader, f"{arch}.expert_count") or 0

        result["n_layers"] = int(n_layers) if n_layers else 0
        result["context_length"] = int(ctx) if ctx else 0
        result["expert_count"] = int(experts) if experts else 0
        result["is_moe"] = result["expert_count"] > 0

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
            # blk.64.nextn.* tensors).
            blk_prefix = f"blk.{n_layers}."
            blk_head_nextn = f"blk.{n_layers - 1}.nextn."
            for t in reader.tensors:
                name = t.name
                if (name.startswith(blk_prefix)
                        or name.startswith(blk_head_nextn)
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
    if "mtp" in tokens or "dflash" in tokens or any(t.startswith("eagle") for t in tokens):
        return True
    return False


def base_stem_from_mtp(stem: str) -> str:
    """Recover the base-model stem from a speculative-decoding draft filename stem.

    Strips the leading/trailing draft-type token (mtp / dflash / eagle*).
    """
    s = stem.lower().replace(" ", "-")
    tokens = s.split("-")
    cleaned = [t for t in tokens
               if t not in ("mtp", "dflash") and not t.startswith("eagle")]
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
            cand_base = base_stem_from_mtp(s)        # e.g. 'qwen3.6-35b-a3b-q4_0'
            cand_core = _strip_quant(cand_base)      # 'qwen3.6-35b-a3b'
            if (cand_core == base_core
                    or cand_core.startswith(base_core)
                    or base_core.startswith(cand_core)):
                return p.name
    except Exception:  # noqa: BLE001
        pass
    return None
