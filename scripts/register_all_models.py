"""Scan every GGUF language model under the llama.cpp bin dir, read its real
metadata (layers / KV heads / head dim / trained context), and register an
optimized ModelProfile into ConfigStore for each one.

Optimization goal: full GPU offload on a 24 GB RTX 5090 mobile, picking the
largest context window that still fits once model weights + KV cache are
accounted for.

Run:
    python scripts/register_all_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project importable regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from gguf import GGUFReader  # noqa: E402
from llamacpp_loader.config.metadata import (  # noqa: E402
    base_stem_from_mtp,
    find_mtp_draft,
    is_mtp_draft_filename,
    read_gguf_meta,
)
from llamacpp_loader.config.store import ConfigStore  # noqa: E402

# llama.cpp bin directory. Provide it via the LLAMACPP_BIN environment variable
# or the --bin command-line flag — the folder that contains llama-server.exe.
# (No machine-specific path is hard-coded here so the script stays portable.)

# ---- VRAM budget for a 24 GB RTX 5090 mobile -------------------------------
# user's verified settings from 2026-08-11 screenshot take priority (see
# EMPIRICAL table below). For unknown models we fall back to a dynamic budget.
#   * 35B-A3B MoE verified: 128K–256K depending on quant
#   * small/dense models empirically cap at 128K and often run best at 32K–64K
# Reserve 1 GB for CUDA ctx + Windows desktop overhead.
GPU_TOTAL_MB = 24 * 1024
RESERVE_MB = 1024
USABLE_MB = GPU_TOTAL_MB - RESERVE_MB
CTX_MIN = 32768            # fallback floor — known models use screenshot values
CTX_MAX = 262144         # 256K absolute ceiling on 24 GB (verified 35B hits it)

# mmproj pairing by model family (relative to the bin dir, must exist on disk)
# More specific families should come first.
MMPROJ = [
    ("qwen3.6-35b-a3b", r"models\HauhauCS\mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
    ("gemma-4-26b-a4b", r"models\google\mmproj-gemma-4-12B-it-QAT-BF16.gguf"),
    ("gemma-4-12b", r"models\google\mmproj-gemma-4-12B-it-QAT-BF16.gguf"),
]

KV_TYPES = [("f16", 2), ("q8_0", 1), ("q4_0", 0.5)]

# user's verified parameters from the 2026-08-11 "Local LLM Web Launcher" list.
# Each entry: (required stem substrings, ctx_size, kv_cache, reasoning, tag_hint)
# Order matters — put more specific matches first.
EMPIRICAL = [
    ("gemma-4-e4b", 131072, "q8_0", False, ""),
    ("qwen3.6-12b-thinking", 32768, "q8_0", True, "thinking"),
    ("gemma-4-12b-qat", 65536, "q8_0", False, "img"),
    ("qwen2.5-14b-instruct", 32768, "q8_0", False, ""),
    ("qwen3-14b-ablit-q4", 32768, "q8_0", False, ""),
    ("qwen2.5-14b-uncensored", 32768, "q8_0", False, ""),
    ("qwen3-14b-ablit-q5", 32768, "q8_0", False, ""),
    ("cydonia-24b", 32768, "q8_0", False, ""),
    ("gemma-4-26b-a4b", 65536, "q8_0", False, "img"),
    ("qwen3.6-27b-fable-neo", 32768, "q8_0", False, ""),
    ("qwen3.6-27b-fable-mtp", 32768, "q8_0", False, ""),
    ("qwen3.6-27b-abliterated", 32768, "q8_0", False, ""),
    ("qwen3.6-35b-a3b-iq2_m", 131072, "q8_0", False, "img"),
    ("qwen3.6-35b-a3b-iq3_m", 262144, "q8_0", False, "img"),
    ("qwen3.6-35b-a3b-q4_k_m", 262144, "q4_0", False, "img"),
]


def _lookup_empirical(stem: str):
    """Return (ctx_size, kv_cache, reasoning, tag_hint) if stem matches an
    entry in user's verified table, otherwise None."""
    s = stem.lower().replace(" ", "-")
    for pattern, ctx, kv, reasoning, tag in EMPIRICAL:
        parts = pattern.split("-")
        if all(p in s for p in parts):
            return ctx, kv, reasoning, tag
    return None


def _field(reader: GGUFReader, name, default=None):
    """Read a decoded metadata value via ReaderField.contents()."""
    f = reader.fields.get(name)
    if f is None:
        return default
    try:
        val = f.contents()
    except Exception:
        return default
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, bytes):
        return val.decode("utf-8", "ignore")
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", "ignore"))
            elif isinstance(x, np.integer):
                out.append(int(x))
            else:
                out.append(x)
        strs = [x for x in out if isinstance(x, str)]
        return strs[0] if strs else (out[0] if out else default)
    return val


def read_meta(path: Path):
    reader = GGUFReader(str(path), "r")
    arch = _field(reader, "general.architecture") or "unknown"
    n_layers = _field(reader, f"{arch}.block_count") or 0
    ctx_train = _field(reader, f"{arch}.context_length") or 4096
    n_kv = _field(reader, f"{arch}.attention.head_count_kv") or 0
    if n_kv <= 0:
        # 0 means MHA (KV heads == query heads); fall back to head_count
        n_kv = _field(reader, f"{arch}.attention.head_count") or 1
    head_dim = _field(reader, f"{arch}.attention.key_length")
    if head_dim is None:
        head_dim = _field(reader, f"{arch}.attention.value_length", 128)
    experts = _field(reader, f"{arch}.expert_count") or 0
    return {
        "arch": arch,
        "n_layers": int(n_layers),
        "ctx_train": int(ctx_train),
        "n_kv": int(n_kv),
        "head_dim": int(head_dim),
        "is_moe": experts > 0,
        "experts": int(experts),
    }


def optimize(meta, weight_mb, stem, allow_256k):
    """Return (ctx_size, kv_cache). user's verified table takes priority;
    unknown models fall back to a dynamic VRAM budget."""
    emp = _lookup_empirical(stem)
    if emp is not None:
        return emp[0], emp[1]

    # Dynamic fallback for unknown models.
    if meta["n_layers"] <= 0 or meta["n_kv"] <= 0:
        return CTX_MIN, "q8_0"
    kv_elem = 2 * meta["n_layers"] * meta["n_kv"] * meta["head_dim"]
    cap = CTX_MAX if allow_256k else 131072
    targets = [t for t in (262144, 131072, 65536)
               if t <= cap and t <= meta["ctx_train"]]
    if not targets:
        targets = [min(cap, meta["ctx_train"], CTX_MAX)]
    for target in targets:
        for kv, bytes_per in KV_TYPES:
            mb_per_token = kv_elem * bytes_per / (1024 * 1024)
            kv_mb = mb_per_token * target
            if weight_mb + kv_mb <= USABLE_MB:
                return target, kv
    return CTX_MIN, "q4_0"


def display_name(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").strip().title()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register all GGUF models.")
    parser.add_argument(
        "--bin", default=os.environ.get("LLAMACPP_BIN"),
        help="llama.cpp bin dir (defaults to the $LLAMACPP_BIN env var)")
    args = parser.parse_args()
    if not args.bin:
        parser.error("provide --bin DIR or set the LLAMACPP_BIN environment variable")
    bin_path = Path(args.bin)
    gguf_files = sorted(p for p in bin_path.rglob("*.gguf")
                        if "mmproj" not in p.name.lower())
    if not gguf_files:
        print("No GGUF language models found under", bin_path)
        return

    # Separate MTP draft files (e.g. "*-mtp.gguf") from base models so the
    # drafts are paired to their base instead of registered as standalone.
    drafts: dict[str, Path] = {}
    base_files: list[Path] = []
    for gpath in gguf_files:
        s = gpath.stem.replace(" ", "-").lower()
        if is_mtp_draft_filename(s):
            drafts[base_stem_from_mtp(s)] = gpath
        else:
            base_files.append(gpath)

    n_threads = os.cpu_count() or 8
    store = ConfigStore()

    print(f"VRAM budget: {GPU_TOTAL_MB} MB total, {USABLE_MB} MB usable for model+KV")
    print(f"CPU threads available: {n_threads}")
    print("=" * 78)

    added, updated, skipped = [], [], []

    for gpath in base_files:
        model_path = str(gpath.parent)
        gguf_file = gpath.name
        stem = gpath.stem
        pname = stem.replace(" ", "-").lower()

        try:
            meta = read_meta(gpath)
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {stem}: metadata read failed: {exc}")
            skipped.append(stem)
            continue

        weight_mb = gpath.stat().st_size / (1024 * 1024)
        allow_256k = "35b" in stem.lower()
        ctx_size, kv_cache = optimize(meta, weight_mb, stem, allow_256k)

        # reasoning: use empirical table if matched, else heuristic on name
        emp = _lookup_empirical(stem)
        if emp is not None:
            reasoning = emp[2]
        else:
            reasoning = any(
                k in stem.lower()
                for k in ("thinking", "reasoning", "qwen3.6-12b-thinking")
            )

        # mmproj pairing
        extra = []
        for fam, rel in MMPROJ:
            if fam in stem.lower():
                mp = bin_path / rel
                if mp.is_file():
                    extra.append(mp.name)
                    break

        # Autonomous capability detection (MoE / MTP support) + MTP draft pairing.
        cap = read_gguf_meta(gpath)
        is_moe = bool(cap.get("is_moe", False))
        mtp_supported = bool(cap.get("mtp_supported", False))
        draft = drafts.get(pname)
        if draft is None:
            draft = find_mtp_draft(model_path, pname)
        draft_name = draft.name if isinstance(draft, Path) else draft

        exists = store.load(pname) is not None
        if exists:
            prof = store.load(pname)
        else:
            prof = store.create_default_profile(
                display_name=display_name(stem),
                model_path=model_path,
                gguf_file=gguf_file,
            )
        prof.inference.gpu_layers = -1
        prof.inference.ctx_size = ctx_size
        prof.inference.n_threads = n_threads
        prof.inference.n_batch = 1024
        prof.inference.n_parallel = 1
        prof.kv_cache = kv_cache
        prof.reasoning = reasoning
        prof.is_moe = is_moe
        prof.mtp_supported = mtp_supported
        if extra:
            object.__setattr__(prof, "extra_files", extra)
        if draft_name:
            prof.mtp_model = draft_name
            prof.mtp_enabled = True

        tag_hint = (emp[3] if emp is not None else "")

        # Seed the locked "default" preset with community-recommended sampling.
        prof.ensure_default_preset()

        store.add(prof)

        tag = "UPDATE" if exists else "ADD"
        (updated if exists else added).append(pname)
        kind = "MoE" if is_moe else "dense"
        print(f"[{tag}] {prof.display_name}")
        print(f"    path : {gguf_file}")
        print(f"    arch : {meta['arch']} ({kind}), layers={meta['n_layers']}, "
              f"kv_heads={meta['n_kv']}, head_dim={meta['head_dim']}, "
              f"ctx_train={meta['ctx_train']}")
        print(f"    size : {weight_mb:8.1f} MB | gpu_layers=-1 (full offload) | "
              f"threads={n_threads} | batch=1024")
        print(f"    KV   : {kv_cache} | ctx_size={ctx_size} | "
              f"reasoning={'on' if reasoning else 'off'} | mmproj={extra or '-'} "
              f"moe={'yes' if is_moe else 'no'} | "
              f"mtp={'on' if draft_name else ('supported' if mtp_supported else '-')} "
              f"tag={tag_hint or '-'}")
        print("-" * 78)

    # Drop empty placeholder if present
    ph = store.load("__new_model__")
    if ph is not None and not ph.gguf_file:
        store.delete("__new_model__")

    print("=" * 78)
    print(f"Added {len(added)} new, updated {len(updated)} existing, "
          f"skipped {len(skipped)}.")
    print(f"Total profiles now: {len(store.list_profiles())}")
    print("REOPEN CHECK:")
    reopen = ConfigStore()
    for n in reopen.list_profiles():
        pr = reopen.load(n)
        print(f"  - {n}: ctx={pr.inference.ctx_size} kv={pr.kv_cache} "
              f"gpu={pr.inference.gpu_layers} mmproj={pr.extra_files or '-'}")


if __name__ == "__main__":
    main()
