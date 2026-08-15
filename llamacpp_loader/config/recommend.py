"""Community-recommended sampler settings and the preset vocabulary.

The llama.cpp / vendor communities publish well-tested sampler defaults per
model family.  We curate them here so the loader can seed every model's locked
``default`` preset with a sensible baseline instead of the bare global defaults.

These are *recommendations*, not gospel — the user can always fork them into a
custom Preset 1/2/3 and tune to taste.  The ``default`` preset itself is locked
(see :data:`PRESET_DEFAULT` / :func:`is_locked_preset`) so it always reflects
the community recommendation and cannot be clobbered by accident.
"""

from __future__ import annotations

from typing import Iterable

from .store import SamplingParams

# ----------------------------------------------------------------- community table

# Each entry: (family_keywords, recommended_sampling_kwargs).
# Order does not matter — the first family whose keyword appears in the model
# stem wins (checked top-to-bottom, first match).  Keep keywords lowercase and
# dash/underscore-insensitive (the lookup normalises the stem).
COMMUNITY_SAMPLING: list[tuple[Iterable[str], dict]] = [
    # Qwen (2.5 / 3 / 3.6): official recommendation ~ temp 0.7, top_p 0.8, top_k 20
    (("qwen",), dict(temperature=0.7, top_k=20, top_p=0.8, repeat_penalty=1.05)),
    # Gemma (2 / 3 / 4): Google recommends temp 1.0, top_k 64, top_p 0.95
    (("gemma",), dict(temperature=1.0, top_k=64, top_p=0.95, repeat_penalty=1.0)),
    # Llama (3 / 3.1 / 3.2): Meta recommends temp 0.7, top_p 0.9, top_k 40
    (("llama",), dict(temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.0)),
    # Mistral / Mixtral
    (("mistral", "mixtral"), dict(temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.1)),
    # DeepSeek (V3 / R1): official recommends temp 0.6, top_p 0.95, top_k 20
    (("deepseek",), dict(temperature=0.6, top_k=20, top_p=0.95, repeat_penalty=1.0)),
    # Phi
    (("phi",), dict(temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.0)),
    # GLM / ChatGLM
    (("glm",), dict(temperature=0.8, top_k=40, top_p=0.9, repeat_penalty=1.05)),
]

# Fallback used when no family matches — mirrors the global SamplingParams
# defaults so behaviour is unchanged for unknown models.
FALLBACK_SAMPLING: dict = dict(
    temperature=0.7, top_k=40, top_p=0.95, repeat_penalty=1.1,
)


def recommend_sampling(stem: str) -> SamplingParams:
    """Return community-recommended :class:`SamplingParams` for a model stem.

    Args:
        stem: Model filename/profile stem (e.g. ``qwen3.6-35b-a3b-q4_k_m``).
    """
    s = (stem or "").lower().replace(" ", "-")
    for keys, params in COMMUNITY_SAMPLING:
        if any(k in s for k in keys):
            gp = dict(FALLBACK_SAMPLING)
            gp.update(params)
            return SamplingParams(**gp)
    return SamplingParams(**dict(FALLBACK_SAMPLING))


# ----------------------------------------------------------------- preset vocab

# The built-in, read-only baseline preset holding community recommendations.
PRESET_DEFAULT = "default"

# User-editable / savable custom preset slots.
USER_PRESETS = ("Preset 1", "Preset 2", "Preset 3")


def is_locked_preset(name: str) -> bool:
    """True if *name* is a built-in locked preset (currently only 'default')."""
    return name == PRESET_DEFAULT
