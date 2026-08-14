"""JSON-backed configuration store for llamacpp-loader.

Architecture: each GGUF model has its own ``ModelProfile`` (inference parameters
plus a human-readable name).  ConfigStore manages the collection of profiles and
provides add / update / delete / load / scan operations.  A single global
defaults profile acts as the template for new models.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- helpers


def _default_path() -> Path:
    """Return platform-appropriate config file path."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", str(Path.home())))
    else:
        base = Path.home() / ".config"
    return base / "llamacpp-loader" / "settings.json"


def _clamp(val, lo, hi):
    """Return *val* clamped to [lo, hi]."""
    return max(lo, min(hi, val))


# --------------------------------------------------------------------------- data classes


@dataclass(slots=True)
class SamplingParams:
    """Sampling (generation) parameters.

    Values are validated on assignment via ``__post_init__`` and the class-level
    ``set_*`` helpers that re-validate on each mutation.
    """

    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "temperature", _clamp(self.temperature, 0.0, 2.0))
        object.__setattr__(self, "top_k", max(1, self.top_k))
        object.__setattr__(self, "top_p", _clamp(self.top_p, 0.0, 1.0))
        object.__setattr__(self, "repeat_penalty", _clamp(self.repeat_penalty, 0.0, 3.0))
        object.__setattr__(self, "frequency_penalty", _clamp(self.frequency_penalty, -2.0, 2.0))
        object.__setattr__(self, "presence_penalty", _clamp(self.presence_penalty, -2.0, 2.0))

    # --- validation helpers for runtime mutation ---
    def set_temperature(self, val):
        object.__setattr__(self, "temperature", _clamp(val, 0.0, 2.0))

    def set_top_k(self, val):
        object.__setattr__(self, "top_k", max(1, int(val)))

    def set_top_p(self, val):
        object.__setattr__(self, "top_p", _clamp(val, 0.0, 1.0))

    def set_repeat_penalty(self, val):
        object.__setattr__(self, "repeat_penalty", _clamp(val, 0.0, 3.0))

    def set_frequency_penalty(self, val):
        object.__setattr__(self, "frequency_penalty", _clamp(val, -2.0, 2.0))

    def set_presence_penalty(self, val):
        object.__setattr__(self, "presence_penalty", _clamp(val, -2.0, 2.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SamplingParams:
        kw = {k: data[k] for k in data if k in SamplingParams.__dataclass_fields__}
        sp = SamplingParams(**kw)  # __post_init__ validates
        return sp


@dataclass(slots=True)
class InferenceParams:
    """Inference (execution) parameters.

    Values are validated on assignment via ``__post_init__`` and the class-level
    ``set_*`` helpers that re-validate on each mutation.
    """

    ctx_size: int = 4096
    gpu_layers: int = -1       # -1 means auto-detect / all layers; use 0 for CPU-only
    n_threads: int = 4         # physical cores recommended
    n_batch: int = 512         # prompt processing batch size
    n_parallel: int = 1        # number of parallel sequences (speculative decoding)
    seed: int = -1             # -1 = random seed each run

    def __post_init__(self):
        object.__setattr__(self, "ctx_size", max(64, self.ctx_size))
        object.__setattr__(self, "n_threads", max(1, self.n_threads))
        object.__setattr__(self, "n_batch", max(1, min(self.n_batch, 8192)))
        object.__setattr__(self, "seed", -1 if self.seed < 0 else self.seed)

    # --- validation helpers for runtime mutation ---
    def set_ctx_size(self, val):
        object.__setattr__(self, "ctx_size", max(64, int(val)))

    def set_n_threads(self, val):
        object.__setattr__(self, "n_threads", max(1, int(val)))

    def set_n_batch(self, val):
        object.__setattr__(self, "n_batch", max(1, min(int(val), 8192)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> InferenceParams:
        kw = {k: data[k] for k in data if k in InferenceParams.__dataclass_fields__}
        ip = InferenceParams(**kw)  # __post_init__ validates
        return ip


@dataclass(slots=True)
class ServerParams:
    """Server networking parameters."""

    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self):
        object.__setattr__(self, "port", max(1, min(self.port, 65535)))

    # --- validation helper for runtime mutation ---
    def set_port(self, val):
        object.__setattr__(self, "port", max(1, min(int(val), 65535)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ServerParams:
        kw = {k: data[k] for k in data if k in ServerParams.__dataclass_fields__}
        sp = ServerParams(**kw)  # __post_init__ validates
        return sp


@dataclass(slots=True)
class ModelProfile:
    """A single model's complete configuration.

    Each profile is keyed by ``profile_name`` (unique identifier).  The user-facing
    label for display purposes is stored in ``display_name``; the actual GGUF path
    lives in ``model_path`` and ``gguf_file``.

    When persisted, only non-default values are written back to save space, but
    ``to_dict()`` always returns a complete snapshot with defaults filled in.
    """

    profile_name: str = ""           # unique key (e.g. "llama-3.2-3b-q4")
    display_name: str = ""           # human-readable label shown in GUI
    model_path: str = ""             # directory containing GGUF files
    gguf_file: str = ""              # selected GGUF filename (relative to model_path)

    server: ServerParams = field(default_factory=ServerParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    sampling: SamplingParams = field(default_factory=SamplingParams)

    def __post_init__(self):
        if not self.profile_name and self.gguf_file:
            # Derive profile_name from GGUF filename if not set
            base = Path(self.gguf_file).stem
            object.__setattr__(self, "profile_name", base.replace(" ", "-").lower())
        if not self.display_name:
            object.__setattr__(self, "display_name", self.profile_name or "Untitled Profile")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete snapshot (all defaults filled in)."""
        return {
            "profile_name": self.profile_name,
            "display_name": self.display_name,
            "model_path": self.model_path,
            "gguf_file": self.gguf_file,
            "server": self.server.to_dict(),
            "inference": self.inference.to_dict(),
            "sampling": self.sampling.to_dict(),
        }


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelProfile:
        """Construct a ModelProfile from its dictionary representation."""
        server_data = data.get("server", {})
        inference_data = data.get("inference", {})
        sampling_data = data.get("sampling", {})
        return cls(
            profile_name=data.get("profile_name", ""),
            display_name=data.get("display_name", ""),
            model_path=data.get("model_path", ""),
            gguf_file=data.get("gguf_file", ""),
            server=ServerParams.from_dict(server_data),
            inference=InferenceParams.from_dict(inference_data),
            sampling=SamplingParams.from_dict(sampling_data),
        )


    def to_server_config(self) -> tuple[str, int]:
        """Convenience: return (model_command_string, port).

        Used by the process manager to build the launch command.
        Returns the full model path as a single string and the listen port.
        """
        if self.model_path and self.gguf_file:
            full_model = os.path.join(self.model_path, self.gguf_file)
        elif self.gguf_file:
            full_model = self.gguf_file
        else:
            full_model = ""
        return full_model, self.server.port


@dataclass(slots=True)
class UiState:
    """Non-model settings saved across sessions (window size, last browse dir)."""

    window_width: int = 960
    window_height: int = 680
    last_browse_dir: str = ""       # remembered directory from Browse dialog


# --------------------------------------------------------------------------- config store


class ConfigStore:
    """Collection manager for ModelProfiles.

    Thread safety::
        All public methods are protected by a threading.Lock so multiple threads
        (GUI thread + process-manager thread) can read/write safely.  On every
        mutation the current state is atomically persisted to disk.

    Persistence format (JSON)::
        {
            "_meta": {"version": 1},
            "defaults": { ... flattened default profile dict ... },
            "profiles": {
                "<profile_name>": { ... ModelProfile.to_dict() ... }
            },
            "ui_state": { ... UiState dict ... }
        }


    Usage::
        store = ConfigStore()                      # loads existing file or creates defaults
        profile = store.create_default_profile("my-model")  # copy global defaults
        profile.inference.ctx_size = 8192          # customize
        store.add(profile)                         # persist
        loaded = store.load("my-model")            # retrieve later
    """

    CURRENT_VERSION = 1

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._path: Path = path if path is not None else _default_path()
        # Internal state
        self._defaults = ModelProfile(
            profile_name="__global_defaults__",
            display_name="Global Defaults",
            server=ServerParams(host="127.0.0.1", port=8080),
        )
        self._profiles: dict[str, ModelProfile] = {}
        self._ui_state = UiState()

    # ------------------------------------------------------------------ load/save
    def _load(self) -> None:
        """Load settings from disk.  Creates defaults if file doesn't exist."""
        if not self._path.exists():
            # First run - write out default structure
            self.save()
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load config %s: %s", self._path, exc)
            # Recoverable - keep in-memory defaults and overwrite on next save.
            return

        meta = raw.get("_meta", {})
        version = meta.get("version")
        if version is not None and version != self.CURRENT_VERSION:
            logger.warning(
                "Config version %d mismatch (current=%d), migrating.", version,
                self.CURRENT_VERSION,
            )

        # Load defaults
        defs_data = raw.get("defaults", {})
        if defs_data:
            self._defaults = ModelProfile.from_dict(defs_data)

        # Load profiles
        profiles_raw = raw.get("profiles", {})
        for name, pdata in profiles_raw.items():
            try:
                self._profiles[name] = ModelProfile.from_dict(pdata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping corrupt profile %s: %s", name, exc)

        # Load UI state
        ui_data = raw.get("ui_state")
        if ui_data:
            valid_keys = UiState.__dataclass_fields__
            self._ui_state = UiState(**{k: v for k, v in ui_data.items() if k in valid_keys})
    def save(self) -> None:
        """Persist current state to disk (thread-safe)."""
        with self._lock:
            data = {
                "_meta": {"version": self.CURRENT_VERSION},
                "defaults": self._defaults.to_dict(),
                "profiles": {name: p.to_dict() for name, p in self._profiles.items()},
                "ui_state": asdict(self._ui_state),
            }
        # Atomic write via temp file + rename
        tmp_path = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(str(tmp_path), str(self._path))
        except OSError as exc:
            logger.error("Failed to save config %s: %s", self._path, exc)

    def reload(self) -> None:
        """Force-reload from disk (useful after external edits)."""
        with self._lock:
            self._profiles.clear()
            self._defaults = ModelProfile(
                profile_name="__global_defaults__",
                display_name="Global Defaults",
            )
            self._ui_state = UiState()
        self._load()

    # ------------------------------------------------------------------ defaults
    @property
    def defaults(self) -> ModelProfile:
        """Return a *copy* of the global default profile (safe to modify)."""
        return ModelProfile.from_dict(self._defaults.to_dict())

    @defaults.setter
    def defaults(self, value: ModelProfile) -> None:
        """Replace the global default template and persist."""
        with self._lock:
            self._defaults = value
        self.save()

    def apply_defaults_to(self, profile: ModelProfile) -> ModelProfile:
        """Create a new profile that starts from global defaults, then applies *profile*'s non-default fields.

        Useful when cloning or creating a new model based on existing settings.
        Returns the merged profile (original ``profile`` is not mutated).
        """
        base = ModelProfile.from_dict(self._defaults.to_dict())
        if profile.model_path:
            object.__setattr__(base, "model_path", profile.model_path)
        if profile.gguf_file:
            object.__setattr__(base, "gguf_file", profile.gguf_file)
        if profile.display_name and profile.display_name != "Untitled Profile":
            object.__setattr__(base, "display_name", profile.display_name)
        # Copy server/inference/sampling only if non-default
        default_server = ServerParams()
        for key, val in profile.server.to_dict().items():
            if val != getattr(default_server, key):
                setattr(base.server, key, val)
        default_infer = InferenceParams()
        for key, val in profile.inference.to_dict().items():
            if val != getattr(default_infer, key):
                setattr(base.inference, key, val)
        default_sampling = SamplingParams()
        for key, val in profile.sampling.to_dict().items():
            if val != getattr(default_sampling, key):
                setattr(base.sampling, key, val)
        return base

    # ------------------------------------------------------------------ UI state
    def get_ui_state(self) -> UiState:
        """Return a copy of the current UI state."""
        with self._lock:
            return UiState(**asdict(self._ui_state))

    def set_ui_state(self, **kwargs: Any) -> None:
        """Update one or more UI state fields and persist."""
        with self._lock:
            for key, val in kwargs.items():
                if hasattr(self._ui_state, key):
                    setattr(self._ui_state, key, val)
        self.save()

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> ConfigStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.save()

    # --------------------------------------------------------------------------- model discovery
    @classmethod
    def scan_models(cls, directory: str) -> dict[str, ModelProfile]:
        """Recursively discover .gguf models in *directory* and generate ModelProfiles.

        For each ``.gguf`` file found, profile_name is derived from the filename stem
        (spaces replaced with dashes, lowercased), augmented with a relative path fragment
        to ensure uniqueness when multiple files share the same stem.  display_name is
        readabilized by title-casing and replacing dashes with spaces.

        Args:
            directory: Root directory to search recursively.

        Returns:
            Dictionary mapping profile_name -> ModelProfile, one per discovered .gguf file.
        """
        from pathlib import Path

        profiles: dict[str, ModelProfile] = {}
        root = Path(directory)

        for gguf_file in root.rglob("*.gguf"):
            relative = gguf_file.relative_to(root)
            # profile_name from filename stem with path fragment for uniqueness
            base_name = gguf_file.stem.replace(" ", "-").lower()
            # Add a relative path fragment to avoid collisions between same-named files in different dirs
            rel_parts = relative.parts
            if len(rel_parts) > 1:
                # Use the first directory component as a prefix
                prefix = rel_parts[0].replace(" ", "-").lower()[:20]
                if prefix:
                    base_name = f"{prefix}-{base_name}"
            # display_name: readable form
            display_name = base_name.replace("-", " ").title()
            # model_path is the parent directory of the .gguf file
            model_path = str(gguf_file.parent)
            gguf_file_rel = relative.as_posix()

            profile = ModelProfile(
                profile_name=base_name,
                display_name=display_name,
                model_path=model_path,
                gguf_file=gguf_file_rel,
            )
            profiles[base_name] = profile

        return profiles

    # ------------------------------------------------------------------ profiles CRUD
    def list_profiles(self) -> list[str]:
        """Return sorted list of all registered profile names."""
        with self._lock:
            return sorted(self._profiles.keys())

    def add(self, profile: ModelProfile) -> bool:
        """Add or update a profile.  Returns True on success."""
        if not profile.profile_name:
            raise ValueError("profile_name is required")
        with self._lock:
            self._profiles[profile.profile_name] = profile
        self.save()
        return True

    def update(self, name: str, updates: dict[str, Any]) -> bool:
        """Update a single field on an existing profile.

        ``updates`` is a flat dict keyed by top-level attribute names
        (``"display_name"``, ``"model_path"``, etc.).  Nested keys use dot
        notation (e.g. ``"inference.ctx_size"``).  Persisted immediately.
        """
        with self._lock:
            if name not in self._profiles:
                return False
            profile = self._profiles[name]
            for key, val in updates.items():
                if "." in key:
                    attr, subkey = key.split(".", 1)
                    inner = getattr(profile, attr)
                    setattr(inner, subkey, val)
                else:
                    setattr(profile, key, val)
        self.save()
        return True

    def delete(self, name: str) -> bool:
        """Delete a profile by name. Returns True if it existed."""
        with self._lock:
            if name not in self._profiles:
                return False
            del self._profiles[name]
        self.save()
        return True

    def load(self, name: str) -> Optional[ModelProfile]:
        """Retrieve a profile by name. Returns None if not found."""
        with self._lock:
            p = self._profiles.get(name)
            if p is not None:
                return ModelProfile.from_dict(p.to_dict())
            return None

    def create_default_profile(
        self, display_name: str | None = None,
        model_path: str = "", gguf_file: str = "",
    ) -> ModelProfile:
        """Create a new profile based on global defaults.

        This is the recommended entry point for adding a new GGUF model - it copies
        the current global default template so every parameter group starts from a
        known-good baseline, then applies any overrides.
        """
        # Start fresh using from_dict of defaults (which calls __post_init__ on sub-objects)
        base = ModelProfile.from_dict(self._defaults.to_dict())
        if display_name:
            object.__setattr__(base, "display_name", display_name)
        if model_path:
            object.__setattr__(base, "model_path", model_path)
        if gguf_file:
            object.__setattr__(base, "gguf_file", gguf_file)
            # Re-derive profile name from new gguf_file since __post_init__ already ran
            derived = Path(gguf_file).stem.replace(" ", "-").lower()
            object.__setattr__(base, "profile_name", derived)
        return base

    # ------------------------------------------------------------------ UI state
    def get_ui_state(self) -> UiState:
        """Return a copy of the current UI state."""
        with self._lock:
            return UiState(**asdict(self._ui_state))

    def set_ui_state(self, **kwargs: Any) -> None:
        """Update one or more UI state fields and persist."""
        with self._lock:
            for key, val in kwargs.items():
                if hasattr(self._ui_state, key):
                    setattr(self._ui_state, key, val)
        self.save()

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> ConfigStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.save()


# --------------------------------------------------------------------------- CLI helper
def main() -> None:  # pragma: no cover
    """Quick smoke test from command line."""
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    store = ConfigStore()
    print("Profiles:", store.list_profiles())
    profile = store.create_default_profile(
        display_name="Test Model",
        model_path="/tmp/models",
        gguf_file="llama-3.2-3b-q4_k_m.gguf",
    )
    profile.inference.ctx_size = 8192
    print("Created:", profile.display_name)
    store.add(profile)
    print("After add:", store.list_profiles())


if __name__ == "__main__":
    main()