"""Tests for config.store module."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from llamacpp_loader.config.store import (
    ConfigStore, ModelProfile, ServerParams, InferenceParams, SamplingParams, UiState,
)


# ===================================================================== defaults fixture


@pytest.fixture()
def store():
    """A fresh ConfigStore backed by a temp file."""
    tmp = Path(f"/tmp/llamacpp-test-config-{id(object())}.json")
    s = ConfigStore(path=tmp)
    yield s
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


# =================================================================== defaults tests


class TestDefaults:

    def test_defaults_are_available(self, store):
        d = store.defaults
        assert isinstance(d, ModelProfile)
        assert d.server.port == 8080
        assert d.inference.ctx_size == 4096
        assert d.sampling.temperature == 0.7

    def test_modify_defaults_does_not_mutate_store(self, store):
        copy = store.defaults
        copy.sampling.set_temperature(1.5)  # use set_* helper for validation
        loaded = store.defaults
        assert loaded.sampling.temperature == 0.7


# ============================================================= create_default_profile


class TestCreateDefaultProfile:

    def test_creates_from_global_defaults(self, store):
        profile = store.create_default_profile(
            display_name="My Model",
            model_path="/tmpmodels",
            gguf_file="model.gguf",
        )
        assert isinstance(profile, ModelProfile)
        assert profile.display_name == "My Model"
        assert profile.model_path == "/tmpmodels"
        assert profile.inference.ctx_size == 4096  # inherited from defaults

    def test_auto_derives_profile_name_from_gguf(self, store):
        profile = store.create_default_profile(
            gguf_file="llama-3.2-3b-q4_k_m.gguf",
        )
        assert profile.profile_name == "llama-3.2-3b-q4_k_m"

    def test_new_profile_inherits_sampling_defaults(self, store):
        profile = store.create_default_profile(display_name="X")
        assert profile.sampling.top_p == 0.95
        assert profile.sampling.repeat_penalty == 1.1


# ============================================================ create_default_profile CRUD operations


class TestCRUD:

    def test_add_and_list(self, store):
        p = ModelProfile(profile_name="alpha", display_name="Alpha")
        result = store.add(p)
        assert result is True
        names = store.list_profiles()
        assert "alpha" in names

    def test_update_field(self, store):
        store.add(ModelProfile(profile_name="beta", display_name="Beta"))
        ok = store.update("beta", {"display_name": "Updated Beta"})
        assert ok is True
        loaded = store.load("beta")
        assert loaded.display_name == "Updated Beta"

    def test_update_nested_field(self, store):
        profile = ModelProfile(profile_name="gamma", display_name="Gamma")
        profile.inference.set_ctx_size(2048)  # use set_* helper
        store.add(profile)
        ok = store.update("gamma", {"inference.ctx_size": 16384})
        assert ok is True
        loaded = store.load("gamma")
        assert loaded.inference.ctx_size == 16384

    def test_update_nonexistent_returns_false(self, store):
        assert store.update("no-such-profile", {"display_name": "X"}) is False

    def test_delete_existing(self, store):
        store.add(ModelProfile(profile_name="delta", display_name="Delta"))
        assert store.delete("delta") is True
        assert store.load("delta") is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("no-such-profile") is False

    def test_add_requires_profile_name(self, store):
        with pytest.raises(ValueError, match="profile_name"):
            store.add(ModelProfile(display_name="No Name"))

# ============================================================= load / persistence


class TestPersistence:

    def test_add_then_load_roundtrip(self, store):
        profile = ModelProfile(
            profile_name="roundtrip",
            display_name="RoundTrip",
            model_path="/models",
            gguf_file="test.gguf",
        )
        profile.inference.set_ctx_size(8192)  # use set_* helper
        profile.sampling.set_temperature(0.3)  # <-- added to persist temperature
        store.add(profile)

        loaded = store.load("roundtrip")
        assert loaded is not None
        assert loaded.profile_name == "roundtrip"
        assert loaded.display_name == "RoundTrip"
        assert loaded.inference.ctx_size == 8192
        assert loaded.sampling.temperature == 0.3

    def test_config_persists_to_disk(self, store):
        profile = ModelProfile(
            profile_name="disk-test",
            display_name="Disk Test",
            model_path="/models",
            gguf_file="test.gguf",
        )
        store.add(profile)
        data = json.loads(store._path.read_text(encoding="utf-8"))
        assert "profiles" in data
        assert "disk-test" in data["profiles"]

    def test_meta_version_written(self, store):
        profile = ModelProfile(
            profile_name="version-test",
            display_name="VTest",
        )
        store.add(profile)
        data = json.loads(store._path.read_text(encoding="utf-8"))
        assert data["_meta"]["version"] == 1

    def test_reload_restores_state(self, store):
        profile = ModelProfile(
            profile_name="reload-test",
            display_name="Reload",
        )
        store.add(profile)
        store.reload()
        loaded = store.load("reload-test")
        assert loaded is not None


# ============================================================= apply_defaults_to


class TestApplyDefaultsTo:

    def test_merged_profile(self, store):
        base = ModelProfile(
            profile_name="base",
            display_name="Base",
            model_path="/models",
            gguf_file="base.gguf",
        )
        custom = ModelProfile.from_dict({
            "profile_name": "",
            "display_name": "",
            "model_path": "/other/models",
            "gguf_file": "custom.gguf",
            "inference": {"ctx_size": 8192},
        })
        merged = store.apply_defaults_to(custom)
        assert "custom" in merged.display_name.lower()
        assert merged.inference.ctx_size == 8192

    def test_inherits_default_sampling(self, store):
        custom = ModelProfile.from_dict({
            "profile_name": "",
            "display_name": "",
            "model_path": "/models",
            "gguf_file": "m.gguf",
        })
        merged = store.apply_defaults_to(custom)
        assert merged.sampling.temperature == 0.7


# ============================================================= UI state


class TestUiState:

    def test_get_ui_state(self, store):
        ui = store.get_ui_state()
        assert isinstance(ui, UiState)
        assert ui.window_width == 960

    def test_set_ui_state_persists(self, store):
        store.set_ui_state(window_width=1280)
        ui = store.get_ui_state()
        assert ui.window_width == 1280


# ============================================================= ModelProfile dataclass


class TestModelProfile:

    def test_to_dict_roundtrip(self):
        orig = ModelProfile(
            profile_name="round",
            display_name="RoundTrip",
            model_path="/models",
            gguf_file="m.gguf",
        )
        orig.inference.set_ctx_size(1024)  # use set_* helper
        d = orig.to_dict()
        restored = ModelProfile.from_dict(d)
        assert restored.profile_name == "round"
        assert restored.display_name == "RoundTrip"
        assert restored.inference.ctx_size == 1024

    def test_to_server_config(self):
        p = ModelProfile(model_path="/models", gguf_file="m.gguf")
        full, port = p.to_server_config()
        assert "m.gguf" in full
        assert port == 8080

    def test_auto_profile_name_from_gguf(self):
        p = ModelProfile(gguf_file="llama-3.2-3b-q4_k_m.gguf")
        assert p.profile_name == "llama-3.2-3b-q4_k_m"


# ============================================================= Validation bounds (using set_* helpers)


class TestValidationBounds:

    def test_temperature_clamped(self):
        sp = SamplingParams()  # defaults validated by __post_init__
        sp.set_temperature(5.0)
        assert sp.temperature == 2.0

    def test_top_k_min_one(self):
        sp = SamplingParams()
        sp.set_top_k(-10)
        assert sp.top_k >= 1

    def test_port_clamped(self):
        p = ServerParams(port=99999)
        assert p.port == 65535

    def test_ctx_size_min_64(self):
        p = InferenceParams(ctx_size=10)
        assert p.ctx_size >= 64


# --------------------------------------------------------------------------- context manager


class TestContextManager:

    @patch("os.replace")
    def test_context_manager_saves_on_exit(self, mock_replace, store):
        with store:
            pass  # save() called on __exit__
        mock_replace.assert_called_once()


# ============================================================= Model scan discovery

class TestModelScan:
    """Tests for ConfigStore.scan_models classmethod."""

    def test_scan_gguf_files(self, tmp_path):
        """Test that scan_models discovers .gguf files recursively."""
        test_dir = tmp_path / "models"
        test_dir.mkdir()
        (test_dir / "model-1.gguf").touch()
        (test_dir / "sub").mkdir(parents=True)
        (test_dir / "sub" / "model-2.gguf").touch()

        profiles = ConfigStore.scan_models(str(test_dir))
        assert len(profiles) == 2

        # Check profile_name derivation
        p1 = profiles["model-1"]
        assert p1.profile_name == "model-1"
        # display_name is title-cased (e.g., "Model 1"), check it contains the base words
        assert "model" in p1.display_name.lower()

        # Check profile_name derivation from subdirectory
        p2 = profiles["sub-model-2"]
        assert p2.profile_name == "sub-model-2"
        # display_name is title-cased (e.g., "Sub Model 2")
        assert "sub-model-2" in p2.display_name or "sub model" in p2.display_name.lower()

    def test_scan_no_gguf_files(self, tmp_path):
        """Test that scan_models returns empty dict when no .gguf files exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        profiles = ConfigStore.scan_models(str(empty_dir))
        assert len(profiles) == 0

    def test_scan_model_name_from_filename(self, tmp_path):
        """Test that profile_name is derived from filename stem."""
        test_dir = tmp_path / "models"
        test_dir.mkdir()
        # File with spaces
        (test_dir / "my gguf model.gguf").touch()

        profiles = ConfigStore.scan_models(str(test_dir))
        assert "my-gguf-model" in profiles
        profile = profiles["my-gguf-model"]
        assert profile.profile_name == "my-gguf-model"
        # display_name should be readable (title-cased)
        assert "My Gguf Model" in profile.display_name or profile.display_name == "My Gguf Model"

    def test_scan_distinct_names_no_collision(self, tmp_path):
        """Test that same-named files in different dirs both get profiles."""
        test_dir = tmp_path / "models"
        test_dir.mkdir()
        sub1 = test_dir / "dir1"
        sub2 = test_dir / "dir2"
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / "model.gguf").touch()
        (sub2 / "model.gguf").touch()

        profiles = ConfigStore.scan_models(str(test_dir))
        # Both should be found with distinct profile names due to path context
        assert len(profiles) == 2

