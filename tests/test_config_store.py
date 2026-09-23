"""Tests for config.store module."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from llamacpp_loader.config.store import (
    ConfigStore, ModelProfile, ServerParams, InferenceParams, SamplingParams, UiState,
)
from llamacpp_loader.config.metadata import looks_moe_from_name


# ===================================================================== defaults fixture


@pytest.fixture()
def store(tmp_path):
    """A fresh ConfigStore backed by a cross-platform temp file."""
    tmp = tmp_path / "config.json"
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


# ===================================================================== cpu_moe


class TestCpuMoePersistence:

    def test_cpu_moe_roundtrip(self):
        p = ModelProfile(profile_name="moe-model", cpu_moe=True)
        data = p.to_dict()
        assert data["cpu_moe"] is True
        p2 = ModelProfile.from_dict(data)
        assert p2.cpu_moe is True

    def test_cpu_moe_defaults_false(self):
        p = ModelProfile(profile_name="dense-model")
        assert p.cpu_moe is False
        assert p.to_dict()["cpu_moe"] is False

    def test_n_cpu_moe_roundtrip(self):
        p = ModelProfile(profile_name="moe-model", n_cpu_moe=18)
        data = p.to_dict()
        assert data["n_cpu_moe"] == 18
        p2 = ModelProfile.from_dict(data)
        assert p2.n_cpu_moe == 18

    def test_n_cpu_moe_defaults_zero(self):
        p = ModelProfile(profile_name="dense-model")
        assert p.n_cpu_moe == 0
        assert p.to_dict()["n_cpu_moe"] == 0


# ============================================================ MoE name fallback


class TestMoENameFallback:
    """MoE must still be detected when the optional `gguf` package is absent."""

    def test_detects_a3b_naming(self):
        assert looks_moe_from_name("Qwen3.6-35B-A3B-Aggressive-Q4_K_M.gguf") is True

    def test_detects_other_active_param_sizes(self):
        assert looks_moe_from_name("DeepSeek-V3-0324-A22B-Q4_K_M.gguf") is True
        assert looks_moe_from_name("Qwen3.5-122B-A10B-Q4_K_M.gguf") is True

    def test_detects_mixtral_family(self):
        assert looks_moe_from_name("Mixtral-8x7B-v0.1-Q4_K_M.gguf") is True

    def test_dense_models_stay_false(self):
        assert looks_moe_from_name("Qwen3.8-27B-Q4_K_M.gguf") is False
        assert looks_moe_from_name("gemma-4-12B-it-QAT-Q4_0.gguf") is False

    def test_empty_name_is_false(self):
        assert looks_moe_from_name("") is False



# ============================================================ N-gram spec decoding


class TestNgramConfig:
    """N-gram selection must round-trip and degrade safely."""

    def test_disabled_by_default(self):
        p = ModelProfile(profile_name="m", gguf_file="m.gguf")
        assert p.ngram_enabled is False
        assert p.ngram_type == "ngram-simple"

    def test_round_trips_through_dict(self):
        p = ModelProfile(profile_name="m", gguf_file="m.gguf",
                         ngram_enabled=True, ngram_type="ngram-mod")
        back = ModelProfile.from_dict(p.to_dict())
        assert back.ngram_enabled is True
        assert back.ngram_type == "ngram-mod"

    def test_legacy_config_without_ngram_key(self):
        """Configs written before n-gram existed must still load (and stay off)."""
        data = {"profile_name": "m", "gguf_file": "m.gguf"}
        p = ModelProfile.from_dict(data)
        assert p.ngram_enabled is False

    def test_unknown_type_falls_back_to_default(self):
        p = ModelProfile.from_dict(
            {"profile_name": "m", "gguf_file": "m.gguf", "ngram_type": "ngram-bogus"})
        assert p.ngram_type == "ngram-simple"

    def test_normalize_accepts_gui_label_and_token(self):
        from llamacpp_loader.config.store import (
            NGRAM_LABELS, normalize_ngram_type, ngram_label)
        assert normalize_ngram_type("Simple") == "ngram-simple"
        assert normalize_ngram_type("ngram-mod") == "ngram-mod"
        assert normalize_ngram_type("Off") == ""
        assert normalize_ngram_type("") == ""
        assert ngram_label("ngram-map-k") == "Map-K"
        assert ngram_label("") == "Off"
        # Every GUI label resolves to a real llama.cpp --spec-type token.
        for label in NGRAM_LABELS:
            assert normalize_ngram_type(label) in ("", *NGRAM_LABELS.values())

    def test_persists_across_store_save(self, store, tmp_path):
        store.add(ModelProfile(profile_name="m", gguf_file="m.gguf",
                               model_path="/models"))
        store.update("m", {"ngram_enabled": True, "ngram_type": "ngram-cache"})
        reloaded = ConfigStore(path=tmp_path / "config.json")
        assert reloaded.load("m").ngram_enabled is True
        assert reloaded.load("m").ngram_type == "ngram-cache"


# ============================================================ n_predict (Max tokens)


class TestNPredict:
    """n_predict caps per-request generation; 0 keeps the server default."""

    def test_defaults_to_unset(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams().n_predict == 0

    def test_validated_non_negative(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(n_predict=-5).n_predict == 0
        assert InferenceParams(n_predict=12000).n_predict == 12000

    def test_round_trips_through_dict(self):
        from llamacpp_loader.config.store import InferenceParams
        back = InferenceParams.from_dict({"n_predict": 12000})
        assert back.n_predict == 12000
        assert InferenceParams.from_dict({}).n_predict == 0

    def test_set_n_predict_clamps(self):
        from llamacpp_loader.config.store import InferenceParams
        ip = InferenceParams()
        ip.set_n_predict(8000)
        assert ip.n_predict == 8000
        ip.set_n_predict(-1)
        assert ip.n_predict == 0

    def test_profile_round_trips_n_predict(self):
        from llamacpp_loader.config.store import InferenceParams, ModelProfile
        p = ModelProfile(profile_name="m", gguf_file="m.gguf",
                         inference=InferenceParams(n_predict=12000))
        back = ModelProfile.from_dict(p.to_dict())
        assert back.inference.n_predict == 12000


class TestUbatchAndBackendSampling:
    """-ub / --backend-sampling (the Hermes prefill recipe) and --min-p."""

    def test_n_ubatch_defaults_to_auto(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams().n_ubatch == 0

    def test_n_ubatch_clamped_to_0_8192(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(n_ubatch=-3).n_ubatch == 0
        assert InferenceParams(n_ubatch=2048).n_ubatch == 2048
        assert InferenceParams(n_ubatch=99999).n_ubatch == 8192

    def test_set_n_ubatch_clamps(self):
        from llamacpp_loader.config.store import InferenceParams
        ip = InferenceParams()
        ip.set_n_ubatch(2048)
        assert ip.n_ubatch == 2048
        ip.set_n_ubatch(-1)
        assert ip.n_ubatch == 0

    def test_backend_sampling_defaults_off(self):
        from llamacpp_loader.config.store import ServerParams
        assert ServerParams().backend_sampling is False

    def test_backend_sampling_coerced_to_bool(self):
        from llamacpp_loader.config.store import ServerParams
        assert ServerParams(backend_sampling=True).backend_sampling is True
        assert ServerParams(backend_sampling=0).backend_sampling is False

    def test_metrics_defaults_off(self):
        from llamacpp_loader.config.store import ServerParams
        assert ServerParams().metrics is False

    def test_metrics_coerced_to_bool(self):
        from llamacpp_loader.config.store import ServerParams
        assert ServerParams(metrics=True).metrics is True
        assert ServerParams(metrics=0).metrics is False

    def test_metrics_setter_coerces(self):
        from llamacpp_loader.config.store import ServerParams
        sp = ServerParams()
        sp.set_metrics(True)
        assert sp.metrics is True
        sp.set_metrics("")
        assert sp.metrics is False

    def test_legacy_server_dict_without_metrics_loads_off(self):
        """Settings written before --metrics existed must still load, and the
        new key must appear once the profile is saved again."""
        from llamacpp_loader.config.store import ServerParams
        legacy = {"host": "127.0.0.1", "port": 8080, "flash_attn": "auto",
                  "backend_sampling": False}
        sp = ServerParams.from_dict(legacy)
        assert sp.metrics is False
        assert sp.to_dict()["metrics"] is False

    def test_profile_round_trips_new_fields(self):
        from llamacpp_loader.config.store import InferenceParams, ModelProfile
        p = ModelProfile(profile_name="m", gguf_file="m.gguf",
                         inference=InferenceParams(n_ubatch=2048))
        p.server.backend_sampling = True
        p.server.metrics = True
        back = ModelProfile.from_dict(p.to_dict())
        assert back.inference.n_ubatch == 2048
        assert back.server.backend_sampling is True
        assert back.server.metrics is True


class TestMinP:
    """--min-p: None = llama.cpp default (0.05); 0.0 is a REAL value (=off)."""

    def test_default_is_none(self):
        from llamacpp_loader.config.store import SamplingParams
        assert SamplingParams().min_p is None

    def test_none_is_the_sentinel_not_zero(self):
        from llamacpp_loader.config.store import SamplingParams
        # 0.0 explicitly disables min-p and must survive as 0.0.
        assert SamplingParams(min_p=0.0).min_p == 0.0
        assert SamplingParams(min_p="").min_p is None
        assert SamplingParams(min_p=None).min_p is None

    def test_clamped_to_0_1(self):
        from llamacpp_loader.config.store import SamplingParams
        assert SamplingParams(min_p=0.1).min_p == 0.1
        assert SamplingParams(min_p=-0.5).min_p == 0.0
        assert SamplingParams(min_p=5).min_p == 1.0

    def test_set_min_p(self):
        from llamacpp_loader.config.store import SamplingParams
        sp = SamplingParams()
        sp.set_min_p("0.05")
        assert sp.min_p == 0.05
        sp.set_min_p("")
        assert sp.min_p is None

    def test_round_trips_through_dict(self):
        from llamacpp_loader.config.store import SamplingParams
        back = SamplingParams.from_dict({"min_p": 0.0})
        assert back.min_p == 0.0
        assert SamplingParams.from_dict({}).min_p is None


# ============================================================ thinking controls


class TestReasoningTriState:
    """Thinking became auto/on/off; legacy bools must keep their behaviour."""

    def test_default_is_auto(self):
        from llamacpp_loader.config.store import ModelProfile
        assert ModelProfile().reasoning == "auto"

    def test_legacy_bool_false_maps_to_auto(self):
        """False used to launch with no reasoning flag at all, i.e. auto."""
        from llamacpp_loader.config.store import ModelProfile
        assert ModelProfile(reasoning=False).reasoning == "auto"
        assert ModelProfile.from_dict({"reasoning": False}).reasoning == "auto"

    def test_legacy_bool_true_maps_to_on(self):
        from llamacpp_loader.config.store import ModelProfile
        assert ModelProfile(reasoning=True).reasoning == "on"
        assert ModelProfile.from_dict({"reasoning": True}).reasoning == "on"

    def test_round_trips_every_mode(self):
        from llamacpp_loader.config.store import ModelProfile
        for mode in ("auto", "on", "off"):
            p = ModelProfile(profile_name="m", gguf_file="m.gguf", reasoning=mode)
            assert ModelProfile.from_dict(p.to_dict()).reasoning == mode

    def test_unknown_value_falls_back_to_auto(self):
        from llamacpp_loader.config.store import ModelProfile
        assert ModelProfile(reasoning="banana").reasoning == "auto"
        assert ModelProfile.from_dict({"reasoning": "banana"}).reasoning == "auto"

    def test_case_insensitive(self):
        from llamacpp_loader.config.store import ModelProfile
        assert ModelProfile(reasoning="ON").reasoning == "on"

    def test_missing_key_defaults_to_auto(self):
        from llamacpp_loader.config.store import ModelProfile
        assert ModelProfile.from_dict({"profile_name": "m"}).reasoning == "auto"


class TestReasoningBudget:
    """Reasoning budget: None means no flag; -1, 0 and N are real values."""

    def test_default_is_unset(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams().reasoning_budget is None

    def test_zero_and_minus_one_are_kept(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_budget=0).reasoning_budget == 0
        assert InferenceParams(reasoning_budget=-1).reasoning_budget == -1

    def test_clamped_at_minus_one(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_budget=-500).reasoning_budget == -1

    def test_empty_string_means_unset(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_budget="").reasoning_budget is None
        assert InferenceParams(reasoning_budget="   ").reasoning_budget is None

    def test_string_digits_accepted(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_budget="2048").reasoning_budget == 2048

    def test_garbage_means_unset(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_budget="abc").reasoning_budget is None

    def test_setter_validates(self):
        from llamacpp_loader.config.store import InferenceParams
        ip = InferenceParams()
        ip.set_reasoning_budget(1024)
        assert ip.reasoning_budget == 1024
        ip.set_reasoning_budget("")
        assert ip.reasoning_budget is None

    def test_profile_round_trips_budget_and_message(self):
        from llamacpp_loader.config.store import InferenceParams, ModelProfile
        p = ModelProfile(
            profile_name="m", gguf_file="m.gguf",
            inference=InferenceParams(reasoning_budget=3072,
                                      reasoning_budget_message="wrap up"))
        back = ModelProfile.from_dict(p.to_dict())
        assert back.inference.reasoning_budget == 3072
        assert back.inference.reasoning_budget_message == "wrap up"

    def test_budget_travels_with_presets(self):
        """Living in InferenceParams is what makes presets carry the budget."""
        from llamacpp_loader.config.store import InferenceParams, ModelProfile
        p = ModelProfile(profile_name="m", gguf_file="m.gguf",
                         inference=InferenceParams(reasoning_budget=4096))
        p.save_preset("thinking")
        p.inference.set_reasoning_budget(None)
        assert p.load_preset("thinking") is True
        assert p.inference.reasoning_budget == 4096


class TestReasoningEffort:
    def test_default(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams().reasoning_effort == "default"

    def test_valid_levels_kept_case_insensitively(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_effort="HIGH").reasoning_effort == "high"
        assert InferenceParams(reasoning_effort="xhigh").reasoning_effort == "xhigh"

    def test_unknown_falls_back_to_default(self):
        from llamacpp_loader.config.store import InferenceParams
        assert InferenceParams(reasoning_effort="banana").reasoning_effort == "default"
        assert InferenceParams(reasoning_effort=None).reasoning_effort == "default"

    def test_setter_validates(self):
        from llamacpp_loader.config.store import InferenceParams
        ip = InferenceParams()
        ip.set_reasoning_effort("medium")
        assert ip.reasoning_effort == "medium"
        ip.set_reasoning_effort("nope")
        assert ip.reasoning_effort == "default"
