"""Shared pytest fixtures for llamacpp-loader tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

tk = pytest.importorskip("tkinter")

@pytest.fixture(scope="session")
def tk_root():
    """Session-scoped Tk root window shared across all GUI tests.
    
    Creates one Tk instance per test session and yields it to tests.
    Tests must call root.destroy() or the fixture will clean up after the session.
    """
    root = tk.Tk()
    root.withdraw()  # Hide during testing; tests can deiconify if needed
    yield root
    root.destroy()

from unittest.mock import MagicMock

import pytest

from llamacpp_loader.config.store import ConfigStore, ModelProfile
from llamacpp_loader.process_manager.manager import ServerConfig


@pytest.fixture()
def sample_config_path(tmp_path):
    """Create a temporary JSON file with valid settings."""
    store = ConfigStore(path=tmp_path / "settings.json")
    profile = store.create_default_profile(
        display_name="TestModel",
        model_path=str(tmp_path / "models"),
        gguf_file="test-model.gguf",
    )
    store.add(profile)
    return tmp_path / "settings.json"


@pytest.fixture()
def valid_config_store(sample_config_path):
    """A ConfigStore loaded from a real JSON file."""
    return ConfigStore(path=sample_config_path)


@pytest.fixture()
def sample_profile():
    """Create a fresh ModelProfile for testing."""
    profile = ModelProfile(
        profile_name="test-model",
        display_name="Test Model",
        model_path="/tmp/models",
        gguf_file="llama-3.2-3b-q4_k_m.gguf",
    )
    return profile


@pytest.fixture()
def sample_server_config(sample_profile):
    """Create a ServerConfig from a sample Profile."""
    server = sample_profile.server
    inference = sample_profile.inference
    sampling = sample_profile.sampling
    model_path, port = sample_profile.to_server_config()
    return ServerConfig(
        model_path=model_path,
        host=server.host,
        port=port,
        ctx_size=inference.ctx_size,
        gpu_layers=inference.gpu_layers,
        n_threads=inference.n_threads,
        temperature=sampling.temperature,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
    )


@pytest.fixture()
def mock_process_manager(sample_server_config):
    """ProcessManager with subprocess.Popen mocked out."""
    from llamacpp_loader.process_manager.manager import ProcessManager

    mgr = ProcessManager(log_callback=None)
    mgr._process = MagicMock()  # type: ignore[assignment]
    mgr._last_config = sample_server_config
    return mgr
