"""Registry tests with the Hugging Face Hub mocked out (no network)."""
import json
import sys
import types

import pytest

import src.registry as registry


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(registry, "MODEL_PATH", tmp_path / "bean_model.keras")
    monkeypatch.setattr(registry, "H5_PATH", tmp_path / "bean_model.h5")
    monkeypatch.setattr(registry, "META_PATH", tmp_path / "bean_model_meta.json")
    return tmp_path


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A fake huggingface_hub whose 'remote repo' is a local folder."""
    remote = tmp_path / "remote"
    remote.mkdir()
    calls = {"uploads": []}

    class NotFound(Exception):
        pass

    def hf_hub_download(repo, filename, token=None, force_download=False, local_dir=None):
        p = remote / filename
        if not p.exists():
            raise NotFound(filename)
        return str(p)

    class HfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, **kwargs):
            calls["uploads"].append(kwargs)
            return types.SimpleNamespace(commit_url="https://hf/commit/1")

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download, hub.HfApi = hf_hub_download, HfApi
    errors = types.ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = errors.RepositoryNotFoundError = NotFound
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_MODEL_REPO", "someone/bean-model")
    return remote, calls


def _write_model(folder, version):
    (folder / "bean_model.keras").write_text(f"keras v{version}")
    (folder / "bean_model.h5").write_text(f"h5 v{version}")
    (folder / "bean_model_meta.json").write_text(json.dumps({"version": version, "metrics": {"accuracy": 0.9, "f1": 0.9}}))


def test_noop_without_config(models_dir, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    assert not registry.enabled()
    assert registry.pull_if_newer() is None
    assert registry.push({"version": 1}) is None


def test_pull_when_remote_is_newer(models_dir, fake_hub):
    remote, _ = fake_hub
    _write_model(models_dir, 1)
    _write_model(remote, 3)
    assert registry.pull_if_newer() == 3
    assert json.loads((models_dir / "bean_model_meta.json").read_text())["version"] == 3
    assert (models_dir / "bean_model.keras").read_text() == "keras v3"
    assert not list(models_dir.glob("*.pulling"))


def test_no_pull_when_local_is_current(models_dir, fake_hub):
    remote, _ = fake_hub
    _write_model(models_dir, 2)
    _write_model(remote, 2)
    assert registry.pull_if_newer() is None
    assert (models_dir / "bean_model.keras").read_text() == "keras v2"


def test_no_pull_when_registry_empty(models_dir, fake_hub):
    _write_model(models_dir, 1)
    assert registry.pull_if_newer() is None


def test_push_uploads_model_files(models_dir, fake_hub):
    _, calls = fake_hub
    _write_model(models_dir, 4)
    url = registry.push({"version": 4, "kind": "retrain", "metrics": {"accuracy": 0.93, "f1": 0.92}})
    assert url == "https://hf/commit/1"
    up = calls["uploads"][0]
    assert up["repo_id"] == "someone/bean-model"
    assert set(up["allow_patterns"]) == {"bean_model.keras", "bean_model.h5", "bean_model_meta.json"}
    assert up["commit_message"].startswith("model v4: acc=0.9300")
