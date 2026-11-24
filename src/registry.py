"""Model registry on the Hugging Face Hub.

Free hosting (e.g. Hugging Face Spaces) has an ephemeral disk, so a retrained model would
be lost on restart. The registry keeps the latest model in a Hub model repo:
  * pull_if_newer() on startup downloads the registry model if it is newer than the local one
  * push(meta) after a successful retrain uploads the new version (one Hub commit per version)

Configured with HF_TOKEN and HF_MODEL_REPO. If either is missing, every function is a no-op,
so local and docker-compose runs behave exactly as before.
"""
import json
import os
import shutil
import tempfile

from src.config import H5_PATH, META_PATH, MODEL_PATH, MODELS_DIR

FILES = [MODEL_PATH.name, H5_PATH.name, META_PATH.name]


def _config():
    token = os.getenv("HF_TOKEN")
    repo = os.getenv("HF_MODEL_REPO")
    return (token, repo) if token and repo else (None, None)


def enabled():
    return _config()[0] is not None


def _local_version():
    if not META_PATH.exists():
        return 0
    return int(json.loads(META_PATH.read_text()).get("version", 0))


def remote_meta():
    token, repo = _config()
    if not token:
        return None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo, META_PATH.name, token=token, force_download=True)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return json.loads(open(path, encoding="utf-8").read())


def pull_if_newer():
    """Download the registry model if its version is newer. Returns the pulled version or None."""
    meta = remote_meta()
    if not meta or int(meta.get("version", 0)) <= _local_version():
        return None
    token, repo = _config()
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in FILES:
            src = hf_hub_download(repo, name, token=token, local_dir=tmp)
            shutil.copy2(src, MODELS_DIR / f"{name}.pulling")
    # model files first, meta last, each an atomic rename (replicas reload on mtime change)
    for name in (H5_PATH.name, MODEL_PATH.name, META_PATH.name):
        os.replace(MODELS_DIR / f"{name}.pulling", MODELS_DIR / name)
    return int(meta["version"])


def push(meta):
    """Upload the current model files as a new registry version. Returns the commit URL or None."""
    token, repo = _config()
    if not token:
        return None
    from huggingface_hub import HfApi

    m = meta.get("metrics", {})
    msg = f"model v{meta.get('version')}: acc={m.get('accuracy', 0):.4f} f1={m.get('f1', 0):.4f} ({meta.get('kind', 'train')})"
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        repo_id=repo,
        folder_path=str(MODELS_DIR),
        allow_patterns=FILES,
        commit_message=msg,
    )
    return getattr(info, "commit_url", str(info))
