"""Deploy the API + model to a Hugging Face Docker Space, with a Hub model repo as registry.

Prerequisite: `hf auth login` with a WRITE token (the token is read from the local HF cache).
Usage: python scripts/deploy_hf_space.py [--space bean-disease-api] [--model-repo bean-disease-model]
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ["bean_model.keras", "bean_model.h5", "bean_model_meta.json"]


def stage_space(dest: Path):
    """Copy exactly what the Space image needs into dest."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp.*")
    shutil.copytree(ROOT / "src", dest / "src", ignore=ignore)
    shutil.copytree(ROOT / "api", dest / "api", ignore=ignore)
    (dest / "docker").mkdir()
    shutil.copy2(ROOT / "docker" / "entrypoint.sh", dest / "docker" / "entrypoint.sh")
    (dest / "models").mkdir()
    for f in MODEL_FILES:
        shutil.copy2(ROOT / "models" / f, dest / "models" / f)
    shutil.copy2(ROOT / "requirements-api.txt", dest / "requirements-api.txt")
    shutil.copy2(ROOT / "deploy" / "hf_space" / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "hf_space" / "README.md", dest / "README.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="bean-disease-api")
    parser.add_argument("--model-repo", default="bean-disease-model")
    parser.add_argument("--retrain-threshold", default="30")
    parser.add_argument("--retrain-epochs", default="2")
    args = parser.parse_args()

    token = get_token()
    if not token:
        sys.exit("Not logged in: run `hf auth login` with a WRITE token first.")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    space_id, model_repo = f"{user}/{args.space}", f"{user}/{args.model_repo}"

    # 1. model registry, seeded with the current local model if empty
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    if "bean_model_meta.json" not in api.list_repo_files(model_repo):
        api.upload_folder(repo_id=model_repo, folder_path=str(ROOT / "models"), allow_patterns=MODEL_FILES,
                          commit_message="model v1: baseline")
        print(f"Seeded registry {model_repo} with the baseline model")

    # 2. Space + configuration (secrets are encrypted and never shown in the Space UI)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space_id, "HF_TOKEN", token)
    api.add_space_variable(space_id, "HF_MODEL_REPO", model_repo)
    api.add_space_variable(space_id, "RETRAIN_THRESHOLD", args.retrain_threshold)
    api.add_space_variable(space_id, "RETRAIN_EPOCHS", args.retrain_epochs)

    # 3. code + seed model (triggers a rebuild)
    with tempfile.TemporaryDirectory() as tmp:
        stage_space(Path(tmp))
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp,
                          commit_message="Deploy bean disease API")

    host = f"https://{user}-{args.space}.hf.space".lower().replace("_", "-")
    print(f"Space:     https://huggingface.co/spaces/{space_id}")
    print(f"API:       {host}/docs")
    print(f"Registry:  https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
