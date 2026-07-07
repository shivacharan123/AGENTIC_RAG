"""
Configuration loader for the Agentic RAG system.
Loads config.yaml to initialize retrieval parameters and validation settings.
"""

import yaml
from pathlib import Path
from typing import Any

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Resolution order when config_path is not provided:
      1) Repo-root config.yaml (Path(.../config.yaml))
      2) This package's config.yaml (agentic_rag/prompts/config.yaml)
    """
    repo_root_candidate = Path(__file__).resolve().parents[2] / "config.yaml"
    package_candidate = Path(__file__).resolve().parent / "config.yaml"

    # Resolve the target path based on user input or fallback to defaults
    if config_path is not None:
        target_path = Path(config_path)
    else:
        target_path = repo_root_candidate if repo_root_candidate.exists() else package_candidate

    if not target_path.exists():
        # Python 3.11 handles exceptions faster and allows for more precise tracebacks
        raise FileNotFoundError(
            "Configuration file 'config.yaml' not found.\n"
            f"Tried:\n- {repo_root_candidate.resolve()}\n- {package_candidate.resolve()}\n"
            "Please provide config_path or ensure the file exists in one of the locations above."
        )

    # Using pathlib's built-in .open() method for cleaner context management
    with target_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data or {}
