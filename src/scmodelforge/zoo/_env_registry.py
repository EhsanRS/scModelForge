"""On-disk environment metadata management for isolated model environments."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENV_DIR = "~/.cache/scmodelforge/envs"


@dataclass
class EnvInfo:
    """Metadata for an installed isolated model environment.

    Attributes
    ----------
    model_name
        Short model identifier (e.g. ``"geneformer"``).
    env_path
        Root directory for this environment.
    venv_path
        Path to the virtualenv inside *env_path*.
    python_path
        Path to the Python interpreter in the virtualenv.
    deps
        List of pip requirements installed in the environment.
    python_version
        Python version string (e.g. ``"3.10"``).
    created_at
        ISO-8601 timestamp of environment creation.
    status
        Current state: ``"installing"``, ``"installed"``, or ``"error"``.
    """

    model_name: str
    env_path: str
    venv_path: str
    python_path: str
    deps: list[str] = field(default_factory=list)
    python_version: str = ""
    created_at: str = ""
    status: str = "installing"


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    """Expand and resolve the base environment directory."""
    if base_dir is None:
        base_dir = DEFAULT_ENV_DIR
    return Path(base_dir).expanduser().resolve()


def get_env_dir(model_name: str, base_dir: str | Path | None = None) -> Path:
    """Return the environment directory for *model_name*.

    Parameters
    ----------
    model_name
        Model identifier (e.g. ``"geneformer"``).
    base_dir
        Base directory for all environments. Defaults to ``~/.cache/scmodelforge/envs``.

    Returns
    -------
    Path
        ``<base_dir>/<model_name>/``
    """
    return _resolve_base_dir(base_dir) / model_name


def load_env_info(model_name: str, base_dir: str | Path | None = None) -> EnvInfo | None:
    """Load environment metadata from ``env.json``.

    Returns ``None`` if the env directory or metadata file does not exist.
    """
    env_dir = get_env_dir(model_name, base_dir)
    meta_path = env_dir / "env.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return EnvInfo(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("Failed to load env metadata for %s: %s", model_name, exc)
        return None


def save_env_info(info: EnvInfo) -> None:
    """Write environment metadata to ``<env_path>/env.json``."""
    env_dir = Path(info.env_path)
    env_dir.mkdir(parents=True, exist_ok=True)
    meta_path = env_dir / "env.json"
    meta_path.write_text(json.dumps(asdict(info), indent=2) + "\n", encoding="utf-8")


def list_installed_envs(base_dir: str | Path | None = None) -> list[EnvInfo]:
    """Return metadata for all installed environments.

    Only returns entries whose ``env.json`` can be loaded successfully.
    """
    resolved = _resolve_base_dir(base_dir)
    if not resolved.is_dir():
        return []
    envs: list[EnvInfo] = []
    for child in sorted(resolved.iterdir()):
        if child.is_dir():
            info = load_env_info(child.name, base_dir)
            if info is not None:
                envs.append(info)
    return envs


def remove_env(model_name: str, base_dir: str | Path | None = None) -> bool:
    """Remove an environment directory.

    Returns ``True`` if the directory existed and was removed, ``False`` otherwise.
    """
    env_dir = get_env_dir(model_name, base_dir)
    if not env_dir.exists():
        return False
    shutil.rmtree(env_dir)
    return True


def is_env_installed(model_name: str, base_dir: str | Path | None = None) -> bool:
    """Check if a model environment is installed and has status ``"installed"``."""
    info = load_env_info(model_name, base_dir)
    return info is not None and info.status == "installed"


def create_env_info(
    model_name: str,
    base_dir: str | Path | None = None,
    *,
    deps: list[str] | None = None,
    python_version: str = "",
    status: str = "installing",
) -> EnvInfo:
    """Create an ``EnvInfo`` with standard paths derived from *model_name*.

    This is a convenience factory — it does NOT create the actual virtualenv.
    """
    env_dir = get_env_dir(model_name, base_dir)
    venv_dir = env_dir / "venv"
    python_path = venv_dir / "bin" / "python"
    return EnvInfo(
        model_name=model_name,
        env_path=str(env_dir),
        venv_path=str(venv_dir),
        python_path=str(python_path),
        deps=deps or [],
        python_version=python_version,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        status=status,
    )
