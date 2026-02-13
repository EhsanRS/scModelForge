"""HuggingFace Hub integration for saving, loading, and sharing models.

Provides standalone functions (not mixins) so models remain plain ``nn.Module``
subclasses. Local save/load works without optional dependencies; Hub operations
require ``huggingface-hub`` and optionally ``safetensors``.

Saved model directory format::

    model_dir/
      config.json           # ModelConfig + TokenizerConfig + metadata
      model.safetensors     # Weights (safetensors preferred, torch fallback)
      gene_vocab.json       # Gene vocabulary (optional)
      README.md             # Auto-generated model card
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from scmodelforge import __version__
from scmodelforge.config.schema import AttentionConfig, MaskingConfig, ModelConfig, TokenizerConfig
from scmodelforge.models.registry import get_model

if TYPE_CHECKING:
    import torch.nn as nn

    from scmodelforge.data.gene_vocab import GeneVocab

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_NAME = "config.json"
WEIGHTS_SAFETENSORS_NAME = "model.safetensors"
WEIGHTS_TORCH_NAME = "model.pt"
VOCAB_NAME = "gene_vocab.json"
README_NAME = "README.md"

# ---------------------------------------------------------------------------
# Internal helpers — config serialization
# ---------------------------------------------------------------------------


def _build_config_dict(
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig | None = None,
) -> dict[str, Any]:
    """Serialize model and tokenizer configs to a JSON-safe dict."""
    result: dict[str, Any] = {
        "scmodelforge_version": __version__,
        "model_type": model_config.architecture,
        "model_config": asdict(model_config),
    }
    if tokenizer_config is not None:
        result["tokenizer_config"] = asdict(tokenizer_config)
    return result


def _config_dict_to_model_config(config_dict: dict[str, Any]) -> ModelConfig:
    """Reconstruct a ``ModelConfig`` from a saved config dict.

    Unknown fields are silently ignored for forward compatibility.
    """
    raw = config_dict["model_config"]
    # Filter to only known fields
    known_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    # Reconstruct nested AttentionConfig
    if "attention" in filtered and isinstance(filtered["attention"], dict):
        attn_fields = {f.name for f in AttentionConfig.__dataclass_fields__.values()}
        attn_filtered = {k: v for k, v in filtered["attention"].items() if k in attn_fields}
        filtered["attention"] = AttentionConfig(**attn_filtered)
    return ModelConfig(**filtered)


def _config_dict_to_tokenizer_config(config_dict: dict[str, Any]) -> TokenizerConfig | None:
    """Reconstruct a ``TokenizerConfig`` from a saved config dict.

    Returns ``None`` if no tokenizer config was saved. Handles nested
    ``MaskingConfig``.
    """
    raw = config_dict.get("tokenizer_config")
    if raw is None:
        return None
    known_fields = {f.name for f in TokenizerConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    # Reconstruct nested MaskingConfig
    if "masking" in filtered and isinstance(filtered["masking"], dict):
        masking_fields = {f.name for f in MaskingConfig.__dataclass_fields__.values()}
        masking_filtered = {k: v for k, v in filtered["masking"].items() if k in masking_fields}
        filtered["masking"] = MaskingConfig(**masking_filtered)
    return TokenizerConfig(**filtered)


# ---------------------------------------------------------------------------
# Internal helpers — weights I/O
# ---------------------------------------------------------------------------


def _save_weights(state_dict: dict[str, torch.Tensor], directory: Path, *, safe_serialization: bool = True) -> str:
    """Save model weights, preferring safetensors if available.

    Returns the filename used.
    """
    if safe_serialization:
        try:
            from safetensors.torch import save_file

            path = directory / WEIGHTS_SAFETENSORS_NAME
            save_file(state_dict, str(path))
            return WEIGHTS_SAFETENSORS_NAME
        except ImportError:
            logger.warning("safetensors not installed, falling back to torch.save")

    path = directory / WEIGHTS_TORCH_NAME
    torch.save(state_dict, path)
    return WEIGHTS_TORCH_NAME


def _load_weights(directory: Path, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load model weights from a directory (safetensors preferred)."""
    safetensors_path = directory / WEIGHTS_SAFETENSORS_NAME
    torch_path = directory / WEIGHTS_TORCH_NAME

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file

            return load_file(str(safetensors_path), device=device)
        except ImportError:
            logger.warning("safetensors not installed, trying torch format")

    if torch_path.exists():
        return torch.load(torch_path, map_location=device, weights_only=True)

    msg = f"No weights file found in {directory} (looked for {WEIGHTS_SAFETENSORS_NAME} and {WEIGHTS_TORCH_NAME})"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Internal helpers — Hub resolution
# ---------------------------------------------------------------------------


def _is_hub_repo_id(path_or_id: str) -> bool:
    """Heuristic: Hub repo IDs are ``user/model`` — one slash, no leading slash."""
    if "/" not in path_or_id:
        return False
    # Absolute paths and multi-level paths are not hub IDs
    if path_or_id.startswith("/") or path_or_id.startswith("."):
        return False
    # Hub IDs have exactly one slash: "user/model"
    if path_or_id.count("/") != 1:
        return False
    # If it exists as a local path, treat as local
    return not Path(path_or_id).exists()


def _resolve_model_directory(
    path_or_repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Resolve a local path or Hub repo ID to a local directory."""
    local_path = Path(path_or_repo_id)
    if local_path.is_dir():
        return local_path

    if _is_hub_repo_id(path_or_repo_id):
        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=path_or_repo_id,
                revision=revision,
                token=token,
            )
            return Path(cache_dir)
        except ImportError:
            msg = (
                "huggingface-hub is required to download models from the Hub. "
                "Install it with: pip install huggingface-hub"
            )
            raise ImportError(msg) from None

    msg = f"Path {path_or_repo_id!r} does not exist and does not look like a Hub repo ID"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Internal helpers — model card
# ---------------------------------------------------------------------------


def _generate_model_card(
    config_dict: dict[str, Any],
    save_directory: Path,
) -> None:
    """Write a README.md model card with YAML frontmatter."""
    model_type = config_dict.get("model_type", "unknown")
    model_cfg = config_dict.get("model_config", {})
    version = config_dict.get("scmodelforge_version", __version__)

    hidden_dim = model_cfg.get("hidden_dim", "?")
    num_layers = model_cfg.get("num_layers", "?")
    num_heads = model_cfg.get("num_heads", "?")
    vocab_size = model_cfg.get("vocab_size", "?")

    card = f"""---
library_name: scmodelforge
tags:
- single-cell
- foundation-model
- gene-expression
---

# scModelForge Model

**Architecture:** {model_type}
**scModelForge version:** {version}

## Model Details

| Parameter | Value |
|-----------|-------|
| Hidden dimension | {hidden_dim} |
| Layers | {num_layers} |
| Attention heads | {num_heads} |
| Vocabulary size | {vocab_size} |

## Usage

```python
from scmodelforge.models.hub import load_pretrained

model, config = load_pretrained("{save_directory.name}")
```
"""
    readme_path = save_directory / README_NAME
    readme_path.write_text(card)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_pretrained(
    model: nn.Module,
    save_directory: str | Path,
    *,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig | None = None,
    gene_vocab: GeneVocab | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Save a model, config, and optional vocab to a directory.

    Creates a self-contained model directory that can be loaded with
    :func:`load_pretrained` or uploaded with :func:`push_to_hub`.

    Parameters
    ----------
    model
        The model to save.
    save_directory
        Directory to save to (created if it doesn't exist).
    model_config
        Model configuration. ``vocab_size`` must be set.
    tokenizer_config
        Optional tokenizer configuration to include.
    gene_vocab
        Optional gene vocabulary to include.
    safe_serialization
        If ``True`` (default), use safetensors format. Falls back to
        ``torch.save`` if safetensors is not installed.

    Returns
    -------
    Path
        Path to the saved directory.
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    if model_config.vocab_size is None:
        msg = "model_config.vocab_size must be set before saving"
        raise ValueError(msg)

    # Save config
    config_dict = _build_config_dict(model_config, tokenizer_config)
    config_path = save_directory / CONFIG_NAME
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save weights (without "model." prefix)
    state_dict = model.state_dict()
    _save_weights(state_dict, save_directory, safe_serialization=safe_serialization)

    # Save gene vocab if provided
    if gene_vocab is not None:
        gene_vocab.save(save_directory / VOCAB_NAME)

    # Generate model card
    _generate_model_card(config_dict, save_directory)

    logger.info("Model saved to %s", save_directory)
    return save_directory


def load_pretrained(
    path_or_repo_id: str | Path,
    *,
    device: str = "cpu",
    revision: str | None = None,
    token: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a model from a local directory or HuggingFace Hub repo.

    Parameters
    ----------
    path_or_repo_id
        Local directory path or Hub repo ID (e.g. ``"user/model-name"``).
    device
        Device to load weights onto.
    revision
        Hub revision (branch, tag, or commit hash).
    token
        HuggingFace Hub token for private repos.

    Returns
    -------
    tuple[nn.Module, dict]
        The loaded model and the raw config dict.
    """
    directory = _resolve_model_directory(str(path_or_repo_id), revision=revision, token=token)

    # Load config
    config_path = directory / CONFIG_NAME
    with open(config_path) as f:
        config_dict = json.load(f)

    # Reconstruct model from config
    model_config = _config_dict_to_model_config(config_dict)
    model = get_model(config_dict["model_type"], model_config)

    # Load weights
    state_dict = _load_weights(directory, device=device)
    model.load_state_dict(state_dict)
    model.to(device)

    logger.info("Model loaded from %s", directory)
    return model, config_dict


def load_pretrained_with_vocab(
    path_or_repo_id: str | Path,
    *,
    device: str = "cpu",
    revision: str | None = None,
    token: str | None = None,
) -> tuple[nn.Module, dict[str, Any], GeneVocab | None]:
    """Load a model and optional gene vocabulary.

    Same as :func:`load_pretrained` but also loads the gene vocabulary
    if one was saved with the model.

    Parameters
    ----------
    path_or_repo_id
        Local directory path or Hub repo ID.
    device
        Device to load weights onto.
    revision
        Hub revision.
    token
        HuggingFace Hub token.

    Returns
    -------
    tuple[nn.Module, dict, GeneVocab | None]
        The loaded model, config dict, and gene vocabulary (or ``None``).
    """
    from scmodelforge.data.gene_vocab import GeneVocab

    directory = _resolve_model_directory(str(path_or_repo_id), revision=revision, token=token)

    # Load config
    config_path = directory / CONFIG_NAME
    with open(config_path) as f:
        config_dict = json.load(f)

    # Reconstruct model
    model_config = _config_dict_to_model_config(config_dict)
    model = get_model(config_dict["model_type"], model_config)

    # Load weights
    state_dict = _load_weights(directory, device=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Load vocab if present
    vocab_path = directory / VOCAB_NAME
    vocab = GeneVocab.from_file(vocab_path) if vocab_path.exists() else None

    logger.info("Model loaded from %s (vocab=%s)", directory, vocab is not None)
    return model, config_dict, vocab


def push_to_hub(
    model_directory: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload scModelForge model",
    token: str | None = None,
) -> str:
    """Upload a saved model directory to HuggingFace Hub.

    Parameters
    ----------
    model_directory
        Local directory containing the saved model (from :func:`save_pretrained`).
    repo_id
        Hub repository ID (e.g. ``"username/model-name"``).
    private
        If ``True``, create a private repository.
    commit_message
        Git commit message for the upload.
    token
        HuggingFace Hub token.

    Returns
    -------
    str
        URL of the uploaded repository.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        msg = (
            "huggingface-hub is required to push models to the Hub. "
            "Install it with: pip install huggingface-hub"
        )
        raise ImportError(msg) from None

    model_directory = Path(model_directory)
    if not model_directory.is_dir():
        msg = f"Model directory does not exist: {model_directory}"
        raise FileNotFoundError(msg)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private, token=token)
    api.upload_folder(
        folder_path=str(model_directory),
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Model pushed to %s", url)
    return url
