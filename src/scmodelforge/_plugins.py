"""Plugin discovery for third-party tokenizers, models, and benchmarks.

Third-party packages register components via Python entry points.  For example,
a plugin's ``pyproject.toml`` might contain::

    [project.entry-points."scmodelforge.tokenizers"]
    my_tokenizer = "my_package.module:MyTokenizerClass"

Discovery is lazy — entry points are scanned on the first call to
``get_*()`` or ``list_*()`` in each registry.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Entry-point group names that third-party packages use to register plugins.
TOKENIZER_GROUP = "scmodelforge.tokenizers"
MODEL_GROUP = "scmodelforge.models"
BENCHMARK_GROUP = "scmodelforge.benchmarks"


def discover_plugins(group: str, registry: dict[str, type]) -> None:
    """Scan *group* entry points and insert discovered classes into *registry*.

    Built-in components (already in *registry* from ``@register_*`` decorators)
    take precedence: if an entry-point name collides with an existing key the
    plugin is skipped with a warning.

    Any exception raised while loading an entry point is caught and logged so
    that a single broken plugin cannot prevent the rest from loading.
    """
    from importlib.metadata import entry_points

    for ep in entry_points(group=group):
        if ep.name in registry:
            logger.warning(
                "Plugin '%s' from '%s' skipped: name already registered",
                ep.name,
                ep.dist,
            )
            continue
        try:
            cls = ep.load()
        except Exception:
            logger.warning(
                "Failed to load plugin '%s' from '%s'",
                ep.name,
                ep.dist,
                exc_info=True,
            )
            continue
        registry[ep.name] = cls
