"""Tests for the external model registry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scmodelforge.zoo.base import BaseModelAdapter, ExternalModelInfo
from scmodelforge.zoo.registry import (
    _REGISTRY,
    _state,
    get_external_model,
    list_external_models,
    register_external_model,
)


class TestRegisterExternalModel:
    def test_decorator_registers(self):
        """A decorated class is added to the registry."""
        # Use a unique name to avoid conflicts
        name = "_test_register_unique_xyz"
        try:

            @register_external_model(name)
            class _TestAdapter(BaseModelAdapter):
                @property
                def info(self):
                    return ExternalModelInfo(name=name)

                def _require_package(self):
                    pass

                def load_model(self):
                    pass

                def extract_embeddings(self, adata, *, batch_size=None, device=None):
                    pass

                def _get_model_genes(self):
                    return []

            assert name in _REGISTRY
            assert _REGISTRY[name] is _TestAdapter
        finally:
            _REGISTRY.pop(name, None)

    def test_duplicate_name_raises(self):
        """Registering the same name twice raises ValueError."""
        name = "_test_dup_unique_xyz"
        try:

            @register_external_model(name)
            class _Adapter1(BaseModelAdapter):
                @property
                def info(self):
                    return ExternalModelInfo(name=name)

                def _require_package(self):
                    pass

                def load_model(self):
                    pass

                def extract_embeddings(self, adata, *, batch_size=None, device=None):
                    pass

                def _get_model_genes(self):
                    return []

            with pytest.raises(ValueError, match="already registered"):

                @register_external_model(name)
                class _Adapter2(BaseModelAdapter):
                    @property
                    def info(self):
                        return ExternalModelInfo(name=name)

                    def _require_package(self):
                        pass

                    def load_model(self):
                        pass

                    def extract_embeddings(self, adata, *, batch_size=None, device=None):
                        pass

                    def _get_model_genes(self):
                        return []

        finally:
            _REGISTRY.pop(name, None)


class TestGetExternalModel:
    def test_get_existing(self):
        """get_external_model returns an instance of the registered class."""
        # "geneformer" is registered by importing zoo/__init__.py
        adapter = get_external_model("geneformer")
        assert adapter.info.name == "geneformer"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown external model"):
            get_external_model("nonexistent_model_xyz")

    def test_kwargs_forwarded(self):
        adapter = get_external_model("geneformer", device="cuda", batch_size=128)
        assert adapter._device == "cuda"
        assert adapter._batch_size == 128


class TestListExternalModels:
    def test_includes_geneformer(self):
        names = list_external_models()
        assert "geneformer" in names

    def test_returns_sorted(self):
        names = list_external_models()
        assert names == sorted(names)


class TestPluginDiscovery:
    def test_plugin_discovered(self):
        """A mock entry point for scmodelforge.zoo is loaded."""
        name = "_plugin_test_adapter"

        class _PluginAdapter:
            pass

        ep = MagicMock()
        ep.name = name
        ep.dist = SimpleNamespace(name="fake-zoo-plugin")
        ep.load.return_value = _PluginAdapter

        # Reset plugin state and clear any previous plugin
        old_state = _state["plugins_loaded"]
        _state["plugins_loaded"] = False
        _REGISTRY.pop(name, None)

        try:
            with patch("importlib.metadata.entry_points", return_value=[ep]):
                names = list_external_models()
                assert name in names
                assert _REGISTRY[name] is _PluginAdapter
        finally:
            _REGISTRY.pop(name, None)
            _state["plugins_loaded"] = old_state

    def test_plugin_skipped_on_collision(self):
        """Plugin with same name as built-in is skipped."""
        ep = MagicMock()
        ep.name = "geneformer"  # Collides with built-in
        ep.dist = SimpleNamespace(name="fake-geneformer-plugin")

        old_state = _state["plugins_loaded"]
        _state["plugins_loaded"] = False

        try:
            with patch("importlib.metadata.entry_points", return_value=[ep]):
                list_external_models()
                # Built-in should still be there, plugin's load() never called
                ep.load.assert_not_called()
        finally:
            _state["plugins_loaded"] = old_state
