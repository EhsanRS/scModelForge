"""Tests for the plugin discovery system."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scmodelforge._plugins import BENCHMARK_GROUP, MODEL_GROUP, TOKENIZER_GROUP, discover_plugins

# ---------------------------------------------------------------------------
# Helpers — fake entry points
# ---------------------------------------------------------------------------


def _make_ep(name: str, cls: type | None = None, *, error: Exception | None = None) -> MagicMock:
    """Create a mock entry point with *name*, returning *cls* on load().

    If *error* is given, ``load()`` will raise it instead.
    """
    ep = MagicMock()
    ep.name = name
    ep.dist = SimpleNamespace(name=f"fake-{name}-pkg")
    if error is not None:
        ep.load.side_effect = error
    else:
        ep.load.return_value = cls if cls is not None else type(name, (), {})
    return ep


# ===========================================================================
# Unit tests for discover_plugins()
# ===========================================================================


class TestDiscoverPlugins:
    """Unit tests for :func:`discover_plugins`."""

    @patch("importlib.metadata.entry_points")
    def test_discover_loads_plugin(self, mock_ep):
        """Happy path: a single plugin entry point is loaded into the registry."""

        class MyPlugin:
            pass

        mock_ep.return_value = [_make_ep("my_plugin", MyPlugin)]
        registry: dict[str, type] = {}
        discover_plugins("some.group", registry)
        assert registry["my_plugin"] is MyPlugin

    @patch("importlib.metadata.entry_points")
    def test_discover_multiple_plugins(self, mock_ep):
        """Two entry points → both registered."""

        class PluginA:
            pass

        class PluginB:
            pass

        mock_ep.return_value = [
            _make_ep("plugin_a", PluginA),
            _make_ep("plugin_b", PluginB),
        ]
        registry: dict[str, type] = {}
        discover_plugins("some.group", registry)
        assert registry["plugin_a"] is PluginA
        assert registry["plugin_b"] is PluginB

    @patch("importlib.metadata.entry_points")
    def test_discover_skips_existing_name(self, mock_ep, caplog):
        """If the name already exists in the registry, the plugin is skipped."""

        class Existing:
            pass

        class PluginDupe:
            pass

        registry: dict[str, type] = {"foo": Existing}
        mock_ep.return_value = [_make_ep("foo", PluginDupe)]

        with caplog.at_level(logging.WARNING, logger="scmodelforge._plugins"):
            discover_plugins("some.group", registry)

        assert registry["foo"] is Existing  # Built-in wins
        assert "skipped: name already registered" in caplog.text

    @patch("importlib.metadata.entry_points")
    def test_discover_skips_on_import_error(self, mock_ep, caplog):
        """A plugin whose load() raises ImportError is skipped gracefully."""
        mock_ep.return_value = [
            _make_ep("bad", error=ImportError("no such module")),
            _make_ep("good", int),
        ]
        registry: dict[str, type] = {}

        with caplog.at_level(logging.WARNING, logger="scmodelforge._plugins"):
            discover_plugins("some.group", registry)

        assert "bad" not in registry
        assert registry["good"] is int
        assert "Failed to load plugin 'bad'" in caplog.text

    @patch("importlib.metadata.entry_points")
    def test_discover_skips_on_attribute_error(self, mock_ep, caplog):
        """A plugin whose load() raises AttributeError is skipped gracefully."""
        mock_ep.return_value = [_make_ep("broken", error=AttributeError("missing attr"))]
        registry: dict[str, type] = {}

        with caplog.at_level(logging.WARNING, logger="scmodelforge._plugins"):
            discover_plugins("some.group", registry)

        assert "broken" not in registry
        assert "Failed to load plugin 'broken'" in caplog.text

    @patch("importlib.metadata.entry_points")
    def test_discover_empty_group(self, mock_ep):
        """No entry points for the group → registry unchanged."""
        mock_ep.return_value = []
        registry: dict[str, type] = {"existing": str}
        discover_plugins("empty.group", registry)
        assert registry == {"existing": str}

    def test_group_constants(self):
        """Entry-point group constants have expected values."""
        assert TOKENIZER_GROUP == "scmodelforge.tokenizers"
        assert MODEL_GROUP == "scmodelforge.models"
        assert BENCHMARK_GROUP == "scmodelforge.benchmarks"


# ===========================================================================
# Integration tests — _ensure_plugins() in each registry
# ===========================================================================


class TestTokenizerRegistryPlugins:
    """Plugin discovery integration for the tokenizer registry."""

    def setup_method(self):
        from scmodelforge.tokenizers import registry as reg

        self._reg = reg
        self._orig_state = reg._state["plugins_loaded"]
        reg._state["plugins_loaded"] = False

    def teardown_method(self):
        self._reg._state["plugins_loaded"] = self._orig_state
        self._reg._REGISTRY.pop("plugin_tok", None)

    @patch("importlib.metadata.entry_points")
    def test_list_triggers_discovery(self, mock_ep):
        class FakeTok:
            pass

        mock_ep.return_value = [_make_ep("plugin_tok", FakeTok)]
        names = self._reg.list_tokenizers()
        assert "plugin_tok" in names
        mock_ep.assert_called_once_with(group=TOKENIZER_GROUP)

    @patch("importlib.metadata.entry_points")
    def test_get_triggers_discovery(self, mock_ep):
        class FakeTok:
            pass

        mock_ep.return_value = [_make_ep("plugin_tok", FakeTok)]
        tok = self._reg.get_tokenizer("plugin_tok")
        assert isinstance(tok, FakeTok)

    @patch("importlib.metadata.entry_points")
    def test_ensure_called_once(self, mock_ep):
        """_ensure_plugins() only calls discover_plugins once (idempotent)."""
        mock_ep.return_value = []
        self._reg.list_tokenizers()
        self._reg.list_tokenizers()
        mock_ep.assert_called_once()


class TestModelRegistryPlugins:
    """Plugin discovery integration for the model registry."""

    def setup_method(self):
        from scmodelforge.models import registry as reg

        self._reg = reg
        self._orig_state = reg._state["plugins_loaded"]
        reg._state["plugins_loaded"] = False

    def teardown_method(self):
        self._reg._state["plugins_loaded"] = self._orig_state
        self._reg._MODEL_REGISTRY.pop("plugin_model", None)

    @patch("importlib.metadata.entry_points")
    def test_list_triggers_discovery(self, mock_ep):
        class FakeModel:
            pass

        mock_ep.return_value = [_make_ep("plugin_model", FakeModel)]
        names = self._reg.list_models()
        assert "plugin_model" in names
        mock_ep.assert_called_once_with(group=MODEL_GROUP)

    @patch("importlib.metadata.entry_points")
    def test_ensure_called_once(self, mock_ep):
        mock_ep.return_value = []
        self._reg.list_models()
        self._reg.list_models()
        mock_ep.assert_called_once()


class TestBenchmarkRegistryPlugins:
    """Plugin discovery integration for the benchmark registry."""

    def setup_method(self):
        from scmodelforge.eval import registry as reg

        self._reg = reg
        self._orig_state = reg._state["plugins_loaded"]
        reg._state["plugins_loaded"] = False

    def teardown_method(self):
        self._reg._state["plugins_loaded"] = self._orig_state
        self._reg._REGISTRY.pop("plugin_bench", None)

    @patch("importlib.metadata.entry_points")
    def test_list_triggers_discovery(self, mock_ep):
        class FakeBench:
            pass

        mock_ep.return_value = [_make_ep("plugin_bench", FakeBench)]
        names = self._reg.list_benchmarks()
        assert "plugin_bench" in names
        mock_ep.assert_called_once_with(group=BENCHMARK_GROUP)

    @patch("importlib.metadata.entry_points")
    def test_get_triggers_discovery(self, mock_ep):
        class FakeBench:
            pass

        mock_ep.return_value = [_make_ep("plugin_bench", FakeBench)]
        bench = self._reg.get_benchmark("plugin_bench")
        assert isinstance(bench, FakeBench)

    @patch("importlib.metadata.entry_points")
    def test_ensure_called_once(self, mock_ep):
        mock_ep.return_value = []
        self._reg.list_benchmarks()
        self._reg.list_benchmarks()
        mock_ep.assert_called_once()
