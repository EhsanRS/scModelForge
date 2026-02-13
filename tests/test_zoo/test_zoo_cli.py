"""Tests for zoo CLI commands: zoo install/list/remove and --isolated flag."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scmodelforge.cli import main
from scmodelforge.zoo._env_registry import create_env_info, save_env_info


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestZooGroup:
    """zoo command group basics."""

    def test_zoo_help(self, runner) -> None:
        result = runner.invoke(main, ["zoo", "--help"])
        assert result.exit_code == 0
        assert "install" in result.output
        assert "list" in result.output
        assert "remove" in result.output


class TestZooInstall:
    """zoo install subcommand."""

    def test_help(self, runner) -> None:
        result = runner.invoke(main, ["zoo", "install", "--help"])
        assert result.exit_code == 0
        assert "--env-dir" in result.output
        assert "--python" in result.output
        assert "--extra-deps" in result.output

    def test_install_calls_install_env(self, runner, tmp_path) -> None:
        with patch("scmodelforge.zoo.isolation.install_env") as mock_install:
            result = runner.invoke(main, ["zoo", "install", "geneformer", "--env-dir", str(tmp_path)])
            assert result.exit_code == 0
            assert "installed successfully" in result.output
            mock_install.assert_called_once_with(
                "geneformer",
                env_dir=str(tmp_path),
                python_version=None,
                extra_deps=None,
            )

    def test_install_with_extra_deps(self, runner, tmp_path) -> None:
        with patch("scmodelforge.zoo.isolation.install_env") as mock_install:
            result = runner.invoke(
                main,
                ["zoo", "install", "geneformer", "--env-dir", str(tmp_path), "--extra-deps", "flash-attn>=2.0"],
            )
            assert result.exit_code == 0
            mock_install.assert_called_once()
            call_kwargs = mock_install.call_args[1]
            assert call_kwargs["extra_deps"] == ["flash-attn>=2.0"]

    def test_install_with_python_version(self, runner, tmp_path) -> None:
        with patch("scmodelforge.zoo.isolation.install_env") as mock_install:
            result = runner.invoke(
                main,
                ["zoo", "install", "geneformer", "--env-dir", str(tmp_path), "--python", "3.10"],
            )
            assert result.exit_code == 0
            call_kwargs = mock_install.call_args[1]
            assert call_kwargs["python_version"] == "3.10"


class TestZooList:
    """zoo list subcommand."""

    def test_empty(self, runner, tmp_path) -> None:
        result = runner.invoke(main, ["zoo", "list", "--env-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No isolated environments" in result.output

    def test_lists_installed(self, runner, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, status="installed")
        save_env_info(info)
        result = runner.invoke(main, ["zoo", "list", "--env-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "geneformer" in result.output
        assert "installed" in result.output


class TestZooRemove:
    """zoo remove subcommand."""

    def test_remove_with_yes_flag(self, runner, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, status="installed")
        save_env_info(info)
        result = runner.invoke(main, ["zoo", "remove", "geneformer", "--env-dir", str(tmp_path), "--yes"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert not (tmp_path / "geneformer").exists()

    def test_remove_nonexistent(self, runner, tmp_path) -> None:
        result = runner.invoke(main, ["zoo", "remove", "missing", "--env-dir", str(tmp_path), "--yes"])
        assert result.exit_code == 0
        assert "No environment found" in result.output


class TestBenchmarkIsolatedFlag:
    """--isolated flag on the benchmark command."""

    def test_benchmark_help_shows_isolated(self, runner) -> None:
        result = runner.invoke(main, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--isolated" in result.output
        assert "--env-dir" in result.output
