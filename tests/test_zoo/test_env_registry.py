"""Tests for zoo._env_registry — on-disk environment metadata management."""

from __future__ import annotations

from scmodelforge.zoo._env_registry import (
    EnvInfo,
    create_env_info,
    get_env_dir,
    is_env_installed,
    list_installed_envs,
    load_env_info,
    remove_env,
    save_env_info,
)


class TestEnvInfo:
    """EnvInfo dataclass basics."""

    def test_defaults(self) -> None:
        info = EnvInfo(model_name="m", env_path="/e", venv_path="/v", python_path="/p")
        assert info.model_name == "m"
        assert info.deps == []
        assert info.status == "installing"

    def test_custom_fields(self) -> None:
        info = EnvInfo(
            model_name="geneformer",
            env_path="/envs/geneformer",
            venv_path="/envs/geneformer/venv",
            python_path="/envs/geneformer/venv/bin/python",
            deps=["geneformer>=0.1", "torch>=2.0"],
            python_version="3.10",
            created_at="2025-01-01T00:00:00+00:00",
            status="installed",
        )
        assert info.deps == ["geneformer>=0.1", "torch>=2.0"]
        assert info.status == "installed"


class TestGetEnvDir:
    """get_env_dir path construction."""

    def test_uses_model_name_subdirectory(self, tmp_path) -> None:
        result = get_env_dir("geneformer", base_dir=tmp_path)
        assert result == tmp_path / "geneformer"

    def test_default_base_dir(self) -> None:
        result = get_env_dir("test_model")
        assert "scmodelforge/envs/test_model" in str(result)


class TestSaveLoadEnvInfo:
    """save_env_info + load_env_info round-trip."""

    def test_round_trip(self, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, deps=["torch>=2.0"], status="installed")
        save_env_info(info)

        loaded = load_env_info("geneformer", base_dir=tmp_path)
        assert loaded is not None
        assert loaded.model_name == "geneformer"
        assert loaded.deps == ["torch>=2.0"]
        assert loaded.status == "installed"

    def test_load_missing_returns_none(self, tmp_path) -> None:
        assert load_env_info("nonexistent", base_dir=tmp_path) is None

    def test_load_corrupt_json_returns_none(self, tmp_path) -> None:
        env_dir = tmp_path / "broken"
        env_dir.mkdir()
        (env_dir / "env.json").write_text("not valid json{{{")
        assert load_env_info("broken", base_dir=tmp_path) is None


class TestListInstalledEnvs:
    """list_installed_envs scanning."""

    def test_empty_dir(self, tmp_path) -> None:
        assert list_installed_envs(base_dir=tmp_path) == []

    def test_nonexistent_dir(self, tmp_path) -> None:
        assert list_installed_envs(base_dir=tmp_path / "missing") == []

    def test_lists_multiple(self, tmp_path) -> None:
        for name in ["alpha", "beta"]:
            info = create_env_info(name, base_dir=tmp_path, status="installed")
            save_env_info(info)
        result = list_installed_envs(base_dir=tmp_path)
        assert len(result) == 2
        assert result[0].model_name == "alpha"
        assert result[1].model_name == "beta"

    def test_skips_corrupt(self, tmp_path) -> None:
        # One good, one corrupt
        good = create_env_info("good", base_dir=tmp_path, status="installed")
        save_env_info(good)
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "env.json").write_text("{invalid")
        result = list_installed_envs(base_dir=tmp_path)
        assert len(result) == 1
        assert result[0].model_name == "good"


class TestRemoveEnv:
    """remove_env cleanup."""

    def test_removes_existing(self, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, status="installed")
        save_env_info(info)
        assert (tmp_path / "geneformer").exists()
        assert remove_env("geneformer", base_dir=tmp_path) is True
        assert not (tmp_path / "geneformer").exists()

    def test_returns_false_for_missing(self, tmp_path) -> None:
        assert remove_env("nonexistent", base_dir=tmp_path) is False


class TestIsEnvInstalled:
    """is_env_installed status check."""

    def test_installed(self, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, status="installed")
        save_env_info(info)
        assert is_env_installed("geneformer", base_dir=tmp_path) is True

    def test_installing_not_ready(self, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, status="installing")
        save_env_info(info)
        assert is_env_installed("geneformer", base_dir=tmp_path) is False

    def test_missing(self, tmp_path) -> None:
        assert is_env_installed("nonexistent", base_dir=tmp_path) is False


class TestCreateEnvInfo:
    """create_env_info factory."""

    def test_standard_paths(self, tmp_path) -> None:
        info = create_env_info("geneformer", base_dir=tmp_path, deps=["torch"], python_version="3.10")
        assert info.model_name == "geneformer"
        assert "geneformer/venv" in info.venv_path
        assert "geneformer/venv/bin/python" in info.python_path
        assert info.deps == ["torch"]
        assert info.python_version == "3.10"
        assert info.created_at  # not empty
        assert info.status == "installing"
