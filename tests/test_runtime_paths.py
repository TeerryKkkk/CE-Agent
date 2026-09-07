from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "climate_pipeline" / "config.py"


def load_config(path: Path = CONFIG):
    spec = importlib.util.spec_from_file_location("runtime_path_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clear_path_overrides(monkeypatch):
    for name in ("CE_AGENT_PROJECT_ROOT", "CE_AGENT_DATA_ROOT", "CE_AGENT_RUN_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_checkout_defaults_keep_runtime_files_in_the_project():
    config = load_config()
    assert config.PROJECT_ROOT == ROOT
    assert config.PROJECT_HOME == ROOT
    assert config.DATA_ROOT == ROOT / "data"
    assert config.RUN_ROOT == ROOT / "runs"
    assert config.RAW_CE_PILOT_CONFIG.is_file()


def test_explicit_runtime_directories_are_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("CE_AGENT_PROJECT_ROOT", str(tmp_path / "project"))
    monkeypatch.setenv("CE_AGENT_DATA_ROOT", str(tmp_path / "inputs"))
    monkeypatch.setenv("CE_AGENT_RUN_ROOT", str(tmp_path / "generated"))
    config = load_config()
    assert config.PROJECT_ROOT == tmp_path / "project"
    assert config.DATA_ROOT == tmp_path / "inputs"
    assert config.RUN_ROOT == tmp_path / "generated"


def test_installed_layout_defaults_to_the_working_directory(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages" / "climate_pipeline" / "config.py"
    installed.parent.mkdir(parents=True)
    shutil.copyfile(CONFIG, installed)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    config = load_config(installed)
    assert config.PROJECT_ROOT == workspace
    assert config.RUN_ROOT == workspace / "runs"
