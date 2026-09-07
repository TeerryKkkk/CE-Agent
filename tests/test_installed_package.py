from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_installed_package_imports_outside_the_checkout(tmp_path):
    script = textwrap.dedent(
        """
        import importlib
        from importlib import metadata, resources
        import json
        from pathlib import Path
        import pkgutil
        import sys

        import climate_pipeline

        root = Path(sys.argv[1]).resolve()
        search_paths = {Path(path).resolve() for path in sys.path}
        assert root not in search_paths, sys.path

        distribution = metadata.distribution("ce-agent")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        editable = direct_url.get("dir_info", {}).get("editable", False)
        helpers = (
            "io_utils", "hashing", "rainfall_p99_pipeline",
            "schema_models", "us_tavily", "web_reader",
        )
        if not editable:
            assert root / "src" not in search_paths, sys.path
            assert not Path(climate_pipeline.__file__).resolve().is_relative_to(root / "src")
            files = {path.as_posix() for path in distribution.files or []}
            assert all(f"climate_pipeline/{name}.py" in files for name in helpers)
            assert not any(path.startswith("src/") for path in files)

        modules = [
            importlib.import_module(item.name)
            for item in pkgutil.walk_packages(
                climate_pipeline.__path__, climate_pipeline.__name__ + "."
            )
        ]
        for name in helpers:
            assert importlib.import_module(f"climate_pipeline.{name}") in modules
            assert name not in sys.modules, name
        assert not any(name == "src" or name.startswith("src.") for name in sys.modules)
        for module in modules:
            for name in getattr(module, "__all__", []):
                assert hasattr(module, name), (module.__name__, name)
        for name in ("california_mainline.yaml", "texas_legacy_pilot.yaml"):
            assert resources.files("climate_pipeline").joinpath("configs", name).is_file()
        from climate_pipeline import config, io_utils
        assert io_utils.repo_relative_path(config.PROJECT_ROOT / "runs" / "probe.json") == "runs/probe.json"
        print(f"Imported {len(modules)} modules from {climate_pipeline.__file__}")
        """
    )
    environment = os.environ.copy()
    # Deliberately supply the checkout paths: isolated Python must ignore them.
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
