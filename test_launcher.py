"""Storage isolation regressions, without downloading or training."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import run


class LauncherTests(unittest.TestCase):
    def test_python_bootstrap_stays_local_without_global_registration(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT) as tmp:
            root = Path(tmp).resolve()
            env = run.local_environment(root, "cpu")
            executable = root / ".runtime/python/cpython-3.11/bin/python3"
            with patch.object(run.subprocess, "run") as install, patch.object(
                    run.subprocess, "check_output", return_value=str(executable)):
                self.assertEqual(run.project_python("uv", env), str(executable))
                command = install.call_args.args[0]
                self.assertIn("--no-bin", command)
                self.assertIn("--no-registry", command)
                self.assertEqual(install.call_args.kwargs["env"]["UV_PYTHON_INSTALL_DIR"],
                                 str(root / ".runtime/python"))
                self.assertEqual(install.call_args.kwargs["env"]["UV_PYTHON_DOWNLOADS"], "automatic")
                self.assertEqual(env["UV_PYTHON_DOWNLOADS"], "never")
            with patch.object(run.subprocess, "run"), patch.object(
                    run.subprocess, "check_output", return_value=str(root.parent / "outside-python")):
                with self.assertRaises(RuntimeError):
                    run.project_python("uv", env)

    def test_project_removal_cleans_runtime_outputs(self):
        # The whole fixture is inside this repository; never delete user cache paths.
        with tempfile.TemporaryDirectory(dir=run.ROOT) as tmp:
            fixture = Path(tmp).resolve()
            project = fixture / "project"
            project.mkdir()
            outside = fixture / "outside-cache"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("pre-existing shared cache", encoding="utf-8")
            before = sentinel.read_bytes()
            inherited = dict(os.environ)
            for key in ("UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "MPLCONFIGDIR",
                        "TORCH_HOME", "TMPDIR", "TEMP", "TMP"):
                inherited[key] = str(outside)
            env = run.local_environment(project, "cpu", inherited)
            probe = """
import json, tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
Path('runs').mkdir()
torch.save({'value': torch.ones(1)}, 'runs/last.pt')
plt.plot([0, 1], [0, 1])
plt.savefig('runs/comparison.png')
plt.close('all')
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(b'project temporary file')
    temporary = f.name
print(json.dumps([temporary, matplotlib.get_cachedir(), torch.hub.get_dir()]))
"""
            paths = json.loads(subprocess.check_output(
                [sys.executable, "-B", "-c", probe], cwd=project, env=env, text=True))
            for path in paths:
                self.assertTrue(Path(path).resolve().is_relative_to(project), path)
            uv_local = run.ROOT / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
            uv = str(uv_local) if uv_local.is_file() else shutil.which("uv")
            self.assertIsNotNone(uv, "Run this test through run.py with uv installed")
            subprocess.run([uv, "venv", env["UV_PROJECT_ENVIRONMENT"], "--python",
                            sys.executable, "--no-managed-python"], cwd=project,
                           env=env, check=True, capture_output=True)
            self.assertTrue((project / "runs/last.pt").is_file())
            self.assertTrue((project / ".venv-uv-cpu/pyvenv.cfg").is_file())
            # Explicitly verify the recursive-delete target is our generated fixture.
            self.assertEqual(project.resolve().parent, fixture)
            self.assertTrue(project.resolve().is_relative_to(run.ROOT))
            shutil.rmtree(project)
            self.assertFalse(project.exists())
            self.assertEqual(list(fixture.iterdir()), [outside])
            self.assertEqual(list(outside.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), before)

    def test_hostile_environment_and_real_tempfile(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT) as tmp:
            root = Path(tmp).resolve()
            inherited = dict(os.environ, UV_PROJECT_ENVIRONMENT="/outside/venv",
                             UV_CACHE_DIR="/outside/cache", TEMP="/outside/tmp",
                             UV_CONFIG_FILE="/outside/uv.toml", VIRTUAL_ENV="/outside/env")
            env = run.local_environment(root, "cpu", inherited)
            self.assertNotIn("UV_CONFIG_FILE", env)
            self.assertNotIn("VIRTUAL_ENV", env)
            self.assertEqual(inherited["TEMP"], "/outside/tmp")
            self.assertEqual(env["UV_PYTHON_DOWNLOADS"], "never")
            for key in ("UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "MPLCONFIGDIR",
                        "TORCH_HOME", "CUDA_CACHE_PATH", "TMPDIR", "TEMP"):
                self.assertTrue(Path(env[key]).is_relative_to(root), key)
            actual = subprocess.check_output([sys.executable, "-B", "-c",
                "import tempfile; print(tempfile.gettempdir())"], env=env, text=True).strip()
            self.assertEqual(Path(actual).resolve(), root / ".runtime/tmp")

    def test_escaping_paths_rejected(self):
        with self.assertRaises(ValueError):
            run.inside(run.ROOT, "../outside")

    def test_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT) as tmp:
            root = Path(tmp)
            try:
                (root / "data").symlink_to(root.parent, target_is_directory=True)
            except OSError:
                self.skipTest("Creating symlinks requires OS permission")
            with self.assertRaises(ValueError):
                run.local_environment(root, "cpu")


if __name__ == "__main__":
    unittest.main()
