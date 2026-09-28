"""uv launcher: project-owned environments, caches and temporary files stay here.

Requires an existing Python 3.11-3.13 and uv >=0.12.6. Does not install either
globally, download Python, edit shell profiles, or change the parent environment.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = ("check", "benchmark", "pilot", "main", "final", "final-test")


def inside(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Refusing path outside project: {path}")
    return path


def local_environment(root, backend, inherited=None):
    root = root.resolve()
    env = dict(os.environ if inherited is None else inherited)
    # Ignore inherited uv installation targets/settings and active environments.
    for key in list(env):
        if key.startswith("UV_") or key in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME"):
            env.pop(key)
    locations = {
        "UV_PROJECT_ENVIRONMENT": f".venv-uv-{backend}",
        "UV_CACHE_DIR": ".runtime/uv-cache",
        "UV_PYTHON_INSTALL_DIR": ".runtime/python",
        "UV_PYTHON_BIN_DIR": ".runtime/bin",
        "UV_TOOL_DIR": ".runtime/tools",
        "UV_TOOL_BIN_DIR": ".runtime/bin",
        "UV_CREDENTIALS_DIR": ".runtime/uv-credentials",
        "XDG_CACHE_HOME": ".runtime/cache",
        "XDG_CONFIG_HOME": ".runtime/config",
        "XDG_DATA_HOME": ".runtime/share",
        "XDG_STATE_HOME": ".runtime/state",
        "TORCH_HOME": ".runtime/torch",
        "TORCH_EXTENSIONS_DIR": ".runtime/torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": ".runtime/torchinductor",
        "TRITON_CACHE_DIR": ".runtime/triton",
        "CUDA_CACHE_PATH": ".runtime/cuda",
        "MPLCONFIGDIR": ".runtime/matplotlib",
        "PIP_CACHE_DIR": ".runtime/pip-cache",
        "TMPDIR": ".runtime/tmp", "TMP": ".runtime/tmp", "TEMP": ".runtime/tmp",
    }
    for key, relative in locations.items():
        path = inside(root, relative)
        # uv creates the venv itself (an existing empty directory is unnecessary).
        if key != "UV_PROJECT_ENVIRONMENT":
            path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for relative in ("data", "runs", "benchmark_runs", "smoke_runs", "test_runs"):
        inside(root, relative)  # reject existing symlinks/junctions escaping the repo
    env.update(UV_NO_CONFIG="1", UV_PYTHON_DOWNLOADS="never",
               UV_PYTHON_INSTALL_REGISTRY="0", UV_PYTHON_INSTALL_BIN="0",
               UV_NO_MODIFY_PATH="1", UV_LINK_MODE="copy",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if "GPU_DEVICE" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["GPU_DEVICE"]
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cu126"))
    parser.add_argument("--installed", action="store_true", help="Use container's preinstalled environment; no uv sync")
    parser.add_argument("command", choices=("lock", "sync", "test", "audit", "self-test", "smoke", "summary", *STAGES))
    args = parser.parse_args()
    backend = args.backend or ("cu126" if args.command in STAGES else "cpu")
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error("Use an existing Python 3.11, 3.12 or 3.13. No global Python will be installed.")
    if args.command in STAGES and (sys.platform != "linux" or backend != "cu126"):
        parser.error("GPU server stages require Linux and --backend cu126.")
    env = local_environment(ROOT, backend)
    if args.command == "audit":
        # Does not download dependencies. Tests actual stdlib/uv paths in child processes.
        report = {k: v for k, v in env.items() if k in (
            "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "TEMP")}
        report["project"] = str(ROOT)
        report["tempfile_actual"] = subprocess.check_output(
            [sys.executable, "-B", "-c", "import tempfile; print(tempfile.gettempdir())"],
            cwd=ROOT, env=env, text=True).strip()
        if not Path(report["tempfile_actual"]).resolve().is_relative_to(ROOT):
            raise RuntimeError("Temporary directory escaped project")
        local_uv = ROOT / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
        uv = str(local_uv) if local_uv.is_file() else shutil.which("uv")
        if uv:
            actual = subprocess.check_output([uv, "cache", "dir"], cwd=ROOT, env=env, text=True).strip()
            if not Path(actual).resolve().is_relative_to(ROOT):
                raise RuntimeError("uv cache escaped project")
            report["uv_cache_actual"] = actual
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if args.installed:
            python = Path(sys.executable)
        if python.is_file():
            probe = (
                "import json, tempfile, matplotlib, torch; "
                "print(json.dumps(dict(tempfile=tempfile.gettempdir(), "
                "matplotlib_config=matplotlib.get_configdir(), "
                "matplotlib_cache=matplotlib.get_cachedir(), torch_hub=torch.hub.get_dir())))"
            )
            actual = json.loads(subprocess.check_output([str(python), "-B", "-c", probe], cwd=ROOT, env=env, text=True))
            for key, path in actual.items():
                if not Path(path).resolve().is_relative_to(ROOT):
                    raise RuntimeError(f"{key} escaped project: {path}")
            report["library_paths_actual"] = actual
        else:
            report["library_paths_actual"] = "Not installed; run sync/test then audit again."
        report["scope"] = "Configured application storage, not an OS-level filesystem sandbox or system-log audit."
        (ROOT / ".runtime" / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if args.installed:
        if args.command in ("lock", "sync"):
            parser.error("--installed cannot lock/sync")
        prefix = [sys.executable, "-B"]
    else:
        local_uv = ROOT / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
        uv = str(local_uv) if local_uv.is_file() else shutil.which("uv")
        if not uv:
            parser.error("uv not found. Put the standalone uv executable in .tools/ or use an existing uv on PATH. See README.")
        common = ["--project", str(ROOT), "--python", sys.executable, "--no-managed-python"]
        if args.command == "lock":
            subprocess.run([uv, "lock", *common], cwd=ROOT, env=env, check=True)
            return
        if args.command == "sync":
            subprocess.run([uv, "sync", "--locked", "--extra", backend, *common], cwd=ROOT, env=env, check=True)
            return
        prefix = [uv, "run", "--locked", "--extra", backend, *common, "python", "-B"]
    if args.command == "test":
        command = ["-m", "unittest", "-v", "test_launcher", "test_experiment"]
    elif args.command in ("self-test", "smoke", "summary"):
        command = ["dino_experiment.py", args.command]
        if args.command == "smoke":
            command += ["--method", "G", "--out", "smoke_runs"]
            if (ROOT / "smoke_runs/SMOKE_G_seed0/last.pt").exists():
                command += ["--resume"]
    else:
        command = ["server.py", args.command]
        if not args.installed:
            print("Native uv mode: no Docker RAM/CPU hard quotas. GPU allocator cap and thread/worker settings still apply.", flush=True)
    subprocess.run([*prefix, *command], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
