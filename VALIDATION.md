# Validation

## uv and project-local storage (2026-09-27)

Final local check on 2026-09-28: `python -B run.py test` ran 9 tests in
45.347 seconds: 8 PASS, 1 SKIP (Windows symlink permission). The added removal
test creates a real uv venv, Matplotlib plot/font cache, PyTorch checkpoint and
temporary file in a generated project, deletes that project, and verifies the
external sentinel cache is unchanged. No user data is deleted by the test.
The actual cache-path audit and Bash syntax/diff checks also pass.

- Windows, existing CPython 3.11.9, uv 0.12.6; clean `.venv-uv-cpu` created from `uv.lock`.
- `python -B run.py test`: 8 tests, 7 PASS, 1 SKIP (Windows symlink creation permission).
- Existing exact checkpoint replay assertions are retained. The CPU replay test now
  uses one thread: the initial two-thread run differed by 9.31e-10 in a parameter;
  the single-thread run passed exact equality. This does not prove bitwise replay
  across platforms or production multithreaded runs. Production threads remain 4.
- `python -B run.py audit`: actual uv cache, tempfile, Matplotlib config/cache,
  and PyTorch hub paths all resolve inside the repository. Report: `.runtime/audit.json`.
- Inherited external uv/venv/temp settings are overridden in the child environment;
  parent environment is unchanged. Path traversal rejection is tested.
- `bash -n server.sh` and `git diff --check` pass.
- `dino_experiment.py` is unchanged, including EMA formulas, model and training settings.
- CUDA/A5000 execution, Docker image build/runtime, and Linux CI have not been run
  for this update. The configured CUDA cache path has not been verified by a GPU run.
- This is an application-storage audit, not a full OS write trace. Docker images,
  historical global caches, OS logs, and pre-existing tools are not automatically removed.

## Previous validation

Local validation: Windows, Python 3.13.14, PyTorch 2.7.1+cpu,
torchvision 0.22.1+cpu, numpy 2.2.6, Pillow 11.2.1.

- Five unittest cases PASS (18.925 seconds).
- Original analytic B/C/G/R tests, warmup, loss stop-gradient, multicrop and 100-epoch schedules PASS.
- C/G clipping and required-algorithm/diagnostic timing classification PASS.
- Fixed bicubic matrix output and gradient match reference within FP32 tolerance.
- G synthetic continuous epoch 1-6 matches epoch 1-5 + resumed epoch 6 exactly:
  student, teacher, center, gap EMA, losses, ratios and evaluation.
- Changed resource configuration is rejected on resume.
- Bash syntax and pip dependency checks PASS.
- Linux/Python 3.11 CPU tests also run in GitHub Actions; see the actual run status.

The A5000 server, Docker GPU image runtime, real CIFAR-10 training, memory usage and
accuracy have NOT been tested locally. Run the provided server check and benchmark.
No empirical learning improvement or wall-time estimate is claimed.
