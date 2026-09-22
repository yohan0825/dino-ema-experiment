#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
image=dino-ema:1.1
if [[ "${1:-}" == build ]]; then
  docker build -t "$image" .
  exit
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "First run: bash server.sh build" >&2
  exit 1
fi
# One fixed container name prevents concurrent jobs on this host.
# CPU quota covers the training process AND its data-loader workers.
tty_flags=()
if [[ -t 0 && -t 1 ]]; then tty_flags=(-it); fi
exec docker run --rm --init "${tty_flags[@]}" \
  --name dino-ema-conservative \
  --gpus "device=${GPU_DEVICE:-0}" \
  --cpus=4 --memory=32g --memory-swap=32g --shm-size=1g --pids-limit=256 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/workspace" \
  "$image" "${@:-check}"
