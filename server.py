"""Sequential server workflow. Run through server.sh for hard RAM/CPU limits."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMON = ["--device", "cuda", "--batch", "64", "--eval-batch", "64",
          "--threads", "4", "--workers", "2", "--gpu-memory-gib", "10.5",
          "--data", "./data"]

def call(*args):
    cmd = [sys.executable, "-u", "dino_experiment.py", *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def train(method, seed, until, out="runs"):
    checkpoint = ROOT / out / f"{method}_seed{seed}" / "last.pt"
    call("train", "--method", method, "--seed", str(seed), "--until", str(until),
         "--out", out, *COMMON, *(["--resume"] if checkpoint.exists() else []))

def check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check NVIDIA driver and Container Toolkit.")
    # Do a real deterministic FP32 CUDA forward/backward under the allocator cap.
    import dino_experiment as d
    args = argparse.Namespace(threads=4, workers=2, eval_batch=64,
                              gpu_memory_gib=10.5, device="cuda")
    d.configure_runtime(args)
    net = d.DINO(smoke=True).cuda()
    loss = sum(x.square().mean() for x in net(
        [torch.randn(2, 3, n, n, device="cuda") for n in (96, 96, 48, 48)]))
    loss.backward()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    report = d.environment()
    report.update(gpu_total_gib=props.total_memory/2**30,
                  allocator_cap_gib=10.5, gpu_budget_gib=12,
                  note_gpu_budget="CUDA overhead is outside the allocator cap; not a strict device-memory quota.",
                  expected_cpu_quota=4, expected_ram_gib=32,
                  cuda_forward_backward="PASS")
    for name in ("cpu.max", "memory.max", "memory.swap.max"):
        path = Path("/sys/fs/cgroup")/name
        report[name] = path.read_text().strip() if path.exists() else "cgroup v2 file unavailable"
    print(json.dumps(report, indent=2))
    subprocess.run(["nvidia-smi"], check=True)
    print("Docker mode: check docker stats for CPU quota=4, RAM=32 GiB. Native uv mode has no such hard quotas. GPU compute utilization is not capped.")

def benchmark():
    train("B", 0, 1, "benchmark_runs")
    path = ROOT/"benchmark_runs/B_seed0/metrics.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    estimate = {
        "observed_first_epoch_train_seconds": float(row["train_epoch_s"]),
        "train_only_100_epoch_hours": float(row["train_epoch_s"])*100/3600,
        "train_only_nine_trajectories_hours": float(row["train_epoch_s"])*900/3600,
        "peak_tensor_vram_mib": float(row["train_peak_vram_mb"]),
        "warning": "Rough baseline estimate: excludes evaluation/checkpoint/setup; first epoch includes startup.",
    }
    (ROOT/"benchmark_runs/estimate.json").write_text(json.dumps(estimate, indent=2))
    print(json.dumps(estimate, indent=2))

def pilot():
    call("self-test")
    benchmark()
    # Freeze the prespecified target BEFORE evaluating C or G.
    # Rerunning a completed pilot only reuses the existing target.
    target = ROOT/"runs/target.json"
    train("B", 0, 20)
    if not target.exists():
        call("freeze-target", "--out", "runs")
    for method in ("C", "G"):
        train(method, 0, 20)
    call("summary", "--out", "runs")
    print("Pilot complete. Review kNN/variance/entropy/gaps/clipping before: bash server.sh main")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "self-test", "smoke", "benchmark",
                                         "pilot", "main", "final", "final-test", "summary"])
    args = parser.parse_args()
    os.chdir(ROOT)
    # Lock also protects direct invocations and checkpoint/data writes.
    import fcntl
    with open(ROOT/".run.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another experiment process is already running in this folder.")
        if args.stage in ("check", "benchmark", "pilot", "main", "final", "final-test"):
            subprocess.run([sys.executable, "-c", "import server; server.check()"], check=True)
        if args.stage == "check":
            return
        if args.stage == "self-test":
            call("self-test")
        elif args.stage == "smoke":
            call("smoke", "--method", "G", "--out", "smoke_runs",
                 *(["--resume"] if (ROOT/"smoke_runs/SMOKE_G_seed0/last.pt").exists() else []))
        elif args.stage == "benchmark":
            benchmark()
        elif args.stage == "pilot":
            pilot()
        elif args.stage == "main":
            if not (ROOT/"runs/target.json").exists():
                raise SystemExit("Run pilot first and inspect its results.")
            # Validate all three pilot trajectories before starting expensive work.
            import dino_experiment as d
            for method in "BCG":
                checkpoint = ROOT/"runs"/f"{method}_seed0"/"last.pt"
                if not checkpoint.exists() or d.load_checkpoint(checkpoint)["epoch"] < 20:
                    raise SystemExit("Incomplete pilot. Run pilot first.")
            for seed in range(3):
                for method in "BCG":
                    train(method, seed, 100)
            call("summary", "--out", "runs")
        elif args.stage in ("final", "final-test"):
            import dino_experiment as d
            for seed in range(3):
                for method in "BCG":
                    path = ROOT/"runs"/f"{method}_seed{seed}"/"last.pt"
                    if not path.exists() or d.load_checkpoint(path)["epoch"] != 100:
                        raise SystemExit("Complete all nine 100-epoch runs first.")
            for seed in range(3):
                for method in "BCG":
                    call("final", "--method", method, "--seed", str(seed),
                         "--out", "runs", *COMMON,
                         *(["--test"] if args.stage == "final-test" else []))
            call("summary", "--out", "runs")
        elif args.stage == "summary":
            call("summary", "--out", "runs")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
