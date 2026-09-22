# Validation

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
