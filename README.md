# DINO group-wise adaptive teacher experiment

RTX A5000 24GB / RAM 128GB 서버에서 일부 자원만 사용하는 독립적인 소규모 DINO 실험입니다.
공식 DINO 성능 재현이나 성능 개선을 주장하지 않습니다.

## uv로 실행: 폴더 안에 설치·캐시·결과 보관

서버에 기존 Python **3.11 이상(3.14도 가능)**과 **uv 0.12.6 이상**이 필요합니다.
실행기가 uv로 **학습용 Python 3.11을 `.runtime/python/` 안에만 자동 다운로드**합니다.
전역 패키지 설치, 시스템 Python 교체, 레지스트리 등록, PATH/셸 설정 변경은 하지 않습니다.
의존성은 `pyproject.toml`과 커밋된 `uv.lock`으로 고정합니다.

**Linux GPU 서버에서 최소 실행 명령 (기본값: 1 epoch 벤치마크)**:

```bash
git clone https://github.com/yohan0825/dino-ema-experiment.git
cd dino-ema-experiment
python3 -B run.py
```

공개 저장소이므로 다운로드에 GitHub 로그인이 필요하지 않습니다.
이미 받은 폴더라면 그 안에서 `git pull` 후 `python3 -B run.py`를 실행하세요.
`-B`는 실행기 자체가 Python 바이트코드 캐시를 쓰지 않도록 합니다.
관리자가 지정한 GPU가 0번 이외라면 마지막 명령을 `GPU_DEVICE=1 python3 -B run.py`처럼 바꾸세요.
스케줄러가 `CUDA_VISIBLE_DEVICES`를 설정한 작업 환경에서는 이를 그대로 따르며,
그때는 `GPU_DEVICE`를 따로 지정하지 마세요. GPU 배정은 서버 관리자 지침을 따릅니다.
Python·패키지·데이터 다운로드 → CUDA 검사 → B seed0 1 epoch → 예상 시간 출력을 자동 수행합니다.
결과는 `benchmark_runs/estimate.json`에도 저장됩니다. 전체 실험은 자동 시작하지 않습니다.

```bash
git clone https://github.com/yohan0825/dino-ema-experiment.git
cd dino-ema-experiment
python run.py audit
python run.py test

# Linux NVIDIA 서버: CUDA 12.6용 PyTorch, GPU 1개에서 순차 실행
python run.py --backend cu126 check
python run.py --backend cu126 pilot
# 파일럿 결과를 검토한 뒤
python run.py --backend cu126 main
python run.py --backend cu126 final
# 실험 선택이 끝난 후에만
python run.py --backend cu126 final-test
```

Linux에서 명령 이름이 `python3`이면 위의 `python`을 `python3`으로 바꾸세요.
Windows에서는 `python run.py test`, `self-test`, `smoke`, `audit`를 사용할 수 있습니다.
`python run.py smoke`는 합성 데이터 검증이며 실제 CIFAR 성능 실험이 아닙니다.
가상환경 activation은 필요 없습니다. 설치만 하려면 `python run.py --backend cpu sync`
또는 `python run.py --backend cu126 sync`를 사용하세요.

**이 uv 직접 실행 경로에는 Docker의 RAM 32GiB·CPU quota 강제 제한이 없습니다.**
PyTorch thread 4, loader worker 2, GPU allocator 10.5GiB 설정은 서버 학습에 유지됩니다.
공유 서버에서 RAM/CPU 강제 제한이 필요하면 아래 Docker 경로를 사용하거나 관리자의
작업 스케줄러 안에서 실행하세요. 실험 중 실행 방식과 자원 조건을 섞지 마세요.

### uv가 없다면

이미 설치된 uv를 사용하거나 [공식 uv 릴리스](https://github.com/astral-sh/uv/releases/tag/0.12.6)에서
운영체제에 맞는 standalone 압축파일을 **이 저장소 안에 다운로드하고 압축 해제**하세요.
실행파일이 Linux에서는 `.tools/uv`, Windows에서는 `.tools/uv.exe`가 되도록 두면
`run.py`가 우선 사용합니다. Linux 파일에는 실행 권한이 필요합니다.
전역 설치 스크립트나 `uv tool install`을 실행할 필요는 없습니다.
시스템 Python·NVIDIA 드라이버는 변경하지 않습니다. 학습용 Python 3.11만 프로젝트 내부에 받습니다.

### 삭제할 때 남는 것 점검

지원하는 `python run.py ...` 경로에서 프로젝트가 관리하는 파일은 다음 위치에 저장됩니다.

| 항목 | 저장소 내부 위치 |
|---|---|
| CPU / CUDA 가상환경 | `.venv-uv-cpu/`, `.venv-uv-cu126/` |
| 학습용 Python 3.11 | `.runtime/python/` |
| uv 다운로드·패키지 캐시 | `.runtime/uv-cache/` |
| 임시파일 | `.runtime/tmp/` |
| Matplotlib 설정·폰트 캐시 | `.runtime/matplotlib/` |
| PyTorch·CUDA·확장/컴파일 캐시 | `.runtime/` 아래 각 전용 폴더 |
| 데이터 | `data/` |
| 결과·체크포인트 | `runs/`, `benchmark_runs/`, `smoke_runs/` |
| 선택적인 standalone uv | `.tools/` |

작업을 종료한 뒤 저장소 폴더를 삭제하면 위 파일들이 함께 삭제됩니다.
탐색기에서 삭제할 때 휴지통으로 이동했다면 디스크 공간은 휴지통을 비워야 반환됩니다.
실행기의 환경변수는 자식 프로세스에만 전달되며 터미널·사용자 환경을 영구 변경하지 않습니다.
기존 `.venv`는 자동으로 삭제하지 않으며, 역시 저장소 안에 있습니다.
폴더 이름을 바꾸거나 다른 서버로 옮긴 가상환경은 재사용하지 말고 해당 가상환경을 재생성하세요.

`python run.py audit`로 설정 경로와 실제 임시파일·uv 캐시 위치를 확인할 수 있습니다.
선택한 backend의 환경이 설치되어 있으면 Matplotlib·PyTorch의 실제 캐시 경로도 검사합니다.
보고서는 `.runtime/audit.json`에 저장됩니다. CUDA 환경 검사에는 `--backend cu126`을 붙이세요.
일반 `uv run`/`pip`/직접 Python 실행으로 실행기를 우회하거나, 저장 경로를 수정하면
이 경로 관리가 적용되지 않습니다. 외부를 가리키는 최상위 출력·캐시 경로의
symlink/junction은 실행기가 거부합니다. 출력 폴더 내부에도 외부 링크를 만들지 마세요.

**이전 실행에서 생긴 공용 캐시나 Docker 이미지가 자동으로 지워지는 것은 아닙니다.**
공용 캐시는 다른 프로젝트와 공유될 수 있어 자동 삭제하지 않습니다. OS/드라이버 로그,
셸 명령 이력, 기존 Python·uv·Git·드라이버는 프로젝트 파일 관리 범위 밖입니다.
따라서 OS 수준의 모든 흔적이 사라진다는 보장은 하지 않습니다.

### Docker 경로의 외부 잔여물

Docker 빌드/실행은 이미지와 빌드 캐시를 Docker 저장소에 남기므로 **폴더 삭제만으로 정리되지 않습니다.**
Docker가 필요 없고 폴더 단위 정리가 우선이면 위 uv 직접 실행을 사용하세요.
Docker 경로도 이제 uv와 같은 `uv.lock`으로 패키지를 설치합니다.
컨테이너는 기존처럼 `--rm`이며 데이터는 저장소 bind mount에 보관합니다.

이 프로젝트의 Docker 사용을 끝낸 뒤 컨테이너가 실행 중이지 않은 상태에서
`docker image rm dino-ema:1.1`로 해당 이미지 태그를 제거할 수 있습니다.
공유 base image와 빌드 캐시는 남을 수 있습니다. **공유 서버에서 `docker system prune`이나
전역 cache 삭제 명령을 실행하지 마세요.** 필요하면 관리자와 소유 자원을 확인하여 정리하세요.

## Docker 자원 제한

아래 강제 자원 제한을 적용하는 실행 경로는 **Linux + Docker + NVIDIA Container Toolkit**입니다.

| 항목 | 기본값 | 적용 범위 |
|---|---:|---|
| RAM | 최대 32 GiB | Docker cgroup, 자식 프로세스·공유 메모리 포함 |
| Swap | 추가 사용 없음 | memory와 memory-swap 모두 32g |
| CPU | 4 logical CPU에 해당하는 quota | 컨테이너 전체, worker 포함 |
| PyTorch 스레드 / loader workers | 4 / 2 | 모든 방법 동일 |
| GPU | 선택한 1개만 노출 | 기본 GPU 0 |
| GPU 메모리 예산 | 12 GiB | allocator 10.5 GiB + CUDA 여유 1.5 GiB |
| Train / eval batch | 64 / 64 | 자동 변경하지 않음 |
| 정밀도 | FP32 | AMP·TF32 끔, deterministic on |

**GPU 12 GiB는 장치 전체 메모리에 대한 hard quota가 아닙니다.**
PyTorch allocator에 10.5 GiB 한도를 적용하지만 CUDA context/라이브러리 메모리는 별도입니다.
따라서 전체 VRAM이 정확히 12 GiB 이하라는 보장은 없습니다. 서버 벤치마크에서
`nvidia-smi`로 실제 사용량을 확인하세요. 엄격한 GPU 메모리 격리가 필수라면 서버 관리자의
GPU 가상화/스케줄러 지원이 필요합니다. GPU 연산 사용률(%)은 제한하지 않습니다.

Docker의 32g는 32 GiB입니다. CPU quota는 특정 물리 코어 네 개를 독점하는 설정이 아닙니다.
같은 이름의 컨테이너와 파일 잠금으로 중복 실행을 막고, B/C/G와 seed는 **한 번에 하나씩** 실행합니다.
다른 폴더/이름으로 직접 여러 컨테이너를 실행하면 자원 제한은 컨테이너마다 적용됩니다.

## Docker로 서버에서 시작

필요 조건: Linux x86_64, NVIDIA 드라이버(CUDA 12.6 wheel 실행을 지원하는 버전),
Docker 실행 권한, NVIDIA Container Toolkit, Git, 최초 설치/데이터 다운로드용 인터넷.
서버 드라이버나 Docker daemon 설정은 이 프로젝트가 자동 변경하지 않습니다.

공식 설치 안내:
- [Docker 설치](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [PyTorch 버전 조합](https://pytorch.org/get-started/previous-versions/)

공개 저장소이므로 clone에 GitHub 인증은 필요하지 않습니다.

```bash
git clone https://github.com/yohan0825/dino-ema-experiment.git
cd dino-ema-experiment
nvidia-smi
bash server.sh build
bash server.sh pilot
```

`pilot`은 다음을 순차 실행합니다.

1. CUDA FP32 forward/backward 및 환경 확인
2. B/C/G/R 수식·stop-gradient·multi-crop·schedule 자체 테스트
3. B seed0 **1 epoch 벤치마크** → `benchmark_runs/`
4. B seed0 **20 epoch** → `runs/`
5. B/seed0의 epoch 20 validation kNN을 `runs/target.json`에 고정
6. C seed0, G seed0를 각각 **20 epoch**
7. 비교 CSV 및 그래프 생성

20 epoch은 **100 epoch 스케줄의 처음 20 epoch**입니다.
벤치마크 학습 가중치를 파일럿으로 가져오지 않습니다.
이미 완료된 epoch는 건너뛰며 중단된 실행은 마지막 epoch 체크포인트부터 재개합니다.
epoch 중간 진행분은 재실행합니다. CUDA OOM 시 batch를 자동으로 바꾸지 않고 중단합니다.

각 단계를 따로 실행할 수도 있습니다.

```bash
bash server.sh check
bash server.sh self-test
bash server.sh benchmark
bash server.sh pilot
```

SSH 연결 종료에도 유지하려면 서버의 `tmux` 세션 안에서 실행하세요.
`Ctrl+C`로 중단 후 같은 명령으로 재개할 수 있습니다.

## 파일럿 이후

`runs/comparison.png`, 각 방법의 `*_statistics.png`, `metrics.csv`를 확인합니다.
B의 kNN과 feature variance/entropy가 정상인지, G의 front/back/head gap이 다른지,
ratio가 0.5/2.0에 계속 붙는지 확인한 후 본 실험을 시작하세요.
고정된 합격 정확도는 정의하지 않았으며, 코드가 연구 결과를 자동 판정하지 않습니다.

```bash
# B/C/G x seeds 0,1,2. seed0 파일럿은 epoch 21부터 이어서 실행.
bash server.sh main

# 9개 모두 100 epoch 완료 후, frozen backbone + linear classifier 100 epochs.
# validation만 평가. test set은 열지 않음.
bash server.sh final

# 모든 실험 선택을 끝낸 후에만 최종 test를 명시적으로 평가.
bash server.sh final-test

bash server.sh summary
```

다른 GPU를 선택하려면 모든 실행에서 `GPU_DEVICE=1 bash server.sh pilot`처럼 지정하세요.
성능 비교 동안 batch/workers/threads/precision/메모리 예산을 바꾸지 마세요.
설정을 바꿔야 하면 기존 결과를 별도 보관하고 모든 B/C/G를 새 조건으로 다시 시작하세요.
checkpoint의 config와 코드 SHA256이 다르면 재개를 거부합니다.

## 결과 파일과 시간 해석

- `benchmark_runs/estimate.json`: 첫 epoch 기준의 거친 예상 시간. 평가·저장 비용 제외.
- `runs/{B,C,G}_seed{0,1,2}/last.pt`: 모델·optimizer·center·updater·RNG·기록.
- `metrics.csv`: 매 epoch 학습 지표, 5 epoch마다 validation 지표.
- `comparison_per_seed.csv`, `comparison_mean_sd.csv`: seed별/평균·표준편차.
- `linear_val.json`, `final_test.json`: 최종 선형평가.
- `target.json`: B seed0 epoch20에서 고정한 목표값.

시간 열은 아래처럼 구분됩니다. 모두 step당 밀리초입니다.

| 열 | 의미 |
|---|---|
| teacher_gap_ms | student–teacher gap 계산 |
| teacher_coefficient_ms | 공통/적응 계수 준비 |
| teacher_ema_ms | 실제 파라미터 EMA 적용 |
| teacher_algorithm_ms | EMA + 계수 + 필요한 gap 계산 |
| teacher_diagnostic_ms | 연구 지표 준비 및 알고리즘에 불필요한 gap 계산 |
| teacher_update_ms | 측정·동기화를 포함한 전체 updater 시간 |

B와 warmup에서 gap은 diagnostic 비용입니다. epoch6 이후 C/G에서 gap은 **필수 알고리즘 비용**이므로
algorithm 시간에 포함합니다. G overhead를 EMA 시간만으로 해석하면 안 됩니다.
동기화 자체의 비용이 있고 CPU/Python 계측 비용도 포함되므로 작은 차이는 신중히 해석하세요.
`train_s`에는 진단 비용이 포함됩니다. 계측을 제거한 처리량 벤치마크가 아닙니다.

`clip_low/high_*`는 각 그룹의 **clamp 전 ratio가 경계에 닿은 step 비율**입니다.
C는 그룹 ratio를 parameter count로 가중 평균한 하나의 계수를 적용하므로
`raw_ratio_*`와 실제 `ratio_*`가 다릅니다.
R은 기존 코드의 탐색용 방법으로 남겨두었지만 서버 워크플로와 새 시간 분리 비교에서는 제외됩니다.
목표 도달은 목표 정확도를 **엄격히 초과(>)**한 첫 평가 시점입니다. 관측 간격은 5 epoch입니다.

별도 터미널에서 다음으로 실제 자원을 확인할 수 있습니다.

```bash
docker stats dino-ema-conservative
watch -n 2 nvidia-smi
```

`train_peak_vram_mb`는 PyTorch tensor allocation 최고값이며 CUDA context나
캐시 예약분을 포함한 nvidia-smi 전체 VRAM과 다릅니다.

## 실험 설계

- CIFAR-10: class-balanced SSL train 10k / validation 5k, split seed 2026.
- Test는 `final-test`에서만 평가. 최초 CIFAR archive 다운로드에는 test 파일도 포함됨.
- ViT-Tiny style: depth12 / dim192 / patch8, head MLP3 / output4096.
- global96 2개 + local48 2개. Teacher는 global만 사용.
- front: embedding·CLS·position·blocks0–5; back: blocks6–11·final norm; head: projection 전체.
- B: 공통 EMA; C: 그룹 배율의 parameter-count 가중 평균을 전체에 적용; G: 그룹별 적용.
- 처음 5 epoch는 공통 EMA. 원본 코드와 동일하게 gap smoothing 누적도 epoch6부터 시작.
- 상대 gap = RMS(S−T)/(RMS(S)+1e−8), smoothing=.99, ratio clamp=[.5,2].
- AdamW, batch-scaled LR, warmup10, 총100 epoch cosine, WD .04→.4, momentum .996→1.
- teacher/student temperature .04/.1, center momentum .9, gradient clip3.
- 첫 epoch에는 head 마지막 weight gradient를 제거.
- teacher는 optimizer step 이후 갱신. kNN은 teacher CLS, cosine k20의 비가중 다수결.
- Linear probe는 저장된 L2-normalized CLS feature에 학습한 선형층, backbone frozen.
- CIFAR 32px를 96px로 확대하므로 공식 ImageNet DINO 재현으로 해석하면 안 됨.
- 공유 서버의 다른 작업은 GPU 속도에 영향을 줄 수 있습니다. 실제 환경/버전은 각 run에 저장됩니다.

## 로컬 CPU 검증

```bash
python run.py test
```

이 테스트는 합성 데이터로 수식·clipping·시간 비용 분류·그룹 분할·adaptive 경계에서의
중단/재개 동일성을 확인합니다. 실제 데이터 성능이나 A5000 VRAM 적합성을 증명하지 않습니다.
Docker 경로를 우회한 직접 Python 실행은 RAM/CPU cgroup 제한을 제공하지 않습니다.

참고: [PyTorch 메모리 제한](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html),
[Docker 자원 제한](https://docs.docker.com/engine/containers/resource_constraints/),
[공식 DINO](https://github.com/facebookresearch/dino).

## CUDA deterministic compatibility

The local positional embedding resize uses a fixed bicubic linear operator (12x12 to 6x6).
Its value and gradient are checked against CPU bicubic interpolation. This preserves the resize
while avoiding the unsupported deterministic CUDA bicubic backward. All methods share this path.
