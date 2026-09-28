# DINO EMA 조절 실험 설정

확인일: 2026-09-27

실행 방식 업데이트: uv 실행기 `run.py`와 잠금 파일 `uv.lock`을 추가했다.
`python run.py --backend cu126 pilot`로 Linux GPU 서버에서 실행하며,
가상환경·캐시·임시파일은 저장소 내부에 보관한다. 이 직접 실행 방식에는
아래 Docker RAM/CPU 강제 제한이 적용되지 않는다. Docker를 쓰면 이미지·빌드 캐시가
저장소 밖에 남으므로 삭제 범위는 최신 README의 uv/삭제 안내를 참고한다.
아래 커밋은 원래 실험 설정을 확인한 시점이며 uv 변경 전이다.

## 저장소와 확인 범위

- GitHub: https://github.com/yohan0825/dino-ema-experiment (공개 전환)
- 로컬 경로: `C:\Users\LG\dino_experiments`
- 확인한 커밋: `3b97154335332a82af73df55dcce01352a3d45e1`
- 확인 당시 로컬 코드와 GitHub main의 커밋이 일치했다.
- 이 문서는 코드에 정의된 실험 설정을 정리한 것으로, 실험 완료나 성능 개선을 의미하지 않는다.

## 실험 목적과 비교 조건

DINO의 teacher를 갱신하는 EMA를 전체에 공통으로 적용하는 방식과, student–teacher 파라미터 차이에 따라 조절하는 방식을 비교한다.

| 조건 | 방식 |
|---|---|
| B: 기준선 | 모든 파라미터에 공통 EMA 적용 |
| C: 공통 적응형 | 그룹별 배수를 파라미터 수로 가중평균하여 전체에 동일하게 적용 |
| G: 그룹별 적응형 | front / back / head에 각각 다른 배수 적용 |

R 방식도 코드에 구현되어 있으나 기본 서버 실행 및 B/C/G 시간 비교에서는 제외된다.

### EMA 설정

- 기본 teacher momentum: cosine schedule로 `0.996 → 1.0`.
- 기본 갱신량은 `a = 1 - momentum`이며, `teacher += a × 배수 × (student - teacher)`로 갱신한다.
- 처음 5 epoch는 모든 조건에서 공통 EMA를 사용한다.
- C/G의 gap 통계 누적과 적응형 조절은 6 epoch부터 시작한다.
- 그룹별 상대 gap: `RMS(student - teacher) / (RMS(student) + 1e-8)`.
- Gap smoothing 계수: `0.99`.
- 그룹별 평활 gap을 세 그룹의 평균으로 나누어 배수를 계산하고 `[0.5, 2.0]`으로 제한한다.
- Teacher는 student의 optimizer step 이후 갱신한다.

| 그룹 | 포함 파라미터 |
|---|---|
| front | Patch embedding, CLS token, position embedding, Transformer blocks 0~5 |
| back | Transformer blocks 6~11, 마지막 LayerNorm |
| head | Projection head 전체 |

## 모델과 모델 크기

공식 DINO 소스나 사전학습 가중치를 내려받지 않는 독립 구현이며, ViT-Tiny 스타일의 모델을 처음부터 학습한다.

| 항목 | 설정 |
|---|---|
| Transformer 깊이 | 12층 |
| 임베딩 차원 | 192 |
| Attention head 수 | 3 |
| Patch 크기 | 8×8 |
| Transformer MLP 비율 | 4배 |
| Projection head | `192 → 2048 → 2048 → 256 → 4096` |
| 백본 파라미터 | 5,403,840개, 약 540만 |
| Projection head 파라미터 | 6,164,736개, 약 616만 |
| 모델 1개 전체 파라미터 | 11,568,576개, 약 1,157만 |
| Student / teacher | 같은 구조를 각각 보유. Teacher는 student 초기값을 복사하고 gradient를 계산하지 않음 |

파라미터 수는 코드의 레이어 구조로 계산한 값이며, optimizer 상태와 활성화 메모리는 포함하지 않는다.

## 데이터와 입력

| 항목 | 설정 |
|---|---|
| 데이터셋 | CIFAR-10 |
| 자기지도 학습 데이터 | 10,000장, 클래스당 1,000장 |
| Validation | 5,000장, 클래스당 500장 |
| 분할 방식 | CIFAR-10 train split에서 클래스 균형을 맞춰 서로 겹치지 않게 추출 |
| 데이터 분할 시드 | 2026 고정 |
| 학습 라벨 사용 | 자기지도 학습 loss에는 라벨 미사용 |
| Global crop | 96×96, 이미지당 2개 |
| Local crop | 48×48, 이미지당 2개 |
| Student 입력 | Global 2개 + local 2개 |
| Teacher 입력 | Global 2개 |
| Test set | 별도 `final-test` 단계에서 명시적으로 평가 |

CIFAR-10 원본 32×32 이미지를 확대하여 사용하므로 공식 ImageNet DINO 성능 재현 실험으로 해석하지 않는다.

## 학습 하이퍼파라미터

| 항목 | 설정 |
|---|---|
| 학습 길이 | 각 조건·시드당 100 epoch |
| Train / eval batch | 64 / 64 |
| Optimizer | AdamW |
| 최대 Learning rate | `0.0005 × batch / 256` = batch 64에서 `0.000125` |
| LR schedule | 10 epoch warmup 후 cosine decay, 마지막 `1e-6` |
| Weight decay | Cosine schedule로 `0.04 → 0.4` |
| Teacher / student temperature | 0.04 / 0.1 |
| Center momentum | 0.9 |
| Gradient clipping | Norm 3.0 |
| 첫 epoch | Projection head 마지막 weight gradient 제거 |
| 정밀도 | FP32 |
| AMP / TF32 | 사용하지 않음 |
| Deterministic 설정 | 켬 |

## 시드와 병렬 실행

**시드 병렬 실행이 아니라 GPU 1개에서 순차 실행하는 설정이다.**

- 학습 시드: `0, 1, 2`.
- 비교 조건: `B, C, G`.
- 총 실험 수: `3개 시드 × 3개 조건 = 9개`.
- 본 실험 실행 순서: `seed0 B → C → G`, `seed1 B → C → G`, `seed2 B → C → G`.
- 같은 폴더의 파일 잠금과 고정된 Docker 컨테이너 이름으로 중복 실행을 방지한다.
- DataLoader worker 2개는 데이터 로딩용이며 시드 병렬 실행을 뜻하지 않는다.
- 데이터 분할 시드 2026과 학습 시드 0/1/2는 서로 다른 설정이다.

## 서버와 자원 설정

| 항목 | 설정 |
|---|---|
| 대상 서버 | NVIDIA RTX A5000 24GB, 시스템 RAM 128GB |
| 실행 환경 | Linux + Docker + NVIDIA Container Toolkit |
| 노출 GPU | 1개, 기본 GPU 0 |
| GPU 메모리 예산 | 12GiB |
| PyTorch allocator 제한 | 10.5GiB |
| CUDA 여유 예산 | 1.5GiB |
| 컨테이너 RAM 제한 | 32GiB |
| 추가 Swap | 없음 |
| CPU quota | Logical CPU 4개 상당 |
| PyTorch thread | 4개 |
| DataLoader worker | 2개 |

GPU 12GiB는 장치 전체에 대한 엄격한 메모리 상한이 아니다. CUDA context와 라이브러리 메모리는 allocator 제한 밖에 있으며 실제 전체 사용량은 서버에서 확인해야 한다. GPU 연산 사용률은 제한하지 않는다.

## 실행 단계와 평가

1. `bash server.sh build`: Docker 이미지 빌드.
2. `bash server.sh pilot`: CUDA 검사, 자체 테스트, B seed0 1 epoch 벤치마크, B/C/G seed0 각각 20 epoch 실행.
3. 파일럿에서 B seed0의 epoch 20 validation kNN 값을 목표값으로 고정하고 결과를 검토한다.
4. `bash server.sh main`: B/C/G × seed 0/1/2를 각각 100 epoch까지 실행. Seed0은 파일럿 이후 이어서 학습한다.
5. `bash server.sh final`: 9개 학습 완료 후 frozen backbone의 linear probe를 100 epoch 학습하고 validation 평가.
6. `bash server.sh final-test`: 실험 선택을 끝낸 뒤 최종 test 평가.
7. `bash server.sh summary`: 비교 CSV와 그래프 생성.

파일럿 20 epoch는 100 epoch schedule의 처음 20 epoch이다. 벤치마크 가중치를 파일럿에 가져오지 않는다.

- 학습 중 평가: 5 epoch마다 teacher CLS feature 기반 cosine kNN, `k=20`, 비가중 다수결.
- Linear probe: 저장된 L2-normalized CLS feature에 선형 분류기를 학습하며 백본은 고정한다.
- 목표 도달 시점: 고정 목표 정확도를 엄격히 초과한 첫 평가 시점. 관측 간격은 5 epoch이다.
- 최종 비교: 시드별 결과와 평균·표준편차.
- 매 epoch checkpoint를 저장하고 중단 시 마지막 완료 epoch부터 재개한다.

## 결과 파일과 검증 상태

| 파일 | 내용 |
|---|---|
| `runs/{B,C,G}_seed{0,1,2}/last.pt` | 모델, optimizer, center, updater, RNG, 이력 |
| 각 run의 `metrics.csv` | Epoch별 학습 및 평가 지표 |
| `runs/target.json` | 파일럿 B seed0 epoch20에서 고정한 목표값 |
| `comparison_per_seed.csv` | 시드별 비교 |
| `comparison_mean_sd.csv` | 평균·표준편차 |
| `linear_val.json`, `final_test.json` | 최종 선형 평가 |
| `benchmark_runs/estimate.json` | 첫 epoch 기준의 거친 시간 추정 |

저장소의 VALIDATION.md에는 CPU 합성 데이터 기반 수식·재개 동일성 등의 검증이 기록되어 있다. 실제 A5000 서버에서의 CIFAR-10 학습, GPU 메모리 사용량, 정확도는 해당 검증 기록에서 확인되지 않는다. 이 문서는 실제 서버 실험 결과나 소요 시간을 주장하지 않는다.

## 출처

- [README: 실험 설계와 실행 절차](https://github.com/yohan0825/dino-ema-experiment/blob/3b97154335332a82af73df55dcce01352a3d45e1/README.md)
- [dino_experiment.py: 모델·EMA·학습·평가 구현](https://github.com/yohan0825/dino-ema-experiment/blob/3b97154335332a82af73df55dcce01352a3d45e1/dino_experiment.py)
- [server.py: 시드와 실행 순서](https://github.com/yohan0825/dino-ema-experiment/blob/3b97154335332a82af73df55dcce01352a3d45e1/server.py)
- [server.sh: Docker 자원 제한](https://github.com/yohan0825/dino-ema-experiment/blob/3b97154335332a82af73df55dcce01352a3d45e1/server.sh)
- [VALIDATION.md: 검증 범위](https://github.com/yohan0825/dino-ema-experiment/blob/3b97154335332a82af73df55dcce01352a3d45e1/VALIDATION.md)
