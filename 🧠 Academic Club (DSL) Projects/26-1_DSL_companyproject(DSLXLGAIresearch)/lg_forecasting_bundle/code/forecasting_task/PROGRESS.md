# xForecast Challenge — 진행 상황 및 다음 방향

100종목 4주 후 종가 방향 예측 (Weighted Hit Rate). DoubleAdapt(KDD'23) 메타러닝 프레임워크 위에 fund-manager 스타일 5-step 전처리 파이프라인을 얹어 실험 중.

## 1. 대회 규칙 핵심 (잊으면 안 되는 것)

- 공식 지표: **Weighted Hit Rate** = `sign(pred_close - anchor_close_4wk_ago) == sign(true_close - anchor_close_4wk_ago)`, 가중치 `w_i,t`는 **"뉴스 신호가 중요한 케이스일수록 높게"** 부여됨. **정확한 가중치 공식은 비공개.**
- Train: 2019-01-01 ~ 2022-12-31. **Private(실제 채점) 구간: 2023-01-01 ~ 2023-12-31.**
- Public leaderboard(2022)는 토이 벤치마크일 뿐 — training 데이터로 타겟 역산 가능해서 퍼펙트 스코어 나올 수 있음, 신뢰 금지. **내부적으로는 test_start=2023-01-01, test_end=2023-12-01로 이미 private 구간과 맞춰서 평가 중.**
- 제출 파일: `submission.csv`, `ID={TICKER}_{date}`, `Close=예측종가`. 2022+2023 전체 기간 필요 (2022는 public 채점용 형식만 맞추면 됨).

## 2. 데이터 / 전처리 구조

- 원본: `data/test.csv` (가격) + `data/{model}_textemb.parquet` × 6개 (linq/bert/gemini/lgai/nvda/qwen 임베딩)
- **Alpha360 스타일**: 샘플 하나 = `[과거 seq_len일] × [factor_num개 피처]` → 1D로 flatten해서 DoubleAdapt에 투입
- 4개 텍스트 레벨: `macro`(거시), `sector`(섹터), `target`(종목 자신), `related`(관련/경쟁사) — 각 레벨의 text_id를 해당 임베딩모델로 lookup → mean pool → **train 구간에 PCA fit** → 6차원으로 압축 (`{level}_pc0~5`)
- 패널 빌드는 `(tickers, dates, PanelConfig, 임베딩파일 mtime)` 해시로 `data/.panel_cache/*.pkl`에 캐싱됨 — 같은 설정이면 재사용, seq_len/flags 등 바뀌면 새로 빌드(수 분~40분 소요, novelty 켜면 더 오래 걸림)
- 핵심 파일: `forecasting_task/preprocessing/{panel,price_factors,text_pooling,macro_factors,sector_factors,target_factors,levels}.py`

### Step2~4 "파생" 피처 (플래그로 on/off, `--use_macro` / `--use_sector` / `--use_target_novelty` / `--use_related_novelty`)
- `macro_streak`, `macro_regime_shift`, `macro_direction`, `macro_price_align`
- `sector_peer_ret`, `sector_rel_strength`, `sector_cluster_size`, `sector_pooled_pc0~5`
- `target_novelty_consensus`, `target_already_priced`, `target_drift_flag` (+related 동일)
  - **k-of-N 컨센서스**: 6개 임베딩 모델이 각자 "오늘 텍스트가 최근 롤링평균과 얼마나 다른가"를 투표, `novelty_consensus_frac=2/3`(6개 중 4개) 이상 동의해야 최종 신호로 인정. 값은 이진이 아니라 "6개 중 몇 개가 동의했나" 비율로 저장.

## 3. Backbone / 프레임워크

- 5개 backbone 교체 가능: GRU, PatchTST, DLinear, TimesNet, iTransformer (`forecasting_task/backbones.py`, `TSLibBackbone`이 TSLib Model들을 `forward(x)->[batch]` 인터페이스로 감쌈)
- DoubleAdapt(`DoubleAdapt/src/{model,net,utils}.py`)는 **원본 KDD'23 코드 그대로 사용** — Data Adapter + Model Adapter를 `higher` 라이브러리로 미분 가능한 inner-loop 삼아 MAML 스타일 메타학습. Train 구간을 rolling task(27개)로 쪼개 offline 메타학습 → test 구간(11개 task)에서 순차 온라인 적응+예측.
- 학습 loss: 기본 `MSELoss()`. 옵션으로 `JointMSEDirLoss`(`--aux_weight>0`) — MSE + 방향 BCE 항 결합, `label_mean/label_std`로 표준화 라벨을 원래 스케일로 shift한 뒤 부호 비교.

## 4. 지금까지 실험 결과 전체 (핵심)

| 구성 | 전체 hit rate | 뉴스-중요 서브셋(n≈3644) |
|---|---|---|
| 무조건 상승 예측 (test 라벨 자체가 55.2%/44.8%로 편향) | 0.5521 | 0.5450 |
| Naive (DoubleAdapt 미적용, 그냥 incremental) | 0.4999 | – |
| DoubleAdapt+GRU, Step1(가격)만 | 0.5254 | – |
| DoubleAdapt+GRU, 전체 Step1-4 피처 (IC로 체크포인트 선택) | 0.5101 | – |
| GRU (Step1+텍스트PCA만, 파생 피처 X) | 0.5265 | – |
| PatchTST | 0.4983 | – |
| **DLinear (lr=0.001 기본)** | 0.5495 | 0.5590 |
| TimesNet | 0.5249 | – |
| iTransformer | 0.4765 (발산, MSE 8e10) | – |
| 5-backbone 평균가 앙상블 | 0.4880 (발산 backbone에 오염) | – |
| 5-backbone 다수결 앙상블 | 0.5344 | – |
| 3-backbone(GRU+DLinear+TimesNet) 다수결 | 0.5580 | – |
| GRU + JointMSEDirLoss(aux=0.3) | 0.5336 (기본 대비 +0.007) | – |
| DLinear + JointMSEDirLoss(aux=0.3) | 0.5471 (기본 대비 -0.002, 오히려 하락) | – |
| DLinear, seq_len=40 | 0.4640 (붕괴) | – |
| DLinear, seq_len=90 | 0.5300 | – |
| DLinear, reg=1.0 | 0.5525 | – |
| **DLinear, lr=0.002 (현재 최선, `logs/dlinear_hp_lr_high`)** | **0.5557** | **0.5645 (+1.95%p, 가장 유의미)** |
| DLinear, lr=0.002 + target/related novelty 피처 추가 | 0.5522 (novelty 넣으니 오히려 하락, 서브셋 우위도 소멸) | 0.5453 |

**현재 최종 추천**: `DoubleAdapt + DLinear, lr=0.002, seq_len=60, reg=0.5, Step1가격+텍스트PCA(linq)만, 파생피처/joint loss 전부 미사용`. 재현 커맨드:
```
python3 -m forecasting_task.run_DoubleAdapt --backbone dlinear --lr 0.002 --logdir logs/dlinear_hp_lr_high
```

## 5. 중요 교훈 / 함정

1. **Test 라벨이 상승 55.2%로 편향돼 있어서, 대부분의 실험이 "무조건 상승" baseline(0.5521)을 통계적으로 유의미하게 못 이김.** n=22000 기준 95% 유의 마진은 ±0.66%p. 개선 주장할 때 이 기준으로 검증 필수.
2. **파생 novelty/macro/sector 피처를 입력에 그냥 얹는 건 지금까지 전부 실패** (attribution에서 permute해도 hit rate가 안 떨어짐 = 기여 없음; 명시적으로 추가해도 오히려 하락).
3. **k-of-N 컨센서스 신호는 "채점 필터"로는 유의미했음** — novelty_consensus/drift_flag가 1인 서브셋(16.6%)에서 DLinear가 always-up 대비 유의미하게 이김. 근데 이 신호를 학습 loss weight로 준 적은 아직 없음(다음 방향 참고).
4. **앙상블은 신중히** — 발산한 backbone(PatchTST, iTransformer) 하나만 섞여도 평균 예측이 완전히 망가짐. 반드시 개별 backbone MSE 먼저 확인 후 정상인 것만 앙상블.
5. **cgroup 메모리 한도 128.8GB** (`free -h`의 503GB 아님, `/sys/fs/cgroup/memory/memory.limit_in_bytes` 확인 필수). 임베딩 6개(각 dim 384~4096, 857,892 rows) 동시 로딩하면 OOM. `EmbeddingStore.release()`로 모델별 순차 로드/해제 필수.
6. **DoubleAdapt 학습이 `higher` 라이브러리로 이중 미분하기 때문에, backbone 내부에 in-place tensor 연산(`x /= y` 등)이 있으면 깨짐** — 반드시 `x = x / y` 형태로.

## 6. 다음 방향 (우선순위 순, 미실행)

### ① Loss weighting (가장 유력, 아직 미구현)
`JointMSEDirLoss`에 샘플별 가중치 추가 — `target_novelty_consensus`/`related_drift_flag`가 높은 샘플의 loss를 2~3배로 키워서 학습. 지금까지는 novelty를 "입력 피처"로만 시도(실패)했고, "loss weight"로는 시도 안 해봄. 대회 채점이 뉴스-중요 케이스 가중이므로 방향성 부합.
- 구현 위치: `DoubleAdapt/src/net.py`의 `JointMSEDirLoss.forward()`에 `sample_weight` 인자 추가, `DoubleAdaptManager`/`IncrementalManager`가 배치별로 novelty 컬럼값을 같이 넘겨주도록 수정 필요.

### ② DLinear lr/reg 조합 탐색 확장
lr=0.002 단독이 최선이었고 reg=1.0도 소폭 개선 — **lr=0.002 + reg=1.0 조합**은 아직 안 해봄. 그리드도 lr 2개, reg 2개 점만 찍어봤으니 더 세밀하게(lr=0.0015, 0.0025 / reg=0.7, 1.5 등) 탐색 여지 있음.

### ③ 다른 임베딩 모델(linq 외) 시도
지금까지 base 텍스트 PCA는 linq 하나만 씀. bert(작은 차원이라 노이즈 걱정했었음)/gemini/qwen 등으로 `--primary_emb_model` 바꿔서 비교 안 해봄.

### ④ 발산 backbone 원인 조사
PatchTST/iTransformer가 MSE 수억~수백억까지 튀는 문제 — 학습률/gradient clipping/정규화 미스매치일 가능성. 고치면 앙상블 후보가 늘어남.

### ⑤ Attribution 신뢰성 재검증
GRU(IC-selected) 체크포인트에서는 price조차 permute하면 좋아지는 이상 현상이 있었음(방향성 편향된 체크포인트로 추정) — attribution 결과를 checkpoint 선택에 따라 달리 해석해야 함. DLinear 체크포인트에서는 그 정도가 훨씬 작았음(±5%p → ±1%p).

## 6.5 업데이트 (야간 자동 실행 중)

**① Loss weighting을 실제로 구현했습니다** (이전까진 미구현이었음):
- `--news_weight_mult N`: `target_novelty_consensus`/`related_drift_flag`가 1인 샘플의 loss를 N배로 가중. 기본 1.0(끄기)이면 완전히 기존과 동일 동작.
- 구현 위치: `DoubleAdapt/src/net.py`(`JointMSEDirLoss.forward(pred,label,weight=None)`), `DoubleAdapt/src/utils.py`(`get_task_data`/`preprocess`가 `w_train`/`w_test`/`w_extra` 텐서를 threading), `DoubleAdapt/src/model.py`(양쪽 `_run_task`에서 criterion 호출 시 weight 전달), `run_DoubleAdapt.py`(`--news_weight_mult` 플래그, panel_df에 `("weight",0)` 컬럼 주입).
- 3티커 스모크 테스트 진행 중 `higher` 모듈이 환경에서 사라진 걸 발견 → `pip install --no-deps higher`로 재설치함 (다른 GPU에서도 이 이슈 겪을 수 있음, 잊지 말고 `--no-deps`로).

**`forecasting_task/overnight_pipeline2.sh`를 백그라운드로 실행해둠** (사용자 잠든 사이 자동 진행):
- Stage A: news_weight_mult 스모크 테스트 (3티커)
- Stage B: news_weight_mult ∈ {2,3,5} 스윕 (DLinear, lr=0.002)
- Stage C: lr=0.002 + reg=1.0 조합 (아직 안 해본 조합)
- Stage D: primary_emb_model ∈ {bert,gemini,qwen} 스윕 (DLinear, lr=0.002) — 지금까지 linq만 써봤던 것 확장
- 결과는 `logs/overnight_pipeline2_summary.txt`에 정리됨. 사용자의 `run_kaggle_weighted` 등 다른 작업과 충돌 안 나게 yield 로직 포함.

## 6.6 업데이트 — 규정 준수 + sector novelty 실험 (진행 중)

**규정 준수 버전이 최종 최고 기록**: `logs/dlinear_final_compliant` — DLinear, lr=0.002, reg=1.0, news_weight_mult=10, `--freeze_online 1`(2022-2023 평가 구간엔 gradient 업데이트 전혀 없음, train.parquet 상당 구간으로만 학습), 데이터 소스는 `data/test.parquet`(train.parquet과 2019-2022 구간 바이트 동일 검증됨). **hit rate = 0.5900**, always-up baseline(0.5521) 대비 확실히 유의미. 제출 파일: `submission_dlinear_final_compliant.csv`.

**진행 중인 실험**: novelty consensus를 target/related뿐 아니라 **sector 레벨에도 적용**하면 어떨지 테스트 중 (`--use_sector_novelty` 새로 구현, `preprocessing/panel.py`/`preprocessing/target_factors.py`는 이미 범용이라 그대로 재사용). macro는 제외했는데, 확인해보니 `macro_category` 텍스트 ID가 **100종목 전부 동일**(그날 거시경제 뉴스는 시장 전체 공통)이라 종목별 novelty 신호로서 의미가 없어서(attribution에서도 delta=0으로 확인됨). sector는 11개 클러스터 단위로만 공유돼서 어느 정도 차별화 여지가 있어 시도해봄. `logs/dlinear_final_plus_sector_novelty`에 결과 저장 예정, `logs/overnight_sector_novelty_summary.txt`에서 baseline(0.5900)과 비교.

**버그 발견/수정**: `EmbeddingStore.release()`가 `_instances.pop()`만 하고 `gc.collect()`를 안 불러서, target+related+sector 3개 레벨(6개 모델×3=18번 임베딩 로드/해제)을 한 프로세스에서 연달아 처리하면 파이썬 힙이 OS에 반환 안 되고 계속 누적되다가 128.8GB cgroup 한도에서 OOM-kill 당함. `embedding_store.py`에 `import gc` + `release()` 끝에 `gc.collect()` 추가해서 해결. **다른 GPU에서도 target+related+sector(혹은 4개 레벨 이상) novelty를 한 프로세스에서 같이 켜면 이 문제 겪을 수 있으니 최신 코드 받아서 써야 함.**

## 7. 재현에 필요한 것

- 코드: `forecasting_task/`, `DoubleAdapt/`
- 데이터: `data/test.csv`, `data/{linq,bert,gemini,lgai,nvda,qwen}_textemb.parquet`
- 패널 캐시(선택, 있으면 40분 절약): `data/.panel_cache/*.pkl` — 다른 머신으로 옮기면 캐시 키(파일 mtime 포함) 안 맞아서 어차피 새로 빌드될 가능성 높음
- 의존성 주의사항: `torch==2.3.1+cu121`, `higher`/`reformer-pytorch` 설치 시 반드시 `--no-deps` (안 그러면 torch 버전 꼬여서 CUDA 깨짐), `pyarrow` 필요
- 로그: `logs/dlinear_hp_lr_high/`(최선 체크포인트), `logs/*.txt`(각종 summary)
