# 체크포인트 sweep 평가 (dp500 + tsr200) 절차

학습 중단/정지 후 체크포인트 N개를 골라 dp bench 500장 + tsr 200장을 평가하는
반복 작업. 로컬(jhyeo, H100 4장) 기준. 애매한 지점(체크포인트 선택, "제일
나은거" 판단)만 확인받고 나머지는 바로 진행.

## 0. GPU 확인

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```
GPU1은 죽은 프로세스가 물고 있는 경우가 잦음 — 비어있는 GPU만 사용(보통 0,2,3).

## 1. 체크포인트 선정

```bash
python3 -c "
import json
d = json.load(open('<CKROOT>/checkpoint-<마지막 STEP>/trainer_state.json'))
print('best_global_step:', d.get('best_global_step'), 'best_metric:', d.get('best_metric'))
for e in d['log_history']:
    if 'eval_loss' in e: print(e['step'], e['eval_loss'])
"
```
`<CKROOT>` = `/home/jhyeo/finetuning/vlm/output/<RUN>/student_sft`.
관례: 초반=200(비교용 고정), 중반=step 중간 지점, 후반=eval_loss 최저 상위 N개.
`eval_loss`만으로 최적을 판단하지 말 것 — TEDS/LLM-Judge와 방향이 다를 수 있음.

## 2. 워커/채점 스크립트 준비

`e22`를 최신 템플릿으로 복사 후 태그만 치환(파이프라인이 그대로면 새로 안 짬):

```bash
mkdir -p /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>
cp /home/jhyeo/finetuning/eval_318/_infer_shards_e22/dp500_withocr_abs.jsonl \
   /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/
cp /home/jhyeo/finetuning/eval_318/_infer_shards_e22/e22_ckpt_worker.sh \
   /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_ckpt_worker.sh
cp /home/jhyeo/finetuning/eval_318/_infer_shards_e22/e22_score_all.sh \
   /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_score_all.sh
sed -i "s/e22/<NEW>/g" /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_ckpt_worker.sh \
                        /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_score_all.sh
chmod +x /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_{ckpt_worker,score_all}.sh
```
sed 뒤 `MAXPX`/`MINPX`가 학습 config의 `max_pixels`/`min_pixels`와 일치하는지
확인(config 헤더 주석에 "이전 대비 변경점" 참고). tsr200 매니페스트
(`_infer_shards/tsr200_html.jsonl`)는 고정이라 그대로 씀.

## 3. 사전 확인

```bash
for f in \
  /home/jhyeo/finetuning/eval_318/tsr_test_eval/genos500_dpbench/scripts/prepare/convert_qwen_pred_to_dpbench.py \
  /home/jhyeo/finetuning/eval_318/tsr_test_eval/genos500_dpbench/scripts/run_eval_genos500.sh \
  /home/jhyeo/finetuning/eval_318/_infer_shards/conv_tsr_for_ppb.py \
  /home/jhyeo/finetuning/eval_318/tsr_test_eval/genos500_dpbench/pdfparsebench/run_ppb_tsr.py \
  /home/jhyeo/finetuning/vlm/eval/evaluate_unified.py \
  <CKROOT>/checkpoint-<STEP1> <CKROOT>/checkpoint-<STEP2> ; do
  [ -e "$f" ] && echo "OK   $f" || echo "MISS $f"
done
```

## 4. 추론 (GPU별 tmux)

체크포인트를 비어있는 GPU에 고르게 분배:

```bash
mkdir -p /home/jhyeo/finetuning/vlm/logs/e<NEW>_ckpt_sweep
tmux new-session -d -s e<NEW>eval_g0 \
  "GPU=0 CKPTS='<ckpt들>' bash /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_ckpt_worker.sh \
   2>&1 | tee /home/jhyeo/finetuning/vlm/logs/e<NEW>_ckpt_sweep/worker_gpu0.log"
# GPU 2, 3도 동일 패턴
```
체크포인트당: LoRA merge → dp500 500장 vLLM 추론 → tsr200 200장 vLLM 추론 →
dp-bench json 변환 → merged 삭제. **긴 백그라운드 작업은 항상 tmux로**(plain
background bash는 세션 끊기면 조용히 죽고 알림이 안 온 전례 있음).

## 5. 채점

```bash
CKPTS="<ckpt들 공백구분>" bash /home/jhyeo/finetuning/eval_318/_infer_shards_<NEW>/<NEW>_score_all.sh
```
dp500→dp-bench(NID/TEDS zero·skip/matched), tsr200→LLM-Judge(GenOS 776, 키는
`.env`의 `genos_key`). `SKIP_JUDGE=1`/`SKIP_DPBENCH=1`로 절반만 재실행 가능.
LLM-Judge는 오래 걸리므로 원하면 맨 마지막에 별도 실행.

## 6. 산출물 경로

```text
eval_results/e<NEW>_sweep_dp500_ckpt<STEP>_WITHOCR/predictions_unified.jsonl
eval_results/e<NEW>_sweep_tsr200_ckpt<STEP>_WITHOCR/{predictions_unified.jsonl,metrics_unified.json}
genos500_dpbench/dp_out/e<NEW>_ckpt<STEP>_WITHOCR/eval_result_gitlab.txt
genos500_dpbench/ppb_out/e<NEW>_tsr200_ckpt<STEP>_WITHOCR/
```
**tsr200은 반드시 rescore 후 사용** — live 추론 채점은 `output_format:"html"`
때문에 표 전용이 아닌 generic HTML TEDS로 잘못 잡힘(`table_samples=0`):
```bash
python -m eval.rescore_unified --predictions eval_results/e<NEW>_sweep_tsr200_ckpt<STEP>_WITHOCR/predictions_unified.jsonl \
  --test_data /home/jhyeo/finetuning/eval_318/_infer_shards/tsr200_html.jsonl \
  --output_dir eval_results/e<NEW>_sweep_tsr200_ckpt<STEP>_WITHOCR_rescored
```

## 7. tsr200 complexity별 TEDS breakdown

```bash
cd /home/jhyeo/finetuning/vlm
python -m eval.complexity_breakdown \
  --tsr200 /home/jhyeo/finetuning/eval_318/_infer_shards/tsr200_html.jsonl --metric teds \
  e<NEW>_ckpt<STEP1>=eval_results/e<NEW>_sweep_tsr200_ckpt<STEP1>_WITHOCR_rescored/predictions_unified.jsonl \
  e<NEW>_ckpt<STEP2>=eval_results/e<NEW>_sweep_tsr200_ckpt<STEP2>_WITHOCR_rescored/predictions_unified.jsonl
```
LLM-Judge 없이 초 단위. 버킷 고정: simple30/medium50/complex40/complex_col30/complex_mix30/complex_row20=200.

## 8. dp500 수식 Edit/CDM (OmniDocBench)

```bash
TEDS_WORKERS=1 MATCH_WORKERS=1 CDM_WORKERS=1 \
bash /home/jhyeo/finetuning/eval_318/tsr_test_eval/genos500_dpbench/omnidocbench/run_omnidocbench_sweep.sh \
  e<NEW>_ckpt<STEP1>=/home/jhyeo/finetuning/vlm/eval_results/e<NEW>_sweep_dp500_ckpt<STEP1>_WITHOCR/predictions_unified.jsonl \
  e<NEW>_ckpt<STEP2>=/home/jhyeo/finetuning/vlm/eval_results/e<NEW>_sweep_dp500_ckpt<STEP2>_WITHOCR/predictions_unified.jsonl
```
`*_WORKERS=1` 필수(기본값 8/8/4는 fork/filelock 에러로 일부 수식이 조용히
채점 실패함). 결과: `$OMNIDOCBENCH_ROOT/result/pred_md_<tag>_quick_match_metric_result.json`
(`display_formula.all.{Edit_dist.ALL_page_avg, CDM.all}`). CDM이 낮으면 렌더링
실패보다 **빈 예측**이나 **quick_match 오정렬**(`*_display_formula_result.json`에서
`pred==''` 또는 `edit>=0.9` 확인) 쪽을 먼저 의심할 것 — 여러 체크포인트에서
같은 `img_id`가 똑같이 틀리면 모델 문제가 아니라 그 문서 GT 정렬 자체가 애매한 것.

## 9. 리더보드 갱신 (매 sweep마다)

새 체크포인트를 3개 표에 추가 → dp500은 TEDS zero(b), tsr200은 TEDS 내림차순
재정렬 → **같은 세대(`e<N>_ckpt*`)가 이미 여러 줄이면 제일 나은 것 1줄만 남기고
나머지 삭제**(막 추가한 이번 세대는 예외, 다음 세대가 들어올 때 정리). 외부
baseline/이미 1줄로 정리된 타 실험은 대상 아님. 애매하면(지표마다 "최고"가
갈릴 때만) 확인받고 나머지는 바로 정리.

e25_plus(2026-08-04): checkpoint-3900을 base(Qwen3.5-9B)에 merge한
`Qwen3.5-9B-e25ck3900-merged` 위에 저rank(r16)·저LR LoRA를 TSR 출신 데이터만
(`combined_e25plus_tsr6598`, prompt_style=="unified_table_with_ocr")으로 이어학습.
셀 A(lr1e-5)/B(lr3e-6)는 TSR 크롭 원본만, 셀 C(synth)는 크롭 절반을 레이아웃
페이지 크기 캔버스에 합성 배치해 실효 해상도를 낮춘 버전(GT는 100% TSR 원본,
`combined_e25plus_tsrsynth5783`). 셀별 eval_loss 최저 2개씩(모두 step 100
부근에서 조기 수렴) 총 6개 체크포인트 sweep. **LLM-Judge는 스킵**
(`SKIP_JUDGE=1`). 이전 세대 e25(ckpt200/2150/3900/4000)는 최고 성적 1줄
(ckpt3900)만 남기고 정리(§9 규칙).


**최종적으로는 이 아래 표들에 모든 열에 대한 실험을 추가해서 빈틈없는 행을 채우는게 목표**

**dp bench 500장**

| 순위 | 모델 | NID | TEDS zero(b) | TEDS-S zero(b) | TEDS skip(a) | TEDS-S skip(a) | 수식 Edit↓ | 수식 CDM↑ | 수식 match(quick_match 매칭 성공) | 수식 Edit(matched만)↓ | 수식 CDM(matched만)↑ | matched/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dots_ocr | 0.8836 | 0.7923 | 0.8020 | 0.8219 | 0.8320 | 0.0072 | 0.9961 | 40/40 (100%) | 0.0041 | 0.9960 | 294/305 |
| 2 | mineru_native | 0.9353 | 0.7259 | 0.7484 | 0.7796 | 0.8037 |  |  |  |  |  | 284/305 |
| 3 | mineru_lora | 0.9305 | 0.7251 | 0.7482 | 0.7760 | 0.8007 |  |  |  |  |  | 285/305 |
| 4 | e24_ckpt1450 (1.53ep) | 0.9090 | 0.7115 | 0.7409 | 0.7750 | 0.8071 | 0.2651 | 0.8125 | 37/40 (92.5%) | 0.2233 | 0.8784 | 280/305 |
| 5 | mineru_hybrid | 0.9026 | 0.7100 | 0.7300 | 0.7874 | 0.8096 |  |  |  |  |  | 275/305 |
| 6 | chandra_tablelayout(e18) | 0.8905 | 0.6999 | 0.7225 | 0.7936 | 0.8192 | 0.2276 | 0.8426 | (원본 산출물 없음) |  |  | 269/305 |
| 7 | **e25plus_noragged_c_ckpt500 (0.66ep, lr1e-5, +수식증강)** | 0.9158 | 0.6979 | 0.7306 | 0.7684 | 0.8044 | 0.2166 | 0.8495 | 35/40 (87.5%) | 0.0819 | **0.9708** | 277/305 |
| 8 | e25_ckpt3900 (3.91ep) | 0.9059 | 0.6951 | 0.7370 | 0.7571 | 0.8028 | 0.4551 | 0.6827 | 35/40 (87.5%) | 0.3952 | 0.7803 | 280/305 |
| 9 | chandra_layout | 0.8903 | 0.6949 | 0.7175 | 0.7679 | 0.7929 | 0.2022 | 0.8746 | 38/40 (95.0%) | 0.1925 | 0.9206 | 276/305 |
| 10 | e26_ckpt800 (0.81ep, no_wt) | 0.9043 | 0.6925 | 0.7260 | 0.7737 | 0.8111 | 0.2208 | 0.8374 | 38/40 (95.0%) | 0.2214 | 0.8814 | 273/305 |
| 11 | paddle | 0.9019 | 0.6897 | 0.7129 | 0.7763 | 0.8024 |  |  |  |  |  | 271/305 |
| 12 | e25plus_noragged_a_ckpt100 (0.14ep, lr1e-5) | 0.9136 | 0.6889 | 0.7238 | 0.7669 | 0.8057 | 0.4977 | 0.6328 | 31/40 (77.5%) | 0.3054 | 0.8166 | 274/305 |
| 13 | e25plus_a_ckpt500 (0.69ep, lr1e-5) | 0.9192 | 0.6880 | 0.7216 | 0.7631 | 0.8003 | 0.4988 | 0.6284 | 31/40 (77.5%) | 0.3071 | 0.8109 | 275/305 |
| 14 | **e27_ckpt1500 (1.46ep, +arxiv_formula)** | 0.8927 | 0.6855 | 0.7243 | 0.7548 | 0.7975 | **0.1392** | **0.9184** | 40/40 (100%) | **0.1242** | 0.9184 | 277/305 |
| 15 | e27_ckpt800 (0.78ep, +arxiv_formula) | 0.8979 | 0.6768 | 0.7094 | 0.7645 | 0.8014 | 0.1705 | 0.8980 | 40/40 (100%) | 0.1523 | 0.8980 | 270/305 |
| 16 | e22_ckpt1400 | 0.8902 | 0.6752 | 0.7126 | 0.7599 | 0.8020 |  |  |  |  |  | 271/305 |
| 17 | e27_ckpt1550 (1.51ep, best eval_loss) | 0.8978 | 0.6713 | 0.7054 | 0.7472 | 0.7852 | 0.2124 | 0.8753 | 40/40 (100%) | 0.1622 | 0.8753 | 274/305 |
| 18 | e19_ckpt200 (0.15ep) | 0.8722 | 0.5822 | 0.6042 | 0.7687 | 0.7977 |  |  |  |  |  | 231/305 |
| 19 | e27_ckpt200 (0.20ep, +arxiv_formula) | 0.8692 | 0.5330 | 0.5547 | 0.7491 | 0.7796 | 0.3970 | 0.7093 | 35/40 (87.5%) | 0.2799 | 0.8106 | 217/305 |
| 20 | e20_ckpt1000 (0.78ep, 픽셀좌표·비교주의) | 0.9173 | 0.9371 | 0.6911 | 0.7649 | - |  |  |  |  |  | - |

**e27(2026-08-11)**: `combined_e27_30000`(e26_28484 + arxiv_formula/dataset_formula_crop 17,078장을
합친 45,562건에서 30,000으로 무작위 다운샘플, [[arxiv-formula-gt-build]])로 e26 config에서
데이터셋 경로만 바꿔 학습. 학습이 step 1550(1.51ep)에서 tmux 세션이 사라진 채 멈춰 있었다(원인
미확인 — eval_loss는 1550까지 계속 단조 하락 중이었고 `best_global_step`도 1550/`best_metric`
0.06268이라 조기수렴은 아님). 관례대로 초반(ckpt200)/중반(ckpt800)/eval_loss 최저 2개(ckpt1500=
0.06413, ckpt1550=0.06268 — 마침 최신 스텝이자 최저)를 sweep. **LLM-Judge는 사용자 지시로 스킵.**

수식 Edit/CDM이 이번 세대의 핵심 성과 — **e27_ckpt1500이 Edit 0.1392/CDM 0.9184(all-sample
기준)로 dots_ocr(0.0072/0.9961) 다음으로 리더보드 전체 2위**, matched만 봐도 CDM 0.9184로
e25plus_noragged_c_ckpt500(0.9708) 다음이다. e27 네 체크포인트 전부 match율 40/40(100%,
ckpt200만 35/40)로 e26/e25 세대의 만성적인 미검출 문제가 크게 줄었다 — arxiv_formula crop
17,078장(KaTeX 렌더 검증된 `$$` 래핑 수식)을 학습 데이터에 섞은 효과가 뚜렷하다. dp500 표
탐지(TEDS zero(b))는 ckpt1500이 0.6855로 e27 최고지만 e26_ckpt800(0.6925)에는 아직 못 미침 —
수식 학습이 늘면서 표 자체 탐지력은 근소하게 트레이드오프가 있는 것으로 보인다.

**e25plus 세대 재정리(2026-08-06, e26 진입에 따라 §9 규칙 소급 적용)**: 원본
A/B/C(6줄) + noragged A/B/C 중 B 계열(원본 b_ckpt100/400, noragged b_ckpt100/400)은
전부 삭제, 원본 C(c_ckpt200/100)도 삭제 — dp500 TEDS 는 이 둘이 가장 높았지만
matched CDM(0.67~0.70)이 낮아 "미검출 빼고 봐도 품질이 낮다"는 이전 결론과 배치되지
않게 정리. 원본 A 는 a_ckpt500 만(a_ckpt100 삭제, dp500 근소 우위), noragged A/C 는
각각 최고 1줄(a_ckpt100, c_ckpt500)만 남겼다.

수식 match = dp500 수식 40개 표본 중 OmniDocBench quick_match 가 예측 텍스트를 찾아
매칭시킨 비율(나머지는 `pred=""` 로 완전 미검출 → Edit 1.0/CDM 0 처리됨). "matched만"
두 열은 그 미검출분을 빼고 매칭된 표본만으로 다시 계산한 값 — 모델이 "찾아서 낸" 수식의
실제 품질을 보여준다. **noragged_c_ckpt500 은 matched 기준 CDM 0.9708 로 dots_ocr
다음으로 확실한 1등**(all-sample 기준 0.8495 로는 chandra_layout·chandra_tablelayout에
비벼지지만, 미검출 계정을 빼면 확실히 앞선다) — 반대로 e25plus_a/e25plus_c 계열은
matched 기준으로도 CDM 0.7~0.81 대라 미검출 비율(22.5%)이 낮은 CDM 의 핵심 원인이 아니라
매칭되고도 품질 자체가 떨어짐을 보여준다. e26 은 ckpt800 이 matched 기준으로도
0.8814 로 가장 안정적이고, ckpt1550 은 matched 기준 CDM 0.7340 으로 개선되긴 하지만
(all-sample 0.5872 대비) 여전히 ckpt800 에 못 미쳐 — 미검출 증가만이 아니라 매칭된
수식 자체의 품질도 ckpt1550 이 ckpt800 보다 나쁘다는 뜻(§8 "GT 정렬 애매" 가설은
doc_0032 4건만 설명하고, 나머지 품질 저하는 실제 회귀로 보임).

(2026-08-06 최초 정리, 아래 §9 재정리 문단에서 한 번 더 갱신됨) noragged B/C 는
§9 규칙대로 각 최고 1줄(B는 전부 삭제, C는 c_ckpt500)만 남기고 정리했다 — TEDS/CDM
어느 지표로도 c_ckpt500을 못 넘었음을 확인 후 정리.

**e26(2026-08-06)**: combined_e26_28484(ragged 268장 + 수식 빈 예측 36장 제외, 수식 있는
페이지 167장은 gt_html 전체를 dots.ocr 예측으로 교체)로, e25 config에서 데이터셋 경로와
`token_weighting`(off, bbox 가중치 없음) 딱 2개만 바꿔 처음부터 새로 학습. 관례대로
초반(ckpt200)/중반(ckpt800)/eval_loss 최저 2개(ckpt1500=0.04878, ckpt1550=0.04865, 1550이
당시 최신 스텝이자 최저)를 sweep. eval_loss는 1550스텝(1.57ep)까지 계속 단조 하락 중이라
아직 수렴 전 — 이후 스텝에서 dp500/tsr200 성능이 더 개선될 가능성이 있다.

dp500 TEDS zero(b)는 eval_loss와 반대로 ckpt800(0.6925)이 가장 높고 ckpt1550(0.6784)·
ckpt1500(0.6690)·ckpt200(0.6098) 순 — eval_loss 최저가 dp500 표 탐지 성능 최고와 일치하지
않는다(§ 함정 모음 "학습이 도중에 죽으면 eval최적≈마지막"과는 다른 패턴). 수식 Edit/CDM도
ckpt800(Edit 0.2208/CDM 0.8374, noragged_c_ckpt500 다음으로 이번 sweep 전체 2등)이 가장
좋고 ckpt1550(Edit 0.3832/CDM 0.5872)이 가장 나쁘다 — 다만 dp500 수식 샘플이 40개뿐이고,
ckpt1550의 빈 예측 8건 중 4건(doc_0032)은 ckpt1500과 겹쳐 모델이 아니라 그 문서의 GT 정렬
자체가 애매한 것으로 보인다(§8 함정 모음). 나머지 4건(doc_0061/172×2/397)만 ckpt1550
고유 실패라 표본이 작아 노이즈일 가능성을 배제 못 함 — 추가 sweep(2000~3000 스텝대)으로
재확인 필요.

원래 e25plus 세대(A/B/C, `combined_e25plus_tsr6598`)는 **A(원본 크롭만, lr1e-5)
한 셀만 남기고 B/C 는 사용자 지시로 디스크 산출물까지 삭제**했다. 표 기록은
한동안 §9 규칙과 무관하게(당시 최신 세대라는 이유로) 6줄 다 남겨뒀었는데, e26
진입 후에도 정리가 안 된 채로 두 세대를 그냥 지나쳐 버렸다 — 2026-08-06 에
뒤늦게 §9 규칙 소급 적용해서 A 는 a_ckpt500 한 줄, B/C 는 전부 삭제(위 표
"e25plus 세대 재정리" 문단 참고). 새 세대
`e25plus_noragged`(2026-08-04~05)는 combined_e25plus_tsr6598 에서 렌더링하면
사각형이 아닌(rowspan/colspan 불일치) 표 42건을 실제 Chromium 픽셀 비교로
제거한 `combined_e25plus_tsr6598_noragged`(6,556장, [[tsr-ragged-table-detection]]
참고)로 A/B 를 재실행했고, C 는 이번엔 synth(합성 배치)가 아니라 **noragged 위에
수식 있는 layout 페이지 351장을 얹되 그 페이지의 Formula div 만 dots.ocr 예측으로
교체**(containment 매칭, train 545/637=85.6%)한 데이터로 바꿨다(rank/LR 은 A와
동일 16/64, 1.0e-5 — 데이터가 유일한 변인).

dp500(페이지 안 표)에서 새 세대 최고는 **noragged_c_ckpt500**(TEDS zero(b)
0.6979, matched 277/305) — e25_ckpt3900 기준선(0.6951/280)의 TEDS 는 넘었지만
matched 는 여전히 못 미친다. 원래 세대의 synth 셀(0.7069/280, 0.7017/284)에는
아직 못 미침 — "합성 배치로 저해상도 조건을 흉내"가 "noragged+수식 데이터 추가"
보다 dp500 in-page 표 탐지에는 더 효과적이었다는 뜻. B/A noragged 는 원래
세대와 비슷하거나 소폭 낮음(0.686~0.692 대).

**수식 Edit/CDM — noragged_c_ckpt500 이 이번 sweep 전체 2등(dots_ocr 제외).**
Edit 0.2166 / CDM 0.8495 로 e25_ckpt3900 기준선(0.4551/0.6827)은 물론
chandra_tablelayout(0.2276/0.8426)·chandra_layout(0.2022/0.8746)급까지
따라잡았다 — dp500 리더보드 전체에서 dots_ocr(0.0072/0.9961) 다음으로 좋은
수식 점수다. 같은 noragged_c 라도 ckpt100 은 Edit 0.4856/CDM 0.6436 으로
평범해서, **500스텝까지 더 학습되면서 수식 능력이 급격히 좋아진 것**으로
보인다(학습 데이터에 수식 545~637개가 섞여 있었던 게 주효). noragged_a/b(수식
데이터 없음, 원본 세대와 동일 구성)는 Edit 0.46~0.50/CDM 0.62~0.63 로 원래
세대와 비슷한 수준의 회귀를 그대로 반복했다 — "TSR 크롭만 학습하면 수식이
퇴화한다"는 원래 세대의 결론을 재확인하면서, 동시에 **"수식 있는 페이지를
소량(전체 학습데이터의 ~5%)만 섞어도 회귀가 거의 완전히 해소된다"**는 걸
보여준 결과. noragged_c_ckpt500 이 dp500 TEDS·matched·수식 품질을 종합했을 때
이번 세대(그리고 원래 e25plus 세대 포함) 전체에서 가장 균형 잡힌 체크포인트.

**tsr 200장** (`(스킵)`=LLM-Judge 미실행)

| 순위 | 모델 | TEDS | TEDS-S | Span F1 | Attr Acc | LLM-Judge/10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | e18_6910 ckpt-500 | 0.9484 | 0.9621 | 0.8005 | 0.8539 | - |
| 2 | tablelayout ckpt-1050 | 0.9439 | 0.9646 | 0.7775 | 0.8293 | 6.81 |
| 3 | e18_nodiv ckpt-400 | 0.9430 | 0.9606 | 0.7677 | 0.8247 | 7.08 |
| 4 | mineru_hybrid | 0.9417 | 0.9682 | 0.8595 | 0.8879 | 6.42 |
| 5 | table ckpt-200 | 0.9384 | 0.9551 | 0.7293 | 0.7879 | 6.91 |
| 6 | e26_ckpt1550 (1.57ep, no_wt) | 0.9253 | 0.9463 | 0.7519 | 0.8120 | (스킵) |
| 7 | e27_ckpt1500 (1.46ep, +arxiv_formula) | 0.9227 | 0.9402 | 0.7275 | 0.7879 | (스킵) |
| 8 | e25plus_a_ckpt500 (0.69ep, lr1e-5) | 0.9216 | 0.9423 | 0.7886 | 0.8532 | (스킵) |
| 9 | e19_ckpt1850 (1.42ep) | 0.9214 | 0.9440 | 0.7149 | 0.7852 | 6.82 |
| 10 | e24_ckpt1450 (1.53ep) | 0.91911 | 0.9376 | 0.7222 | 0.8056 | (스킵) |
| 11 | e25plus_noragged_a_ckpt100 (0.14ep, lr1e-5) | 0.9184 | 0.9409 | 0.7929 | 0.8556 | (스킵) |
| 12 | mineru_lora | 0.9182 | 0.9434 | 0.8425 | 0.8688 | 6.14 |
| 13 | paddle | 0.9181 | 0.9613 | 0.8549 | 0.8872 | 6.00 |
| 14 | e25plus_noragged_c_ckpt500 (0.66ep, lr1e-5, +수식증강) | 0.9179 | 0.9388 | 0.7940 | 0.8596 | (스킵) |
| 15 | e20_ckpt1000 (0.78ep) | 0.9173 | 0.9371 | 0.6911 | 0.7649 | - |
| 16 | e22_ckpt1000 (1.72ep) | 0.9167 | 0.9357 | - | - | 6.84 |
| 17 | e25_ckpt3900 (3.91ep) | 0.9149 | 0.9377 | 0.7459 | 0.8111 | 7.413 |
| 18 | e27_ckpt800 (0.78ep, +arxiv_formula) | 0.9111 | 0.9335 | 0.7011 | 0.7790 | (스킵) |
| 19 | e27_ckpt1550 (1.51ep, best eval_loss) | 0.9072 | 0.9259 | 0.7072 | 0.7805 | (스킵) |
| 20 | mineru_native | 0.9000 | 0.9298 | 0.8234 | 0.8600 | 5.92 |
| 21 | e27_ckpt200 (0.20ep, +arxiv_formula) | 0.8694 | 0.8943 | 0.5780 | 0.6694 | (스킵) |
| 22 | dots_ocr | 0.8254 | 0.8530 | 0.4588 | 0.5290 | 4.73 |

**e27 tsr200(크롭 단독)**: dp500/수식과 반대로 e26_ckpt1550(0.9253)을 못 넘었다 — e27 최고인
ckpt1500이 0.9227로 근소하게 밀린다(7위→e26 다음 순위). arxiv_formula crop 데이터가 늘면서
표 크롭 자체에 대한 과최적화는 약간 줄어든 것으로 보이나, dp500 표 탐지·수식 품질 개선폭에 비하면
미미한 트레이드오프. e26 세대는 이번 sweep(§9 규칙)으로 각 표 자체 기준 최고 1줄만 남겼다 —
dp500 표는 ckpt800(TEDS zero(b) 최고), tsr200/complexity 표는 ckpt1550(TEDS 최고) 유지, 나머지
(ckpt200/1500, 그리고 표별로 다른 쪽)는 삭제.

e26은 eval_loss와 반대로 tsr200(크롭 단독)에서는 스텝이 늘수록 계속 좋아진다 —
ckpt1550(TEDS 0.9253)이 이번 세대 전체 6위로 e25plus_a 계열을 처음 앞질렀고
ckpt1500(0.9235)도 근소 차 7위. dp500과 tsr200이 반대 방향을 가리키므로(§9 "애매하면
확인" 케이스), 어느 체크포인트를 최종으로 쓸지는 사용자 확인 필요 — dp500 종합(표
탐지+수식)은 ckpt800, tsr200 크롭 성능은 ckpt1550이 각각 우세하다.

(2026-08-06 기준 과거 기록, 아래 noragged_c_ckpt100/b_ckpt400 행은 e26 진입 시
§9 정리 규칙으로 표에서 삭제됨 — c_ckpt500 만 남음) tsr200(크롭 단독)에서는
원래 세대와 마찬가지로 A 계열이 강세다 — 이번 세대 1등은 **noragged_c_ckpt100**
(TEDS 0.9226, +수식증강인데도 크롭 성능이 A 만큼 나옴) 이고
noragged_b_ckpt400(0.9215)도 근소 차. noragged_a 는 0.9184~0.9185로
원래 A(0.9216~0.9229)보다 오히려 살짝 낮다 — ragged 42건을 뺀 게 tsr200
크롭 성능 자체에는 도움이 안 됐다는 뜻(에러 있는 42건도 tsr200 200장과는
겹치지 않는 별개 표라 직접 영향은 아니고, 단순 train 데이터가 줄어든 효과로
보임). noragged_c(수식증강, train 6034)가 A/B(5748)보다 데이터가 많은데도
tsr200 성능이 밀리지 않는 건 고무적 — dp500 §ckpt500 결과와 같이 보면
noragged_c 가 이번 세대에서 가장 균형 잡힌 셀.

**tsr 200장 — complexity별 TEDS** (§7)

| run | overall | simple(30) | medium(50) | complex(40) | complex_col(30) | complex_mix(30) | complex_row(20) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dots_ocr | 0.8254 | 0.9745 | 0.9035 | 0.6786 | 0.7818 | 0.8579 | 0.7170 |
| table(final) | 0.9262 | 0.9875 | 0.9829 | 0.8674 | 0.8325 | 0.9492 | 0.9162 |
| tablelayout(final) | 0.9374 | 0.9906 | 0.9737 | 0.8927 | 0.8789 | 0.9618 | 0.9072 |
| e18_nodiv(final) | 0.9408 | 0.9913 | 0.9828 | 0.8757 | 0.8945 | 0.9528 | 0.9419 |
| e24_ckpt1450 | 0.9191 | 0.9872 | 0.9770 | 0.8454 | 0.8870 | 0.8900 | 0.9111 |
| e25_ckpt3900 | 0.9149 | 0.9853 | 0.9673 | 0.8590 | 0.8612 | 0.8948 | 0.9011 |
| e25plus_a_ckpt500 | 0.9216 | 0.9819 | 0.9745 | 0.8720 | 0.8745 | 0.8766 | 0.9359 |
| e25plus_noragged_a_ckpt100 | 0.9184 | 0.9819 | 0.9738 | 0.8536 | 0.8746 | 0.8799 | 0.9378 |
| e25plus_noragged_c_ckpt500 | 0.9179 | 0.9819 | 0.9753 | 0.8533 | 0.8593 | 0.8878 | 0.9402 |
| e26_ckpt1550 | 0.9253 | 0.9844 | 0.9834 | 0.8817 | 0.8563 | 0.8953 | 0.9272 |
| e27_ckpt200 | 0.8694 | 0.9777 | 0.9492 | 0.7922 | 0.7963 | 0.8218 | 0.8428 |
| e27_ckpt800 | 0.9111 | 0.9857 | 0.9774 | 0.8375 | 0.8382 | 0.8837 | 0.9310 |
| e27_ckpt1500 | 0.9227 | 0.9871 | 0.9807 | 0.8834 | 0.8354 | 0.8920 | 0.9369 |
| e27_ckpt1550 | 0.9072 | 0.9864 | 0.9756 | 0.8553 | 0.7928 | 0.8886 | 0.9204 |

## 함정 모음

- dp500은 항상 WITH-OCR(bbox 좌표를 프롬프트로 줌 — NO-OCR과 같은 표에서
  절대 비교 금지). tsr200 매니페스트는 원래부터 WITH-OCR 고정.
- 학습이 도중에 죽으면 "eval최적 N개" ≈ "마지막 N개"인 경우가 많음.
- MAXPX/MINPX가 학습 config와 다르면(특히 min_pixels 기본값 262144 ≠ 학습 65536)
  추론 해상도가 어긋남 — 워커 스크립트에 명시 확인.
- `latexmlc: FileNotFoundError` 로그는 CDM 계산과 무관한 별개 도구라 무해.
