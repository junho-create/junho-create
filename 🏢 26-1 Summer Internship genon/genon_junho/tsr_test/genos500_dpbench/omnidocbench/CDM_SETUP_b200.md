# CDM 실행 환경 셋업 (b200, root 불필요)

이슈 doc_parser#318. CDM은 GT/pred 수식을 실제 렌더링(pdflatex→PDF→PNG)해
이미지 레벨로 문자 매칭하는 metric이라 시스템 바이너리가 필요하다.
b200 현황: `convert`(ImageMagick 6)·`gs` 있음, `pdflatex` 없음, docker 없음.
→ TeX Live를 **유저 디렉토리에 설치**한다(시스템 경로 안 건드림, `rm -rf`로 제거 가능).

사전 파악된 요구사항(upstream 코드 근거):

- pdflatex 탐색: PATH 또는 env `CDM_PDFLATEX` (`src/metrics/cdm/modules/texlive_env.py:142`)
- ImageMagick은 **`magick` 명령을 하드코딩** 호출 (`latex2bbox_color.py:208`)
  → IM6만 있는 서버에선 `convert`로 넘겨주는 shim 필요
- LaTeX 패키지: geometry, booktabs, multirow, amssymb, upgreek, amsmath, xcolor, CJK
- CJK 폰트 기본값 `gkai`(중국어)는 **한글 글리프가 없음** → 한국어 폰트 설치 후
  env `CDM_CJK_FONT`로 교체 (GT 수식에 `\text{신용위험스프레드}` 등 한글 존재)

## 1. TeX Live 유저 설치 (1회, 수 GB·수십 분)

```bash
HCKIM=/NHNHOME/WORKSPACE/0426030039_A/hckim
mkdir -p $HCKIM/texlive_installer && cd $HCKIM/texlive_installer
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar xzf install-tl-unx.tar.gz && cd install-tl-2*

perl ./install-tl --scheme=scheme-small --no-interaction \
  --texdir=$HCKIM/texlive/2025 \
  --texuserdir=$HCKIM/texlive/user

export PATH=$HCKIM/texlive/2025/bin/x86_64-linux:$PATH
which pdflatex tlmgr   # 둘 다 hckim 경로로 나와야 함
```

## 2. 추가 LaTeX 패키지 (CJK + CDM 템플릿용 + 한국어)

```bash
tlmgr install cjk cjkutils arphic gkai multirow upgreek was booktabs \
  cjk-ko nanumtype1 uhc
```

## 3. `magick` shim (IM6 convert 위임)

```bash
mkdir -p $HCKIM/bin
printf '#!/bin/sh\nexec convert "$@"\n' > $HCKIM/bin/magick
chmod +x $HCKIM/bin/magick
export PATH=$HCKIM/bin:$PATH
magick -version   # ImageMagick 6.x 출력이면 OK
```

## 4. 렌더 스모크 테스트 (전체 실행 전 필수)

```bash
cd $(mktemp -d)
# 4-1) 영문 수식 + PDF→PNG
cat > t1.tex <<'EOF'
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,xcolor,booktabs,multirow,upgreek,geometry}
\pagestyle{empty}
\begin{document}
$g_t = Fg_{t-1} + Wv_t, v_t \sim N(0, \Sigma)$
\end{document}
EOF
pdflatex -interaction=nonstopmode t1.tex && magick -density 200 t1.pdf t1.png && echo "TEST1 OK"

# 4-2) 한글 수식 (CDM과 동일한 CJK 환경, 폰트=mj)
cat > t2.tex <<'EOF'
\documentclass[12pt]{article}
\usepackage{amsmath,CJK}
\pagestyle{empty}
\begin{document}
\begin{CJK}{UTF8}{mj}
$\text{신용위험스프레드} = \max(PD + CoD, 0)$
\end{CJK}
\end{document}
EOF
pdflatex -interaction=nonstopmode t2.tex && magick -density 200 t2.pdf t2.png && echo "TEST2 OK"
```

- TEST1 실패(`not authorized` 등) → IM6 정책이 PDF 읽기를 막는 경우.
  유저 정책으로 우회:
  ```bash
  mkdir -p $HCKIM/im_policy
  printf '<policymap>\n<policy domain="coder" rights="read|write" pattern="PDF" />\n</policymap>\n' \
    > $HCKIM/im_policy/policy.xml
  export MAGICK_CONFIGURE_PATH=$HCKIM/im_policy
  ```
- TEST2 실패(`c70mj.fd not found` 등) → `tlmgr install cjk-ko nanumtype1 uhc` 재확인,
  그래도 안 되면 t2.tex의 `{mj}`를 `{nanummj}`로 바꿔 재시도. **성공한 폰트명을
  5단계의 `CDM_CJK_FONT`에 그대로 쓴다.**
- t1.png/t2.png를 열어 글자(특히 한글)가 실제로 보이는지 확인.

## 5. CDM 포함 전체 평가 실행

Phase 1b와 같은 wrapper에 `WITH_CDM=1`만 추가 (매칭·TEDS도 다시 돌지만 수 분 수준):

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/hckim/tsr_test_issue318
HCKIM=/NHNHOME/WORKSPACE/0426030039_A/hckim
export PATH=$HCKIM/texlive/2025/bin/x86_64-linux:$HCKIM/bin:$PATH
export CDM_CJK_FONT=mj   # 4-2에서 성공한 폰트명

WITH_CDM=1 \
OMNIDOCBENCH_ROOT=$HCKIM/upstream_OmniDocBench \
GT_JSON=genos500_dpbench/genos-500-set/reference_dp_bench.json \
IMAGES_DIR=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_data/500_genos_val_set/images \
DOTS_PRED=genos500_dpbench/dots_pred_from_jhyeo.json \
CHANDRA_LAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_layout/predictions_unified.jsonl \
CHANDRA_TABLELAYOUT_JSONL=/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/eval_results/genos500_ocr_tablelayout_epoch3/predictions_unified.jsonl \
nohup bash genos500_dpbench/omnidocbench/run_omnidocbench_genos500.sh > omni_run_cdm.log 2>&1 &
```

결과: `result/pred_md_<tag>_quick_match_metric_result.json`의 `display_formula`에
CDM(F1/recall/precision)이 추가되고, `result/<save_name>/CDM/`에 렌더·매칭
시각화 PNG가 남는다. 렌더 실패 수식은 `metric_result.json`의 `metric_debug`에 기록됨
— 한글 수식(30개 중 다수)이 실패 목록에 있으면 폰트 문제이므로 4-2로 돌아갈 것.
