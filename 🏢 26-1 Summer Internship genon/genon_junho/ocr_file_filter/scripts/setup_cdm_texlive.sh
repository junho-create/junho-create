#!/usr/bin/env bash
# CDM(수식 채점, ocr_filter/metrics/cdm.py) 실행에 필요한 TeX Live를 프로젝트 루트 기준
# 유저 디렉토리에 설치한다 (root 불필요, 시스템 경로 안 건드림, 이 프로젝트를 통째로 다른
# 서버로 옮겨도 이 스크립트만 재실행하면 재현됨). 원본 절차는
# hckim/tsr_test_issue318(doc_parser#318)의 CDM_SETUP_b200.md 를 참고해 경로만 이
# 프로젝트(ocr_file_filter) 기준 상대경로로 바꾼 것 — 그 디렉토리 자체는 참조/의존하지 않음.
#
# 설치 후 위치 (전부 .gitignore 처리됨, git엔 안 올라감):
#   _texlive/2025/bin/x86_64-linux/{pdflatex,tlmgr}
#   _texlive/bin/magick        (ImageMagick6 convert 로 위임하는 shim)
#   _texlive/im_policy/policy.xml
#
# 이미 설치돼 있으면 다시 안 받고 스킵(멱등) — 재설치하려면 _texlive/ 를 지우고 재실행.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TL="$ROOT/_texlive"
INSTALLER="$ROOT/_texlive_installer"

if [ -x "$TL/2025/bin/x86_64-linux/pdflatex" ]; then
  echo "이미 설치됨: $TL/2025/bin/x86_64-linux/pdflatex -- 스킵"
else
  echo "=== 1. TeX Live 유저 설치 (수 GB, 수십 분 걸릴 수 있음) ==="
  mkdir -p "$INSTALLER"
  cd "$INSTALLER"
  if [ ! -f install-tl-unx.tar.gz ]; then
    wget -q https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
  fi
  tar xzf install-tl-unx.tar.gz
  cd install-tl-2*

  perl ./install-tl --scheme=scheme-small --no-interaction \
    --texdir="$TL/2025" \
    --texuserdir="$TL/user"
fi

export PATH="$TL/2025/bin/x86_64-linux:$PATH"
which pdflatex tlmgr

echo "=== 2. 추가 LaTeX 패키지 (CJK + CDM 템플릿용 + 한국어) ==="
# upgreek 은 별도 패키지가 아니라 was 패키지 안의 upgreek.sty 로 제공됨(미러에 upgreek
# 단독 패키지 없음, 2026-07-19 실측) -- was 만 설치하면 됨. gkai(중국어 CJK 폰트)도 이
# 미러엔 없는데, 우리는 CDM_CJK_FONT=mj(한국어 폰트, cjk-ko/uhc 로 제공)로 고정해서 쓰므로
# 없어도 무방 -- 목록에서 뺌.
tlmgr install cjk cjkutils arphic multirow was booktabs \
  cjk-ko nanumtype1 uhc 2>&1 | tail -20

echo "=== 3. magick shim (IM6 convert 로 위임) ==="
mkdir -p "$TL/bin"
printf '#!/bin/sh\nexec convert "$@"\n' > "$TL/bin/magick"
chmod +x "$TL/bin/magick"
export PATH="$TL/bin:$PATH"
magick -version | head -1

echo "=== 4. ImageMagick PDF 읽기 정책 (유저 정책, 시스템 정책 안 건드림) ==="
mkdir -p "$TL/im_policy"
printf '<policymap>\n<policy domain="coder" rights="read|write" pattern="PDF" />\n</policymap>\n' \
  > "$TL/im_policy/policy.xml"

echo
echo "=== 5. 렌더 스모크 테스트 ==="
export MAGICK_CONFIGURE_PATH="$TL/im_policy"
TMPD=$(mktemp -d)
cd "$TMPD"

cat > t1.tex <<'EOF'
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,xcolor,booktabs,multirow,upgreek,geometry}
\pagestyle{empty}
\begin{document}
$g_t = Fg_{t-1} + Wv_t, v_t \sim N(0, \Sigma)$
\end{document}
EOF
pdflatex -interaction=nonstopmode t1.tex >/dev/null && magick -density 200 t1.pdf t1.png && echo "TEST1(영문) OK"

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
pdflatex -interaction=nonstopmode t2.tex >/dev/null && magick -density 200 t2.pdf t2.png && echo "TEST2(한글) OK"

cd /
rm -rf "$TMPD"

echo
echo "설치 완료. 이 프로젝트 안에서 CDM 사용시 필요한 환경변수:"
echo "  export PATH=\"$TL/2025/bin/x86_64-linux:$TL/bin:\$PATH\""
echo "  export MAGICK_CONFIGURE_PATH=\"$TL/im_policy\""
echo "  export CDM_CJK_FONT=mj"
