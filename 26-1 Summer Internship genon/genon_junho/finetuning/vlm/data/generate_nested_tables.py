"""
Synthetic Nested Table 데이터 생성 스크립트

기존 simple/medium 테이블을 조합하여 nested table 학습 데이터를 합성한다.
- 부모 테이블: simple/medium 복잡도, small/medium 크기, 중첩 없음, GT 품질 이슈 없음
- 자식 테이블: small 크기, 중첩 없음, GT 품질 이슈 없음
- colspan/rowspan 없는 일반 td 셀에 자식 테이블 삽입
- Playwright로 HTML → PNG 렌더링 + td/th DOM bbox 추출
- Slim 포맷 JSONL 저장

Usage:
    python -m data.generate_nested_tables \\
        --index_path data/index/training_index.jsonl \\
        --output_dir data/experiments/nested_synthetic \\
        --count 5000 \\
        --n_child_per_parent 1 \\
        --dedup_mode html_or_source_pair \\
        --exclude_jsonl data/experiments/nested_synthetic_prev/nested_synthetic.jsonl \\
        --gt_quality_filter \\
        --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.html_utils import normalize_html  # noqa: E402


# ---------------------------------------------------------------------------
# HTML 렌더링 템플릿
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_BASE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 16px;
      font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
      font-size: {font_size};
      background: white;
      color: #111;
    }}
    table {{
      border-collapse: collapse;
    }}
    td, th {{
      border: {border_width} solid {border_color};
      padding: {cell_padding};
      vertical-align: middle;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .nested-parent-cell-text {{
      line-height: 1.25;
    }}
    .nested-parent-cell-text-top {{
      margin: 0 0 4px 0;
    }}
    .nested-parent-cell-text-bottom {{
      margin: 4px 0 0 0;
    }}
    thead th {{
      background: {header_bg};
      font-weight: bold;
    }}
    /* 중첩 테이블: 별도 마진 없이 셀 내부에 꽉 채움 */
    td > table, th > table {{
      margin: 0;
      width: 100%;
    }}
  </style>
</head>
<body>
{table_html}
</body>
</html>
"""


def _build_html_template(rng: random.Random, table_html: str = "") -> str:
    """랜덤 스타일을 적용한 완성된 HTML을 반환한다.

    _HTML_TEMPLATE_BASE의 CSS 중괄호({{/}})와 {table_html} 플레이스홀더를
    한 번의 .format() 호출로 모두 치환한다.
    두 번 나눠 format()을 호출하면 CSS 내 단독 '{}'가 ValueError를 유발한다.
    """
    border_color = rng.choice(["#111", "#333", "#555", "#888"])
    border_width = rng.choice(["1px", "2px"])
    header_bg    = rng.choice(["#e8e8e8", "#d0dce8", "#e8e0d0", "#d8e8d8"])
    font_size    = rng.choice(["12px", "13px", "14px"])
    cell_padding = rng.choice(["2px 6px", "4px 8px", "6px 10px"])
    return _HTML_TEMPLATE_BASE.format(
        border_color=border_color,
        border_width=border_width,
        header_bg=header_bg,
        font_size=font_size,
        cell_padding=cell_padding,
        table_html=table_html,
    )


# ---------------------------------------------------------------------------
# 인덱스 로딩 및 후보 필터링
# ---------------------------------------------------------------------------

def load_index(index_path: str) -> list[dict]:
    """인덱스 JSONL을 로드한다."""
    entries: list[dict] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _build_html_dedup_key(html: str) -> str:
    """중복 판별용 HTML 키를 생성한다."""
    html = (html or "").strip()
    if not html:
        return ""
    try:
        return normalize_html(html)
    except Exception:
        return html


def _build_source_pair_key(
    parent_source: str,
    child_source: str,
) -> Optional[tuple[str, str]]:
    """중복 판별용 (parent_source, child_source) 키를 생성한다."""
    parent = (parent_source or "").strip()
    child = (child_source or "").strip()
    if not parent or not child:
        return None
    return (parent, child)


def _use_html_dedup(dedup_mode: str) -> bool:
    return dedup_mode in ("html", "html_or_source_pair")


def _use_source_pair_dedup(dedup_mode: str) -> bool:
    return dedup_mode in ("source_pair", "html_or_source_pair")


def load_existing_dedup_keys(
    jsonl_paths: list[Path],
    dedup_mode: str,
) -> tuple[set[str], set[tuple[str, str]], dict]:
    """
    기존 생성 JSONL에서 dedup 기준 키를 로드한다.

    Returns:
        (seen_html_keys, seen_source_pair_keys, stats)
        - seen_html_keys: 중복 제외에 사용할 gt_html 키 집합
        - seen_source_pair_keys: 중복 제외에 사용할 source pair 키 집합
        - stats: 로드 통계
    """
    use_html = _use_html_dedup(dedup_mode)
    use_source_pair = _use_source_pair_dedup(dedup_mode)
    seen_html_keys: set[str] = set()
    seen_source_pair_keys: set[tuple[str, str]] = set()

    stats = {
        "paths": [],
        "rows": 0,
        "invalid_json_lines": 0,
        "missing_html_rows": 0,
        "missing_source_pair_rows": 0,
    }

    for path in jsonl_paths:
        if not path.exists():
            raise FileNotFoundError(f"기존 데이터 파일을 찾을 수 없습니다: {path}")
        if not path.is_file():
            raise ValueError(f"기존 데이터 경로가 파일이 아닙니다: {path}")

        stats["paths"].append(str(path))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["rows"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json_lines"] += 1
                    continue

                if use_html:
                    html = rec.get("gt_html") or rec.get("normalized_html") or ""
                    html_key = _build_html_dedup_key(html)
                    if html_key:
                        seen_html_keys.add(html_key)
                    else:
                        stats["missing_html_rows"] += 1

                if use_source_pair:
                    source_pair_key = _build_source_pair_key(
                        rec.get("parent_source", ""),
                        rec.get("child_source", ""),
                    )
                    if source_pair_key:
                        seen_source_pair_keys.add(source_pair_key)
                    else:
                        stats["missing_source_pair_rows"] += 1

    stats["loaded_html_keys"] = len(seen_html_keys)
    stats["loaded_source_pair_keys"] = len(seen_source_pair_keys)
    # backward compatibility
    stats["loaded_keys"] = stats["loaded_html_keys"]
    return seen_html_keys, seen_source_pair_keys, stats


def _get_table_size_cat(entry: dict) -> str:
    """table_size_cat 필드 없는 구버전 인덱스 호환: num_rows*num_cols 로 계산."""
    cat = entry.get("table_size_cat")
    if cat:
        return cat
    cells = entry.get("table_size_cells") or (
        entry.get("num_rows", 0) * entry.get("num_cols", 0)
    )
    if cells <= 20:
        return "small"
    if cells <= 80:
        return "medium"
    return "large"


def filter_parent_candidates(entries: list[dict]) -> list[dict]:
    """
    부모 테이블 후보 필터링.
    조건: complexity in [simple, medium], table_size_cat in [small, medium], 중첩 없음
    """
    return [
        e for e in entries
        if e.get("complexity") in ("simple", "medium")
        and _get_table_size_cat(e) in ("small", "medium")
        and e.get("nested_table_count", 0) == 0
        and e.get("normalized_html")
    ]


def filter_child_candidates(entries: list[dict]) -> list[dict]:
    """
    자식 테이블 후보 필터링.
    조건: table_size_cat == small, 중첩 없음
    (복잡도는 모두 허용하여 다양성 확보)
    """
    return [
        e for e in entries
        if _get_table_size_cat(e) == "small"
        and e.get("nested_table_count", 0) == 0
        and e.get("normalized_html")
    ]


# ---------------------------------------------------------------------------
# Nested HTML 생성
# ---------------------------------------------------------------------------

def create_nested_html(
    parent_html: str,
    child_html: str,
    n_cells: int = 1,
    min_parent_text_len: int = 10,
    rng: Optional[random.Random] = None,
) -> tuple[str, int]:
    """
    부모 테이블의 일반 td 셀(rowspan/colspan 없음)에 자식 테이블을 삽입한다.

    Args:
        parent_html: 부모 테이블 HTML
        child_html:  자식 테이블 HTML
        n_cells:     삽입할 셀 수 (1~2 권장)
        min_parent_text_len: 삽입 대상 부모 셀 최소 텍스트 길이(공백 제외)
        rng:         random.Random 인스턴스

    Returns:
        (nested_html, n_inserted): 결과 HTML과 실제 삽입된 셀 수.
        삽입 불가능한 경우 n_inserted=0을 반환한다.
    """
    if rng is None:
        rng = random.Random()

    soup = BeautifulSoup(parent_html, "lxml")
    table = soup.find("table")
    if table is None:
        return parent_html, 0

    # span 없는 일반 td 중 텍스트 길이(공백 제외)가 기준 이상인 셀만 선택
    # (th는 헤더, 각 행 첫 번째 td는 왼쪽 열이므로 제외)
    plain_tds_with_text = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        for idx, td in enumerate(tds):
            if idx == 0:  # 왼쪽 열 제외
                continue
            if int(td.get("rowspan", 1)) == 1 and int(td.get("colspan", 1)) == 1:
                cell_text = td.get_text(" ", strip=True)
                text_len_no_ws = len("".join(cell_text.split()))
                if text_len_no_ws >= min_parent_text_len:
                    plain_tds_with_text.append((td, cell_text))
    if not plain_tds_with_text:
        return parent_html, 0

    # 자식 테이블 추출
    child_soup = BeautifulSoup(child_html, "lxml")
    child_table = child_soup.find("table")
    if child_table is None:
        return parent_html, 0

    # 랜덤으로 n_cells개 선택하여 자식 테이블 삽입
    # 삽입 셀의 기존 텍스트는 위/아래/위아래 위치로 랜덤 유지한다.
    n_to_insert = min(n_cells, len(plain_tds_with_text))
    chosen = rng.sample(plain_tds_with_text, n_to_insert)
    for td, cell_text in chosen:
        text_position = rng.choice(("top", "bottom", "both"))
        td.clear()

        if text_position in ("top", "both"):
            top_text = soup.new_tag("div")
            top_text["class"] = [
                "nested-parent-cell-text",
                "nested-parent-cell-text-top",
            ]
            top_text.string = cell_text
            td.append(top_text)

        td.append(copy.copy(child_table))

        if text_position in ("bottom", "both"):
            bottom_text = soup.new_tag("div")
            bottom_text["class"] = [
                "nested-parent-cell-text",
                "nested-parent-cell-text-bottom",
            ]
            bottom_text.string = cell_text
            td.append(bottom_text)

    result_table = soup.find("table")
    if result_table is None:
        return parent_html, 0

    return normalize_html(str(result_table)), n_to_insert


# ---------------------------------------------------------------------------
# Playwright 렌더러
# ---------------------------------------------------------------------------

class NestedTableRenderer:
    """
    Playwright 기반 테이블 렌더러.

    단일 Browser 인스턴스를 재사용하며, 각 렌더링마다 독립된 BrowserContext를
    생성/소멸하여 격리를 보장한다.
    """

    def __init__(
        self,
        viewport_width: int = 1400,
        device_scale_factor: float = 2.0,
        bbox_scale: int = 1024,
        wait_ms: int = 300,
    ):
        self.viewport_width = viewport_width
        self.device_scale_factor = device_scale_factor
        self.bbox_scale = bbox_scale
        self.wait_ms = wait_ms
        self._playwright = None
        self._browser = None

    def start(self) -> None:
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)

    def stop(self) -> None:
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def render(
        self,
        table_html: str,
        output_png: Path,
        rng: Optional[random.Random] = None,
    ) -> list[dict]:
        """
        테이블 HTML을 렌더링하여 PNG 저장 후, td/th 셀의 OCR bbox를 반환한다.

        Returns:
            ocr_info: [{"text": str, "bbox": [x1, y1, x2, y2]}, ...]
                      bbox 값은 0~bbox_scale 범위로 정규화됨
        """
        if rng is None:
            rng = random.Random()
        full_html = _build_html_template(rng, table_html=table_html)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(full_html)
            tmp_path = Path(tmp.name)

        try:
            return self._do_render(tmp_path, output_png)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _do_render(self, html_path: Path, output_png: Path) -> list[dict]:
        assert self._browser is not None, "renderer.start()를 먼저 호출하세요"

        context = self._browser.new_context(
            viewport={"width": self.viewport_width, "height": 900},
            device_scale_factor=self.device_scale_factor,
        )
        page = context.new_page()
        try:
            page.goto(
                f"file://{html_path.resolve()}",
                wait_until="networkidle",
                timeout=30000,
            )
            if self.wait_ms > 0:
                page.wait_for_timeout(self.wait_ms)

            # 페이지 전체 크기 (CSS 픽셀, 스크롤 포함)
            page_dims = page.evaluate(
                "() => ({"
                "  w: document.documentElement.scrollWidth,"
                "  h: document.documentElement.scrollHeight"
                "})"
            )
            page_w = max(page_dims.get("w", self.viewport_width), 1)
            page_h = max(page_dims.get("h", 900), 1)

            # td/th 셀의 텍스트와 bbox 추출
            # getBoundingClientRect()는 scrollY=0 상태에서 document 좌표와 동일
            cells_info: list[dict] = page.evaluate("""
                () => {
                    const items = [];
                    document.querySelectorAll('td, th').forEach(el => {
                        const rect = el.getBoundingClientRect();
                        const text = (el.innerText || el.textContent || '').trim();
                        if (rect.width > 0 && rect.height > 0) {
                            items.push({
                                text: text,
                                x: rect.left,
                                y: rect.top,
                                w: rect.width,
                                h: rect.height
                            });
                        }
                    });
                    return items;
                }
            """)

            output_png.parent.mkdir(parents=True, exist_ok=True)

            # table 요소의 bounding box를 가져와 여백 포함 크롭 스크린샷
            table_box = page.evaluate("""
                () => {
                    const table = document.querySelector('table');
                    if (!table) return null;
                    const rect = table.getBoundingClientRect();
                    return {x: rect.left, y: rect.top, width: rect.width, height: rect.height};
                }
            """)
            if table_box:
                pad = 16
                clip = {
                    "x":      max(0, table_box["x"] - pad),
                    "y":      max(0, table_box["y"] - pad),
                    "width":  table_box["width"]  + pad * 2,
                    "height": table_box["height"] + pad * 2,
                }
                page_w = clip["width"]
                page_h = clip["height"]
                page.screenshot(path=str(output_png), clip=clip)
            else:
                page.screenshot(path=str(output_png), full_page=True)

        finally:
            page.close()
            context.close()

        # bbox 정규화 (0~bbox_scale), clip 기준 오프셋 보정
        scale = self.bbox_scale
        clip_x = clip["x"] if table_box else 0
        clip_y = clip["y"] if table_box else 0
        ocr_info: list[dict] = []
        for cell in cells_info:
            text = cell.get("text", "").strip()
            cx = cell["x"] - clip_x
            cy = cell["y"] - clip_y
            x1 = max(0, int(cx * scale / page_w))
            y1 = max(0, int(cy * scale / page_h))
            x2 = min(scale, int((cx + cell["w"]) * scale / page_w))
            y2 = min(scale, int((cy + cell["h"]) * scale / page_h))
            if x2 > x1 and y2 > y1:
                ocr_info.append({"text": text, "bbox": [x1, y1, x2, y2]})

        return ocr_info


# ---------------------------------------------------------------------------
# 메인 생성 로직
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    started = time.time()
    rng = random.Random(args.seed)

    # 1. 인덱스 로드
    print(f"인덱스 로드: {args.index_path}")
    entries = load_index(args.index_path)
    print(f"  총 엔트리: {len(entries)}")

    # 2. GT 품질 필터 (선택)
    if args.gt_quality_filter:
        try:
            from data.gt_quality_filter import validate_gt_html
            before = len(entries)
            entries = [
                e for e in entries
                if not validate_gt_html(e.get("normalized_html", ""))
            ]
            print(f"  GT 품질 필터: {before - len(entries)}건 제외, 남은: {len(entries)}")
        except ImportError:
            print("  경고: gt_quality_filter 모듈을 찾을 수 없어 필터 건너뜀")

    # 3. 후보 풀 구성
    parent_pool = filter_parent_candidates(entries)
    child_pool = filter_child_candidates(entries)
    print(f"  부모 후보: {len(parent_pool)}건 (simple/medium, small/medium size)")
    print(f"  자식 후보: {len(child_pool)}건 (small size, 모든 복잡도)")

    if not parent_pool:
        raise ValueError("부모 후보 풀이 비어있습니다. 필터 조건을 확인하세요.")
    if not child_pool:
        raise ValueError("자식 후보 풀이 비어있습니다. 필터 조건을 확인하세요.")

    # 4. 출력 디렉토리 준비
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "nested_synthetic.jsonl"

    # 5. dedup 모드 설정 + 기존 생성 데이터 로드
    use_html_dedup = _use_html_dedup(args.dedup_mode)
    use_source_pair_dedup = _use_source_pair_dedup(args.dedup_mode)

    exclude_paths_raw = [Path(p) for p in (args.exclude_jsonl or [])]
    # 입력 중복 경로 제거(순서 유지)
    exclude_paths: list[Path] = list(dict.fromkeys(exclude_paths_raw))
    existing_seen_html: set[str] = set()
    existing_seen_source_pairs: set[tuple[str, str]] = set()
    existing_stats = {
        "paths": [],
        "rows": 0,
        "invalid_json_lines": 0,
        "missing_html_rows": 0,
        "missing_source_pair_rows": 0,
        "loaded_html_keys": 0,
        "loaded_source_pair_keys": 0,
        "loaded_keys": 0,
    }
    if exclude_paths:
        print(f"\n기존 생성 데이터 로드 (중복 제외 기준: {args.dedup_mode})")
        (
            existing_seen_html,
            existing_seen_source_pairs,
            existing_stats,
        ) = load_existing_dedup_keys(
            exclude_paths,
            dedup_mode=args.dedup_mode,
        )
        print(f"  입력 파일: {len(existing_stats['paths'])}개")
        print(f"  읽은 행 수: {existing_stats['rows']}")
        if use_html_dedup:
            print(f"  유효 HTML 키 수: {existing_stats['loaded_html_keys']}")
        if use_source_pair_dedup:
            print(f"  유효 source pair 수: {existing_stats['loaded_source_pair_keys']}")
        if existing_stats["invalid_json_lines"] > 0:
            print(f"  경고: JSON 파싱 실패 행: {existing_stats['invalid_json_lines']}")
        if use_html_dedup and existing_stats["missing_html_rows"] > 0:
            print(f"  참고: gt_html 누락 행: {existing_stats['missing_html_rows']}")
        if use_source_pair_dedup and existing_stats["missing_source_pair_rows"] > 0:
            print(
                "  참고: parent_source/child_source 누락 행: "
                f"{existing_stats['missing_source_pair_rows']}"
            )

    # 6. 렌더러 초기화
    print("\nPlaywright 브라우저 초기화...")
    renderer = NestedTableRenderer(
        viewport_width=args.viewport_width,
        device_scale_factor=args.device_scale_factor,
        bbox_scale=args.bbox_scale,
        wait_ms=args.wait_ms,
    )
    renderer.start()

    # 7. 생성 루프
    n_generated = 0
    n_failed = 0
    n_duplicate = 0
    n_duplicate_html = 0
    n_duplicate_source_pair = 0
    attempted = 0
    max_attempts = max(args.count, args.count * args.max_attempts_factor)
    # 기존 + 현재 생성 결과를 함께 추적하여 중복 방지
    seen_html_keys = set(existing_seen_html)
    seen_source_pair_keys = set(existing_seen_source_pairs)

    try:
        with open(output_jsonl, "w", encoding="utf-8") as out_f:
            pbar = tqdm(total=args.count, desc="nested 테이블 생성")
            while n_generated < args.count and attempted < max_attempts:
                attempted += 1

                parent_entry = rng.choice(parent_pool)
                child_entry = rng.choice(child_pool)

                parent_html = parent_entry.get("normalized_html", "")
                child_html = child_entry.get("normalized_html", "")
                parent_source = Path(parent_entry.get("image_path", "")).name
                child_source = Path(child_entry.get("image_path", "")).name

                source_pair_key = _build_source_pair_key(parent_source, child_source)
                if use_source_pair_dedup:
                    if not source_pair_key:
                        tqdm.write(
                            "  [SKIP] source pair 키 생성 실패 "
                            f"(attempt={attempted})"
                        )
                        n_failed += 1
                        continue
                    if source_pair_key in seen_source_pair_keys:
                        n_duplicate += 1
                        n_duplicate_source_pair += 1
                        continue

                # Nested HTML 생성
                try:
                    nested_html, n_inserted = create_nested_html(
                        parent_html,
                        child_html,
                        n_cells=args.n_child_per_parent,
                        min_parent_text_len=args.min_parent_text_len,
                        rng=rng,
                    )
                except Exception as e:
                    tqdm.write(f"  [SKIP] nested HTML 생성 실패 (attempt={attempted}): {e}")
                    n_failed += 1
                    continue

                if n_inserted == 0:
                    tqdm.write(
                        "  [SKIP] 조건 충족 td 셀 없음 "
                        f"(span 없음 + 텍스트 {args.min_parent_text_len}자 이상, "
                        f"attempt={attempted})"
                    )
                    n_failed += 1
                    continue

                html_dedup_key = ""
                if use_html_dedup:
                    html_dedup_key = _build_html_dedup_key(nested_html)
                    if not html_dedup_key:
                        tqdm.write(f"  [SKIP] html dedup 키 생성 실패 (attempt={attempted})")
                        n_failed += 1
                        continue
                    if html_dedup_key in seen_html_keys:
                        n_duplicate += 1
                        n_duplicate_html += 1
                        continue

                # 이미지 파일 경로
                img_filename = f"nested_{n_generated:06d}.png"
                img_path = images_dir / img_filename

                # 렌더링 + OCR bbox 추출
                try:
                    ocr_info = renderer.render(nested_html, img_path, rng)
                except Exception as e:
                    tqdm.write(f"  [SKIP] 렌더링 실패 (attempt={attempted}): {e}")
                    n_failed += 1
                    continue

                # Slim 포맷 레코드 생성
                record: dict = {
                    "image_path": f"images/{img_filename}",
                    "gt_html": nested_html,
                    "thinking": "",  # nested 테이블 thinking은 별도 생성 가능
                    "complexity": "nested",
                    "prompt_style": args.prompt_style,
                    "parent_source": parent_source,
                    "child_source": child_source,
                    "n_child_cells": n_inserted,
                    "ocr_info": ocr_info,
                    "bbox_scale": args.bbox_scale,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_generated += 1
                if use_html_dedup:
                    seen_html_keys.add(html_dedup_key)
                if use_source_pair_dedup and source_pair_key:
                    seen_source_pair_keys.add(source_pair_key)
                pbar.update(1)

            pbar.close()

    finally:
        renderer.stop()

    max_attempts_reached = n_generated < args.count and attempted >= max_attempts

    # 8. 요약 리포트
    elapsed = round(time.time() - started, 2)
    report = {
        "requested_count": args.count,
        "generated": n_generated,
        "failed": n_failed,
        "duplicate_skipped": n_duplicate,
        "duplicate_skipped_html": n_duplicate_html,
        "duplicate_skipped_source_pair": n_duplicate_source_pair,
        "attempted": attempted,
        "max_attempts": max_attempts,
        "max_attempts_reached": max_attempts_reached,
        "parent_pool_size": len(parent_pool),
        "child_pool_size": len(child_pool),
        "dedup_mode": args.dedup_mode,
        "exclude_jsonl_paths": existing_stats["paths"],
        "excluded_existing_keys": existing_stats["loaded_keys"],
        "excluded_existing_html_keys": existing_stats["loaded_html_keys"],
        "excluded_existing_source_pair_keys": existing_stats["loaded_source_pair_keys"],
        "exclude_rows_read": existing_stats["rows"],
        "exclude_invalid_json_lines": existing_stats["invalid_json_lines"],
        "exclude_missing_html_rows": existing_stats["missing_html_rows"],
        "exclude_missing_source_pair_rows": existing_stats["missing_source_pair_rows"],
        "n_child_per_parent": args.n_child_per_parent,
        "min_parent_text_len": args.min_parent_text_len,
        "seed": args.seed,
        "bbox_scale": args.bbox_scale,
        "prompt_style": args.prompt_style,
        "elapsed_sec": elapsed,
    }
    report_path = output_dir / "nested_synthetic.report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== 생성 완료 ===")
    print(f"  생성: {n_generated}건")
    print(f"  실패: {n_failed}건")
    print(f"  중복 스킵: {n_duplicate}건")
    if n_duplicate_html > 0:
        print(f"    - html 기준: {n_duplicate_html}건")
    if n_duplicate_source_pair > 0:
        print(f"    - source pair 기준: {n_duplicate_source_pair}건")
    print(f"  시도 횟수: {attempted}/{max_attempts}")
    if max_attempts_reached:
        print("  경고: 최대 시도 횟수에 도달하여 요청 수량을 모두 채우지 못했습니다.")
    print(f"  출력: {output_jsonl}")
    print(f"  리포트: {report_path}")
    print(f"  소요: {elapsed}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Nested Table 데이터 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index_path",
        required=True,
        help="analyze_aihub.py가 생성한 인덱스 JSONL 경로",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="출력 디렉토리 (images/ 서브폴더에 PNG 저장, nested_synthetic.jsonl 생성)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="생성할 nested 테이블 수",
    )
    parser.add_argument(
        "--n_child_per_parent",
        type=int,
        default=1,
        help="부모 테이블에 삽입할 자식 테이블 수 (1~2 권장)",
    )
    parser.add_argument(
        "--min_parent_text_len",
        type=int,
        default=10,
        help="자식 테이블 삽입 대상 부모 셀의 최소 텍스트 길이 (공백 제외)",
    )
    parser.add_argument(
        "--dedup_mode",
        choices=("html", "source_pair", "html_or_source_pair"),
        default="html",
        help=(
            "중복 판별 기준. "
            "html=gt_html 기준, "
            "source_pair=parent_source+child_source 기준, "
            "html_or_source_pair=둘 중 하나라도 중복이면 제외"
        ),
    )
    parser.add_argument(
        "--exclude_jsonl",
        action="append",
        default=[],
        help=(
            "기존 생성 JSONL 경로(중복 제외용). "
            "여러 파일을 제외하려면 --exclude_jsonl를 여러 번 지정"
        ),
    )
    parser.add_argument(
        "--max_attempts_factor",
        type=int,
        default=30,
        help="최대 시도 배수 (max_attempts = count * max_attempts_factor)",
    )
    parser.add_argument(
        "--gt_quality_filter",
        action="store_true",
        help="GT HTML 구조적 품질 필터 활성화 (validate_gt_html 사용)",
    )
    parser.add_argument(
        "--prompt_style",
        default="chandra_table_with_ocr",
        help="프롬프트 스타일",
    )
    parser.add_argument(
        "--bbox_scale",
        type=int,
        default=1024,
        help="OCR bbox 정규화 스케일",
    )
    parser.add_argument(
        "--viewport_width",
        type=int,
        default=1400,
        help="Playwright 뷰포트 너비 (CSS 픽셀)",
    )
    parser.add_argument(
        "--device_scale_factor",
        type=float,
        default=2.0,
        help="Playwright device_scale_factor (retina: 2.0)",
    )
    parser.add_argument(
        "--wait_ms",
        type=int,
        default=300,
        help="렌더링 후 안정 대기 시간 (ms)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count는 양수여야 합니다")
    if args.n_child_per_parent < 1:
        raise ValueError("--n_child_per_parent는 1 이상이어야 합니다")
    if args.min_parent_text_len < 1:
        raise ValueError("--min_parent_text_len은 1 이상이어야 합니다")
    if args.bbox_scale <= 0:
        raise ValueError("--bbox_scale는 양수여야 합니다")
    if args.max_attempts_factor < 1:
        raise ValueError("--max_attempts_factor는 1 이상이어야 합니다")

    run(args)


if __name__ == "__main__":
    main()
