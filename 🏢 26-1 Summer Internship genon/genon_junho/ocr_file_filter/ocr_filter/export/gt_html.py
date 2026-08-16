"""`final_dataset.jsonl`(큐레이션 라벨, `{category,text,bbox}` 리스트) → SFT 학습 포맷(gt_html 문자열)
변환. 콘텐츠 조립(`<div data-bbox=... data-label=...><p>...</p></div>`) 자체는
`_vendor_chandra_convert.py`(jhshin 원본 스크립트를 읽기전용으로 복사)의 `_content_for`/`_div`를
그대로 재사용한다 — 기존 16,886건짜리 레퍼런스 학습셋과 같은 SFT 런에 섞이므로 포맷이 정확히
일치해야 한다(2026-07-20 확인).

이 모듈이 추가로 하는 일은 tier마다 다른 bbox 좌표계(Easy=target 자체 출력이라 이미 0-1000 정규화,
Medium/Hard=원본 이미지 픽셀 좌표)를 벤더 스크립트가 기대하는 0-1000 정수로 통일하는 것뿐이다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from ocr_filter.cmcv.normalize import normalize_category
from ocr_filter.export import _vendor_chandra_convert as _vendor

Image.MAX_IMAGE_PIXELS = None  # 200dpi 원본 스캔 중 1억+ 픽셀짜리가 있어 PIL 기본 밤체크 비활성화

# tier -> label 의 bbox 좌표계 (ocr_filter/cmcv/run.py 의 _BBOX_COORD_SYSTEM 과 동일한 개념).
# Easy: target(Chandra) 자체 출력, 이미 0-1000 정규화. Medium/Hard: 원본 이미지 픽셀 좌표.
_TIER_COORD_SYSTEM = {"Easy": "norm1000", "Medium": "pixel", "Hard": "pixel"}

# 2026-07-20 실측: hardcase_judge(397B) 파서가 최근에 정규화 버그를 갖고 있어서(고쳤음,
# ocr_filter/hardcase/parse.py 참고), category 가 정확히 소문자 "table"인 경우 원본
# <table>...</table> HTML 이 이미 태그가 벗겨진 채로 저장돼 있다(2,309개 요소, 1,494페이지,
# 전부 Hard tier) — 원본 모델 응답을 안 남겨둬서 복구 불가능. 이런 레코드는 이 export 단계에서
# 페이지 전체를 스킵한다(깨진 표를 "Table" 이라고 우기는 가짜 HTML 을 학습에 넣지 않기 위해).
_BROKEN_TABLE_CATEGORY = "table"


def normalize_bbox_to_1000(
    bbox: list[float], coord_system: str, img_w: int | None = None, img_h: int | None = None,
) -> list[int]:
    """bbox 를 0-1000 정규화 정수로. coord_system="pixel" 이면 img_w/img_h 필수."""
    if coord_system == "pixel":
        if not img_w or not img_h:
            raise ValueError("coord_system='pixel' 이면 img_w/img_h 필요")
        return _vendor.renorm_from_px(bbox, img_w, img_h, scale=1000)
    if coord_system == "norm1000":
        return [_vendor._clamp(round(float(v)), 0, 1000) for v in bbox[:4]]
    raise ValueError(f"알 수 없는 coord_system: {coord_system!r}")


def elements_to_gt_html(elements: list[dict]) -> str:
    """이미 0-1000 정규화된 bbox 를 가진 elements 를 Chandra div-HTML gt_html 문자열로 조립.
    _vendor_chandra_convert._content_for/_div 를 그대로 재사용(Table=원본 HTML 그대로,
    Picture=빈 문자열, 나머지=<p>{escape(strip_leading_md_heading(text))}</p>). category 는
    normalize_category() 로 표준 11종으로 정규화한다 — 학습 타겟의 레이블 어휘를 항상 고정된
    11종으로 맞춰야 기존 레퍼런스 학습셋과 같은 SFT 런에 섞일 수 있다(2026-07-20 확인, hardcase
    judge 출처 레코드의 13%가 비표준 레이블)."""
    divs = []
    for e in elements:
        bbox = e.get("bbox") or [0, 0, 1000, 1000]
        category = normalize_category(e.get("category") or "")
        content = _vendor._content_for(category, e.get("text") or "")
        divs.append(_vendor._div(bbox, category, content))
    return "\n".join(divs)


def build_gt_html_dataset(
    final_dataset_path: str | Path, unified_path: str | Path, out_path: str | Path,
    tiers: set[str] | None = None,
) -> dict:
    """`final_dataset_path`(resolved=True 인 것만) + `unified_path`(id->image_path 조인)로
    gt_html 학습 포맷 jsonl 을 `out_path`에 새로 쓴다(기존 파일 덮어쓰기 없음, 매 호출 새로 생성).
    `tiers`를 주면 그 tier(예: {"Medium", "Hard"})만 포함 — Easy 는 target 모델 자기 자신의
    출력이라 다시 학습시켜도 새로운 교정 신호가 없어서, SFT에는 보통 Medium/Hard만 쓴다
    (2026-07-20, 사용자 확인)."""
    id_to_image: dict[str, str] = {}
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            id_to_image[d["id"]] = d["image_path"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_written = n_skipped_unresolved = n_missing_image = n_bad_bbox = 0
    n_skipped_broken_table = n_skipped_wrong_tier = 0
    tier_counts: Counter = Counter()

    with open(final_dataset_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1

            if not d.get("resolved"):
                n_skipped_unresolved += 1
                continue

            if tiers is not None and d.get("tier") not in tiers:
                n_skipped_wrong_tier += 1
                continue

            label = d.get("label") or []
            if any(e.get("category") == _BROKEN_TABLE_CATEGORY for e in label):
                n_skipped_broken_table += 1
                continue

            image_path = id_to_image.get(d["id"])
            if not image_path:
                n_missing_image += 1
                continue

            tier = d.get("tier")
            coord_system = _TIER_COORD_SYSTEM.get(tier, "pixel")
            img_w = img_h = None
            if coord_system == "pixel":
                with Image.open(image_path) as im:
                    img_w, img_h = im.size

            normalized = []
            for e in label:
                bbox = e.get("bbox")
                if not bbox or len(bbox) != 4:
                    n_bad_bbox += 1
                    nb = [0, 0, 1000, 1000]
                else:
                    nb = normalize_bbox_to_1000(bbox, coord_system, img_w, img_h)
                normalized.append({**e, "bbox": nb})

            out_record = {
                "image_path": image_path,
                "gt_html": elements_to_gt_html(normalized),
                "prompt_style": "unified_layout",
                "bbox_scale": 1000,
                "output_format": "html",
                "ocr_info": [],
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            n_written += 1
            tier_counts[tier] += 1

    return {
        "n_total": n_total,
        "n_written": n_written,
        "n_skipped_unresolved": n_skipped_unresolved,
        "n_skipped_wrong_tier": n_skipped_wrong_tier,
        "n_skipped_broken_table": n_skipped_broken_table,
        "n_missing_image": n_missing_image,
        "n_bad_bbox": n_bad_bbox,
        "tier_counts": dict(tier_counts),
    }
