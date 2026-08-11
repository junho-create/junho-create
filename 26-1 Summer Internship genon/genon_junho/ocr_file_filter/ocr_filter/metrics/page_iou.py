"""PageIoU (Dolphin/PP-DocLayout 계열 논문에서 제안된 페이지 레이아웃 커버리지 지표).

박스 단위 IoU(mAP류)는 문서 레이아웃처럼 블록 경계가 모호한 경우 정성 평가와 어긋나는
경우가 잦다 — 문단 전체를 뭉뚱그린 박스가 줄 단위로 정확히 쪼갠 박스보다 오히려 recall이
높게 나오는 역설이 논문 Figure 4 예시. PageIoU는 개별 박스를 매칭하는 대신, 페이지를
격자로 나눠 **커버리지 맵(각 셀을 몇 개의 박스가 덮는가)** 을 만들고 그 교집합/합집합
비율을 본다 — 박스 분할 방식이 달라도 실제로 덮는 영역이 비슷하면 높은 점수가 나온다.

여기 구현은 논문 수식(픽셀 단위 min/max 합)을 grid×grid 격자로 근사한다(박스 개수가
페이지당 수십 개 수준이라 정확한 픽셀 래스터화 없이도 충분히 안정적).
좌표는 호출측에서 미리 [0,1] 정규화 좌표로 맞춰서 넘겨야 한다(target=0-1000, dots.ocr/
paddle=원본 픽셀이라 좌표계가 서로 달라서 — `cmcv/run.py` 의 `_normalized_boxes` 참고).
"""

from __future__ import annotations

import numpy as np


def _coverage_grid(boxes: list[list[float]], grid: int) -> np.ndarray:
    cover = np.zeros((grid, grid), dtype=np.int32)
    for bbox in boxes:
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        gx0 = min(grid, max(0, int(x0 * grid)))
        gx1 = min(grid, max(gx0 + 1, int(np.ceil(x1 * grid))))
        gy0 = min(grid, max(0, int(y0 * grid)))
        gy1 = min(grid, max(gy0 + 1, int(np.ceil(y1 * grid))))
        cover[gy0:gy1, gx0:gx1] += 1
    return cover


def page_iou_score(a_boxes: list[list[float]], b_boxes: list[list[float]], grid: int = 200) -> float:
    """a_boxes/b_boxes: [[x0,y0,x1,y1], ...] (0-1 정규화 좌표). 둘 다 박스가 없으면
    1.0(완전일치 취급, 다른 스코어러들과 같은 관례)."""
    a_valid = [b for b in a_boxes if b and len(b) == 4]
    b_valid = [b for b in b_boxes if b and len(b) == 4]
    if not a_valid and not b_valid:
        return 1.0

    a_cover = _coverage_grid(a_valid, grid)
    b_cover = _coverage_grid(b_valid, grid)
    inter = np.minimum(a_cover, b_cover).sum()
    union = np.maximum(a_cover, b_cover).sum()
    if union == 0:
        return 1.0
    return float(inter / union)
