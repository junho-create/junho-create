"""cmcv 결과 시각화: GT/target/dots.ocr/paddle bbox 오버레이 HTML 갤러리."""

from ocr_filter.report.gallery import build_html, pick_ids_by_tier
from ocr_filter.report.render import draw_boxes, to_data_uri

__all__ = ["build_html", "pick_ids_by_tier", "draw_boxes", "to_data_uri"]
