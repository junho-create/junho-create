"""
PaddleOCR 기반 OCR 추출기

테이블 이미지에서 텍스트와 바운딩박스를 추출한다.
sample_aihub_with_ocr.py에서 분리된 독립 모듈.

Usage:
    from utils.ocr_extractor import PaddleOCRExtractor, normalize_ocr_items

    extractor = PaddleOCRExtractor(lang="korean", bbox_scale=1024)
    ocr_items = extractor.extract("/path/to/image.jpg")
    normalized = normalize_ocr_items(ocr_items, keep_score=False)
"""

from __future__ import annotations

import inspect
from typing import Optional

from PIL import Image


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def _poly_to_bbox(poly) -> Optional[list[float]]:
    """다양한 폴리곤 형식을 [x0, y0, x1, y1] 바운딩박스로 변환한다."""
    if poly is None:
        return None

    # [x0, y0, x1, y1] 형태
    if isinstance(poly, (list, tuple)) and len(poly) == 4:
        if all(isinstance(v, (int, float)) for v in poly):
            x0, y0, x1, y1 = [float(v) for v in poly]
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    # [x1, y1, x2, y2, x3, y3, x4, y4] 형태
    if isinstance(poly, (list, tuple)) and len(poly) == 8:
        if all(isinstance(v, (int, float)) for v in poly):
            xs = [float(poly[i]) for i in (0, 2, 4, 6)]
            ys = [float(poly[i]) for i in (1, 3, 5, 7)]
            return [min(xs), min(ys), max(xs), max(ys)]

    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
    except Exception:
        return None
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _normalize_bbox(bbox: list[float], width: int, height: int, bbox_scale: int) -> list[int]:
    """픽셀 좌표를 정규화된 좌표(0~bbox_scale)로 변환한다."""
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = bbox
    nx0 = _clamp_int((x0 / width) * bbox_scale, 0, bbox_scale)
    ny0 = _clamp_int((y0 / height) * bbox_scale, 0, bbox_scale)
    nx1 = _clamp_int((x1 / width) * bbox_scale, 0, bbox_scale)
    ny1 = _clamp_int((y1 / height) * bbox_scale, 0, bbox_scale)
    if nx1 < nx0:
        nx0, nx1 = nx1, nx0
    if ny1 < ny0:
        ny0, ny1 = ny1, ny0
    return [nx0, ny0, nx1, ny1]


class PaddleOCRExtractor:
    """Lazy PaddleOCR wrapper."""

    def __init__(self, lang: str = "korean", bbox_scale: int = 1024, device: str = "",
                 ocr_version: str = ""):
        self.lang = lang
        self.bbox_scale = int(bbox_scale)
        self.device = str(device or "").strip()
        # PaddleOCR 은 lang 별로 기본 버전이 다르다 — "korean" 은 PP-OCRv5 로 떨어지지만
        # "en" 은 _PPOCRV6_LANGS 목록에 있어서 지정 안 하면 조용히 PP-OCRv6 로 올라간다
        # (실측: --ocr_lang en 만 줬더니 PP-OCRv6_medium_det/rec 가 로드됨). v5 를 원하면
        # 명시적으로 줘야 한다.
        self.ocr_version = str(ocr_version or "").strip()
        self.ocr = None
        self._init_error = None
        self.first_image_open_error = None
        self.first_predict_error = None
        self.first_unknown_result_preview = None

    def _init(self) -> bool:
        if self.ocr is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            self._init_error = e
            return False

        try:
            sig = inspect.signature(PaddleOCR.__init__)
            params = sig.parameters

            ocr_kwargs = {"lang": self.lang}
            if self.ocr_version and "ocr_version" in params:
                ocr_kwargs["ocr_version"] = self.ocr_version
            if "use_doc_orientation_classify" in params:
                ocr_kwargs["use_doc_orientation_classify"] = False
            if "use_doc_unwarping" in params:
                ocr_kwargs["use_doc_unwarping"] = False
            if "use_textline_orientation" in params:
                ocr_kwargs["use_textline_orientation"] = False
            if "use_angle_cls" in params:
                # PaddleOCR 2.x 호환
                ocr_kwargs["use_angle_cls"] = False
            if self.device:
                if "device" in params:
                    # PaddleOCR 3.x (paddlex backend)
                    ocr_kwargs["device"] = self.device
                elif "use_gpu" in params:
                    # PaddleOCR 2.x 호환
                    ocr_kwargs["use_gpu"] = self.device.startswith("gpu")

            self.ocr = PaddleOCR(**ocr_kwargs)
            return True
        except Exception as e:
            self._init_error = e
            return False

    @staticmethod
    def _first_non_empty(*candidates):
        """None이 아니고 길이가 0보다 큰 첫 번째 후보를 반환한다."""
        for value in candidates:
            if value is None:
                continue
            try:
                if len(value) == 0:
                    continue
            except TypeError:
                return value
            except Exception:
                return value
            return value
        return []

    def _extract_from_v3_result(self, result_item, width: int, height: int) -> list[dict]:
        rec_texts = self._first_non_empty(result_item.get("rec_texts"), result_item.get("texts"), [])
        rec_scores = self._first_non_empty(result_item.get("rec_scores"), result_item.get("scores"), [])
        rec_polys = self._first_non_empty(
            result_item.get("rec_polys"),
            result_item.get("dt_polys"),
            result_item.get("rec_boxes"),
            result_item.get("dt_boxes"),
            [],
        )

        ocr_items = []
        n = min(len(rec_texts), len(rec_polys))
        for i in range(n):
            text = str(rec_texts[i]).strip()
            if not text:
                continue
            score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            bbox = _poly_to_bbox(rec_polys[i])
            if bbox is None:
                continue
            ocr_items.append(
                {
                    "text": text,
                    "bbox": _normalize_bbox(bbox, width, height, self.bbox_scale),
                    "score": score,
                }
            )
        return ocr_items

    def _coerce_result_item_to_dict(self, item) -> Optional[dict]:
        """PaddleOCR 결과 item을 dict로 정규화한다."""
        if isinstance(item, dict):
            return item

        if hasattr(item, "res"):
            try:
                res = getattr(item, "res")
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

        if hasattr(item, "to_dict"):
            try:
                obj = item.to_dict()
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        out = {}
        for key in ("rec_texts", "rec_scores", "rec_polys", "dt_polys", "texts", "scores"):
            if hasattr(item, key):
                try:
                    out[key] = getattr(item, key)
                except Exception:
                    continue
        return out if out else None

    def _extract_from_legacy_result(self, legacy_items, width: int, height: int) -> list[dict]:
        """구형 PaddleOCR 포맷: [[poly, [text, score]], ...]"""
        if not isinstance(legacy_items, (list, tuple)):
            return []

        ocr_items = []
        for entry in legacy_items:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue

            poly = entry[0]
            payload = entry[1]
            if isinstance(payload, (list, tuple)):
                text = str(payload[0]).strip() if len(payload) > 0 else ""
                try:
                    score = float(payload[1]) if len(payload) > 1 else 0.0
                except Exception:
                    score = 0.0
            else:
                text = str(payload).strip()
                score = 0.0

            if not text:
                continue

            bbox = _poly_to_bbox(poly)
            if bbox is None:
                continue

            ocr_items.append(
                {
                    "text": text,
                    "bbox": _normalize_bbox(bbox, width, height, self.bbox_scale),
                    "score": score,
                }
            )
        return ocr_items

    def extract(self, image_path: str) -> list[dict]:
        """이미지에서 OCR 텍스트와 바운딩박스를 추출한다.

        Returns:
            [{"text": str, "bbox": [x0,y0,x1,y1], "score": float}, ...]
            읽기 순서(위→아래, 왼→오른)로 정렬됨.
        """
        if not self._init():
            raise RuntimeError(
                "PaddleOCR initialization failed. "
                "Install dependencies and ensure model download/local cache is available:\n"
                "  pip install paddleocr paddlepaddle\n"
                f"  detail: {self._init_error}"
            ) from self._init_error

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            if self.first_image_open_error is None:
                self.first_image_open_error = f"{type(e).__name__}: {e}"
            return []

        try:
            result = self.ocr.predict(
                image_path,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
            )
        except Exception as e:
            if self.first_predict_error is None:
                self.first_predict_error = f"{type(e).__name__}: {e}"
            return []

        if not result:
            return []

        first = result[0]
        normalized = self._coerce_result_item_to_dict(first)
        if normalized is not None:
            items = self._extract_from_v3_result(normalized, width, height)
        elif isinstance(first, (list, tuple)):
            items = self._extract_from_legacy_result(first, width, height)
        elif isinstance(result, (list, tuple)):
            items = self._extract_from_legacy_result(result, width, height)
        else:
            items = []

        if not items and self.first_unknown_result_preview is None:
            preview = repr(first)
            if len(preview) > 500:
                preview = preview[:500] + "...(truncated)"
            self.first_unknown_result_preview = preview

        # 읽기 순서 정렬
        items.sort(key=lambda x: (x["bbox"][1], x["bbox"][0], x["text"]))
        return items


def normalize_ocr_items(items: list[dict], keep_score: bool = False) -> list[dict]:
    """OCR 결과에서 필요한 필드만 추출한다."""
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row: dict = {}
        if "text" in item:
            row["text"] = item["text"]
        if "bbox" in item:
            row["bbox"] = item["bbox"]
        if keep_score and "score" in item:
            row["score"] = item["score"]
        if row:
            normalized.append(row)
    return normalized
