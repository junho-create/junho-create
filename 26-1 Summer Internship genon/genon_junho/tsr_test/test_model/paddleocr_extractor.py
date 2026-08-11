from typing import Optional
from PIL import Image


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def _poly_to_bbox(poly) -> Optional[list[float]]:
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

    def __init__(self, lang: str = "korean", bbox_scale: int = 1024):
        self.lang = lang
        self.bbox_scale = int(bbox_scale)
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
            # Keep same OCR switches used in tsr_lable_tool/label_tool/auto_generator.py
            # to avoid loading unnecessary orientation/unwarping sub-pipelines.
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            return True
        except Exception as e:
            self._init_error = e
            return False

    @staticmethod
    def _first_non_empty(*candidates):
        """
        Return the first candidate that is not None and has length > 0.
        Avoids truth-value checks on numpy arrays (e.g., `arr or ...`).
        """
        for value in candidates:
            if value is None:
                continue
            try:
                if len(value) == 0:
                    continue
            except TypeError:
                # Scalar-like object: treat as present.
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
            # score는 소수점 3자리로 반올림
            score = round(score, 3)
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

        # paddlex result object (.res) 지원
        if hasattr(item, "res"):
            try:
                res = getattr(item, "res")
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

        # 일반 객체 -> to_dict 지원
        if hasattr(item, "to_dict"):
            try:
                obj = item.to_dict()
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 속성 기반 fallback
        out = {}
        for key in ("rec_texts", "rec_scores", "rec_polys", "dt_polys", "texts", "scores"):
            if hasattr(item, key):
                try:
                    out[key] = getattr(item, key)
                except Exception:
                    continue
        return out if out else None

    def _extract_from_legacy_result(self, legacy_items, width: int, height: int) -> list[dict]:
        """
        구형 PaddleOCR 포맷을 처리한다.
        예시: [[poly, [text, score]], ...]
        """
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
            # 일부 환경에서는 result 자체가 legacy 엔트리 리스트다.
            items = self._extract_from_legacy_result(result, width, height)
        else:
            items = []

        if not items and self.first_unknown_result_preview is None:
            preview = repr(first)
            if len(preview) > 500:
                preview = preview[:500] + "...(truncated)"
            self.first_unknown_result_preview = preview

        # Stable reading-order sort
        items.sort(key=lambda x: (x["bbox"][1], x["bbox"][0], x["text"]))
        return items
