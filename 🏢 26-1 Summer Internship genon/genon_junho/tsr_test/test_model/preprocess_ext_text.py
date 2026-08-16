from __future__ import annotations

import json
import os
from pathlib import Path
import logging
import io

from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple
import requests
import base64
from io import BytesIO

_log = logging.getLogger(__name__)

# docling imports

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
# from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    # OcrEngine,
    # PdfBackend,
    PdfPipelineOptions,
    TableFormerMode,
    PipelineOptions,
    PaddleOcrOptions,
    # EasyOcrOptions,
)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    FormatOption
)
from docling.datamodel.pipeline_options import DataEnrichmentOptions
from docling.utils.document_enrichment import enrich_document, check_document
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)
from docling_core.types import DoclingDocument

from pandas import DataFrame
import asyncio
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    DocumentOrigin,
    LevelNumber,
    ListItem,
    CodeItem,
    ContentLayer,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    DocItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem
)
from collections import Counter
import re
import json
import warnings
from typing import Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

try:
    import semchunk
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )

try:
    from genos_utils import upload_files
except ImportError:
    upload_files = None

# ============================================
#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

class DocumentProcessor:

    def __init__(self):
        '''
        initialize Document Converter
        '''
        self.ocr_endpoint = "http://192.168.73.172:48080/ocr"
        ocr_options = PaddleOcrOptions(
            force_full_page_ocr=False,
            lang=['korean'],
            ocr_endpoint=self.ocr_endpoint,
            text_score=0.3)

        self.page_chunk_counts = defaultdict(int)
        device = AcceleratorDevice.AUTO
        num_threads = 8
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
        # PDF 파이프라인 옵션 설정
        self.pipe_line_options = PdfPipelineOptions()
        self.pipe_line_options.generate_page_images = True
        self.pipe_line_options.generate_picture_images = False
        self.pipe_line_options.do_ocr = False
        self.pipe_line_options.ocr_options = ocr_options
        self.pipe_line_options.do_table_structure = False
        self.pipe_line_options.images_scale = 2
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipe_line_options.accelerator_options = accelerator_options

        # ocr 파이프라인 옵션
        self.ocr_pipe_line_options = PdfPipelineOptions()
        self.ocr_pipe_line_options = self.pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options.model_copy(deep=True)
        self.ocr_pipe_line_options.ocr_options.force_full_page_ocr = True

        # 기본 컨버터들 생성
        self._create_converters()

        # enrichment 옵션 설정
        self.enrichment_options = DataEnrichmentOptions(
            do_toc_enrichment=False,
            extract_metadata=True,
            toc_api_provider="custom",

            # Mistral-Small-3.1-24B-Instruct-2503, 운영망
            toc_api_base_url="https://genos.genon.ai:3443/api/gateway/rep/serving/502/v1/chat/completions",
            metadata_api_base_url="https://genos.genon.ai:3443/api/gateway/rep/serving/502/v1/chat/completions",
            toc_api_key="022653a3743849e299f19f19d323490b",
            metadata_api_key="022653a3743849e299f19f19d323490b",

            toc_model="model",
            metadata_model="model",

            toc_temperature=0.0,
            toc_top_p=0.00001,
            toc_seed=33,
            toc_max_tokens=1000
        )

    def _create_converters(self):
        """컨버터들을 생성하는 헬퍼 메서드"""
        self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipe_line_options,
                        backend=DoclingParseV4DocumentBackend
                    ),
                }
            )
        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )
        self.ocr_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.ocr_pipe_line_options,
                        backend=DoclingParseV4DocumentBackend
                    ),
                }
            )
        self.ocr_second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.ocr_pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )

    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> ConversionResult:
        try:
            conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.second_converter.convert(file_path, raises_on_error=True)
        return conv_result

    def load_documents_with_docling_ocr(self, file_path: str, **kwargs: dict) -> ConversionResult:
        try:
            conv_result: ConversionResult = self.ocr_converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.ocr_second_converter.convert(file_path, raises_on_error=True)
        return conv_result

    def load_documents(self, file_path: str, **kwargs) -> ConversionResult:
        return self.load_documents_with_docling(file_path, **kwargs)

    def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
        """텍스트에 GLYPH 항목이 있는지 확인하는 메서드"""
        if not text:
            return False

        # GLYPH 항목이 있는지 정규식으로 확인
        matches = re.findall(r'GLYPH\w*', text)
        if len(matches) >= threshold:
            # print(f"Text has glyphs. len(matches): {len(matches)}. ")
            return True

        return False

    def check_glyphs(self, document: DoclingDocument) -> bool:
        """문서에 글리프가 있는지 확인하는 메서드"""
        for item, level in document.iterate_items():
            if isinstance(item, TextItem) and hasattr(item, 'prov') and item.prov:
                page_no = item.prov[0].page_no
                # page_texts += item.text

                # GLYPH 항목이 있는지 확인. 정규식사용
                matches = re.findall(r'GLYPH\w*', item.text)
                if len(matches) > 10:
                    # print(f"Document has glyphs on page {page_no}. len(matches): {len(matches)}. ")
                    return True

        return False

    def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> List[Dict[str, Any]]:
        """
        글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR을 수행합니다.
        Args:
            document: DoclingDocument 객체
            pdf_path: PDF 파일 경로
        Returns:
            OCR이 완료된 문서의 DoclingDocument 객체
        """
        import fitz
        import base64
        import requests

        def post_ocr_bytes(img_bytes: bytes, timeout=60) -> dict:
            HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {"file": base64.b64encode(img_bytes).decode("ascii"), "fileType": 1, "visualize": False}
            r = requests.post(self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout)
            if not r.ok:
                # 진단에 도움되도록 본문 일부 출력
                raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text[:500]}")
            return r.json()

        def extract_ocr_fields(resp: dict):
            """
            resp: 위와 같은 OCR 응답 JSON(dict)
            return: (rec_texts, rec_scores, rec_boxes) — 모두 list
            """
            if resp is None:
                return [], [], []

            # 최상위 상태 체크
            if resp.get("errorCode") not in (0, None):
                return [], [], []

            ocr_results = (
                resp.get("result", {})
                    .get("ocrResults", [])
            )
            if not ocr_results:
                return [], [], []

            pruned = (
                ocr_results[0]
                .get("prunedResult", {})
            )
            if not pruned:
                return [], [], []

            rec_texts  = pruned.get("rec_texts", [])   # list[str]
            rec_scores = pruned.get("rec_scores", [])  # list[float]
            rec_boxes  = pruned.get("rec_boxes", [])   # list[[x1,y1,x2,y2]]

            # 길이 불일치 방어: 최소 길이에 맞춰 자르기
            n = min(len(rec_texts), len(rec_scores), len(rec_boxes))
            return rec_texts[:n], rec_scores[:n], rec_boxes[:n]

        try:
            doc = fitz.open(pdf_path)

            for table_idx, table_item in enumerate(document.tables):
                if not table_item.data or not table_item.data.table_cells:
                    continue

                b_ocr = False
                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if self.check_glyph_text(cell.text, threshold=1):
                        b_ocr = True
                        break

                if b_ocr is False:
                    # 글리프 깨진 텍스트가 없는 경우, OCR을 수행하지 않음
                    continue

                for cell_idx, cell in enumerate(table_item.data.table_cells):

                    # Provenance 정보에서 위치 정보 추출
                    if not table_item.prov:
                        continue

                    page_no = table_item.prov[0].page_no - 1
                    bbox = cell.bbox

                    page = doc.load_page(page_no)

                    # 셀의 바운딩 박스를 사용하여 이미지에서 해당 영역을 잘라냄
                    cell_bbox = fitz.Rect(
                        bbox.l, min(bbox.t, bbox.b),
                        bbox.r, max(bbox.t, bbox.b)
                    )

                    # bbox 높이 계산 (PDF 좌표계 단위)
                    bbox_height = cell_bbox.height

                    # 목표 픽셀 높이
                    target_height = 20

                    # zoom factor 계산
                    # (너무 작은 bbox일 경우 0으로 나누는 걸 방지)
                    zoom_factor = target_height / bbox_height if bbox_height > 0 else 1.0
                    zoom_factor = min(zoom_factor, 4.0)  # 최대 확대 비율 제한
                    zoom_factor = max(zoom_factor, 1)  # 최소 확대 비율 제한

                    # 페이지를 이미지로 렌더링
                    mat = fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=mat, clip=cell_bbox)
                    img_data = pix.tobytes("png")

                    result = post_ocr_bytes(img_data, timeout=60)
                    rec_texts, rec_scores, rec_boxes = extract_ocr_fields(result)

                    cell.text = ""
                    for t in rec_texts:
                        if len(cell.text) > 0:
                            cell.text += " "
                        cell.text += t if t else ""
        except Exception as e:
            print(f"OCR processing failed: {e}")
            pass

        return document

    def setup_logging(self, level_num: int):
        """
            5"DEBUG", 4"INFO", 3"WARNING", 2"ERROR", 1"CRITICAL", 0"NOLOG" 중 하나를 받아서 로깅 레벨을 설정하는 메서드
        """
        def get_level_name(level_num: int) -> str:
            level_map = {
                5: "DEBUG",
                4: "INFO",
                3: "WARNING",
                2: "ERROR",
                1: "CRITICAL",
                0: "NOLOG"
            }
            return level_map.get(level_num, "INFO")
        level_name = get_level_name(level_num)
        print(f"Setting log level to: {level_name}")

        if level_name == "NOLOG" or not hasattr(logging, level_name):
            logging.disable(logging.CRITICAL)  # 모든 로그 비활성화
            return

        level = getattr(logging, level_name.upper())

        # root logger 설정 (핸들러는 main에서만 설정)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]   # 콘솔 출력
        )

        # root logger level 적용
        logging.getLogger().setLevel(level)

    async def __call__(self, file_path: str, **kwargs: dict):
        self.setup_logging(kwargs.get('log_level', 4))

        _log.info(f"file_path: {file_path}")
        _log.info(f"kwargs: {kwargs}")

        conv_result: ConversionResult = self.load_documents(file_path, **kwargs)
        document: DoclingDocument = conv_result.document

        if not check_document(document, self.enrichment_options) or self.check_glyphs(document):
            # OCR이 필요하다고 판단되면 OCR 수행
            conv_result: ConversionResult = self.load_documents_with_docling_ocr(file_path, **kwargs)
            document: DoclingDocument = conv_result.document
        # 글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR 수행 (청크토큰 8k이상 발생 방지)
        document: DoclingDocument = self.ocr_all_table_cells(document, file_path)

        text_info_extractor = TextInfoExtractor()
        page_list = text_info_extractor(conv_result)
        return page_list


class TextInfoExtractor:

    def __init__(self):
        self.scale = 2.0  # 이미지 스케일 설정

    def _to_base64_image(self, image) -> str:
        """
        이미지를 base64 인코딩 문자열로 변환합니다.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    @staticmethod
    def save_base64_png(base64_str: str, output_path: str):
        # data:image/png;base64, 제거
        if base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_str)

        with open(output_path, "wb") as f:
            f.write(image_bytes)

    def get_text_from_item(self, item: DocItem) -> str:
        """DocItem에서 텍스트 추출"""
        # if isinstance(item, TableItem):
        #     return item.export_to_markdown(document)
        if hasattr(item, 'text') and item.text:
            return item.text
        # elif isinstance(item, PictureItem):
        #     text = ""
        #     for annotation in item.annotations:
        #         if hasattr(annotation, 'text'):
        #             text += annotation.text
        #     return text
        return ""

    def __call__(self, conv_result: ConversionResult):
        if not conv_result or not conv_result.document:
            return None

        # page 별 텍스트, 좌표 추출, 이미지도 추출
        page_list = []
        for page_no, page in enumerate(conv_result.pages):
            page_text_info = []
            for item, _level in conv_result.document.iterate_items():
                if not hasattr(item, 'prov') or not item.prov:
                    continue
                # 아이템이 해당 페이지에 있는지 확인
                if item.prov[0].page_no != page_no + 1:
                    continue
                item_text = self.get_text_from_item(item)
                if len(item_text.strip()) == 0:
                    continue
                prov = item.prov[0]
                bbox = prov.bbox.to_top_left_origin(page.size.height)

                page_text_info.append({
                    "text": item_text,
                    "bbox": [bbox.l, bbox.t, bbox.r, bbox.b],
                    "score": 0.99
                })

            page_img = page.get_image(scale=self.scale)

            page_list.append({"page_no": page_no + 1,
                            "text_info": page_text_info,
                            "image_base64": self._to_base64_image(page_img)})
        return page_list
