import requests
import os
from dotenv import load_dotenv
import time
import multiprocessing
from tqdm.contrib.concurrent import process_map
import argparse
import json
import re
import difflib
from html import unescape, escape
from html.parser import HTMLParser
from typing import Optional, Iterable, Any, List, Dict, Tuple
import fitz  # PyMuPDF
# import chandra.prompts as ch_prompt
import asyncio
import base64
from io import BytesIO
from PIL import Image
import tqdm
from pathlib import Path

from html_template import html_template
from table_prompt import prompt_user_with_ocr, prompt_correction
from paddleocr_extractor import PaddleOCRExtractor

import sys
sys.path.insert(0, "../../../doc_parser/") # 현재 doc_parser의 docling 폴더 참조
from preprocess_ext_text import DocumentProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_PATH = str(SCRIPT_DIR / "model_profiles.json")

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_dir", help="Path to the input directory containing pdf files.")
parser.add_argument("--output_dir", default="output", help="Path to the output directory.")
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('-c', '--cpu', type=int, default=0)
parser.add_argument('--postprocess', choices=['none', 'fuzzy', 'llm'], default='none',
                    help='후보정 방식 선택: none(없음), fuzzy(difflib 매칭), llm(LLM 기반 교정)')
parser.add_argument('--cache_dir', default="_cache", help="Path to cache directory for intermediate data.")
parser.add_argument('--profiles', default=DEFAULT_PROFILES_PATH, help='Model profile JSON path')
parser.add_argument('--target', default=None, help='Profile target name')
parser.add_argument('--url', default=None, help='Override inference URL')
parser.add_argument('--model', default=None, help='Override inference model name')
parser.add_argument('--list_targets', action='store_true', help='List available targets and exit')
parser.add_argument('--correction_url', default=None, help='Override correction URL')
parser.add_argument('--correction_model', default=None, help='Override correction model name')
parser.add_argument('--max_tokens', type=int, default=None, help='Override generation max_tokens (default: 10000)')
args = parser.parse_args()


#-----------------------------------------------------------------
# LLM 설정
#-----------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 기본값: 환경변수로 오버라이드 가능
DEFAULT_URL = os.getenv("TABLE_VLM_URL", "http://192.168.75.173:8010/v1/chat/completions")
DEFAULT_MODEL = os.getenv("TABLE_VLM_MODEL", "")
DEFAULT_CORRECTION_URL = os.getenv(
    "TABLE_CORRECTION_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
DEFAULT_CORRECTION_MODEL = os.getenv(
    "TABLE_CORRECTION_MODEL",
    "mistralai/mistral-small-3.1-24b-instruct-2503",
)


def _requires_api_key(url: str) -> bool:
    return "openrouter.ai" in (url or "")


def _api_key_for_url(url: str) -> Optional[str]:
    if _requires_api_key(url):
        if not API_KEY:
            raise RuntimeError(
                f"OPENAI_API_KEY가 필요합니다. (url={url}) "
                "환경변수 OPENAI_API_KEY를 설정해 주세요."
            )
        return API_KEY
    return None


def _derive_models_endpoint(infer_url: str) -> str:
    u = (infer_url or "").strip().rstrip("/")
    if not u:
        return ""
    if "/v1/chat/completions" in u:
        return u.split("/v1/chat/completions", 1)[0] + "/v1/models"
    if u.endswith("/chat/completions"):
        return u.rsplit("/chat/completions", 1)[0] + "/models"
    if "/v1/" in u:
        return u.split("/v1/", 1)[0] + "/v1/models"
    if u.endswith("/v1"):
        return u + "/models"
    return u + "/v1/models"


def auto_discover_model_name(infer_url: str) -> Optional[str]:
    models_url = _derive_models_endpoint(infer_url)
    if not models_url:
        return None

    headers = {"Content-Type": "application/json"}
    api_key = _api_key_for_url(infer_url)
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    try:
        resp = requests.get(models_url, headers=headers, timeout=8)
        if resp.status_code != 200:
            print(f"[config] Warning: failed to query models endpoint ({resp.status_code}): {models_url}")
            return None
        payload = resp.json()
    except Exception as e:
        print(f"[config] Warning: failed to auto-discover model from {models_url} ({e})")
        return None

    data = payload.get("data")
    if not isinstance(data, list):
        return None

    model_ids = [
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id").strip()
    ]
    if not model_ids:
        return None

    preferred = [m for m in model_ids if m.lower() not in {"default", "base"}]
    return preferred[0] if preferred else model_ids[0]


def load_model_profiles(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read profiles file: {path} ({e})") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid profile format: top-level must be object ({path})")
    return data


def resolve_llm_config(parsed_args) -> tuple[str, str, str, str]:
    profiles = load_model_profiles(parsed_args.profiles)

    if parsed_args.list_targets:
        print("Available targets:")
        if not profiles:
            print("  (none)")
        else:
            for name, cfg in sorted(profiles.items()):
                url = cfg.get("url", "")
                model = cfg.get("model", "")
                print(f"  - {name}: model={model}, url={url}")
        raise SystemExit(0)

    target_cfg = {}
    if parsed_args.target:
        if parsed_args.target not in profiles:
            raise RuntimeError(
                f"Unknown target '{parsed_args.target}'. "
                f"Check --profiles file: {parsed_args.profiles}"
            )
        target_cfg = profiles[parsed_args.target]
        if not isinstance(target_cfg, dict):
            raise RuntimeError(f"Invalid target config type for '{parsed_args.target}'")

    url = parsed_args.url or target_cfg.get("url") or DEFAULT_URL
    model = parsed_args.model or target_cfg.get("model") or DEFAULT_MODEL
    correction_url = (
        parsed_args.correction_url
        or target_cfg.get("correction_url")
        or DEFAULT_CORRECTION_URL
    )
    correction_model = (
        parsed_args.correction_model
        or target_cfg.get("correction_model")
        or DEFAULT_CORRECTION_MODEL
    )

    if not url:
        raise RuntimeError("Inference URL is empty. Provide --url or set TABLE_VLM_URL.")
    if not model:
        discovered = auto_discover_model_name(url)
        if discovered:
            model = discovered
            print(f"[config] Auto-discovered inference model: {model}")
        else:
            raise RuntimeError(
                "Inference model is empty and auto-discovery failed. "
                "Provide --model, set target model in profiles, or set TABLE_VLM_MODEL."
            )

    return url, model, correction_url, correction_model


def get_file_list(input_dir, extension=".pdf"):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith(extension):
                files.append(os.path.join(root, filename))
    files=sorted(files)  # 정렬
    return files


def print_env (
    system_cpu_count,
    used_cpu_count
    ):
    print('--------------------------------------------------------')
    print('chandra, qwen3 test')
    print('- processing info')
    print('  - system cpu count: ', system_cpu_count)
    print('  - used cpu count: ', used_cpu_count)
    print('--------------------------------------------------------')


def get_png_params_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "png_params_cache.json")


def build_png_input_signature(png_files: List[str]) -> Dict[str, Any]:
    files_meta = []
    for file_path in png_files:
        stat = os.stat(file_path)
        files_meta.append(
            {
                "path": os.path.abspath(file_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "count": len(files_meta),
        "files": files_meta,
    }


def load_cached_png_payload(cache_path: str, signature: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"[cache] Failed to load cache file: {cache_path} ({e})")
        return None

    if cache_data.get("signature") != signature:
        print("[cache] PNG input changed. Rebuilding params cache.")
        return None

    payload = cache_data.get("payload")
    if not isinstance(payload, list):
        print("[cache] Invalid cache payload format. Rebuilding params cache.")
        return None

    print(f"[cache] Loaded PNG params payload from {cache_path} (items={len(payload)})")
    return payload


def save_cached_png_payload(cache_path: str, signature: Dict[str, Any], payload: List[Dict[str, Any]]) -> None:
    cache_data = {
        "version": 1,
        "signature": signature,
        "payload": payload,
        "saved_at": int(time.time()),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"[cache] Saved PNG params payload to {cache_path} (items={len(payload)})")


def build_params_list_from_png_payload(
    payload: List[Dict[str, Any]],
    llm_caller: Any,
    post_data_dir: Optional[str],
    correction_caller: Optional[Any],
    default_zoom: float,
) -> List[List[Any]]:
    params_list = []
    for item in payload:
        file_path = item["file"]
        if not os.path.exists(file_path):
            print(f"[cache] Skip missing file in cached payload: {file_path}")
            continue

        params = []
        params.append(llm_caller)
        params.append(file_path)
        params.append(item.get("text_info_list", []))
        params.append(item.get("zoom", default_zoom))
        params.append(item.get("index", 0))
        params.append(post_data_dir)
        params.append(correction_caller)
        params_list.append(params)
    return params_list


# llm 호출을 관리하는 class
class LLMCaller:

    def __init__(self, url: str, model: str, api_key: Optional[str], max_tokens: int = 10000):
        self.url = url
        self.model = model
        self.api_key = api_key

        if self.api_key:
            self.headers = {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            }
        else:
            self.headers = {
                "Content-Type": "application/json",
            }

        self.timeout = 120
        self.temperature = 0.0
        self.max_tokens = max_tokens

    def _gen_body(self, system_prompt, user_prompt, raw_data_list, image_base64_list: Optional[list[str]] = None):
        content = []

        if image_base64_list:
            for b64 in image_base64_list:
                if not b64:
                    continue
                # 혹시 data URL 형태로 들어온 경우도 처리
                if b64.startswith("data:image/"):
                    url_str = b64
                else:
                    url_str = f"data:image/png;base64,{b64}"

                content.append({
                    "type": "image_url",
                    "image_url": {"url": url_str}
                })

        if raw_data_list:
            prompt = user_prompt.format(**raw_data_list)
        else:
            prompt = user_prompt

        content.append({"type": "text", "text": prompt})

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        return data

    def _get_json_string(self, text):
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text)
        if match:
            text = match.group(1)
        return text

    def _get_html_string(self, text):
        html_pattern = r'```html\s*([\s\S]*?)\s*```'
        match = re.search(html_pattern, text)
        if match:
            text = match.group(1)
        return text

    def __call__(self, system_prompt, user_prompt, raw_data_list, image_base64_list: Optional[list[str]] = None, save_debug_file = None):

        data = self._gen_body(system_prompt, user_prompt, raw_data_list, image_base64_list)

        try:
            response = requests.post(self.url, headers=self.headers, json=data, timeout=self.timeout)
            # print(f"AI Response Status Code: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code} with message: {response.text}")
                return ""

            response_text = response.json()["choices"][0]["message"]["content"]

            if save_debug_file:
                save_data = {
                    "request": data,
                    "response": response_text
                }
                with open(save_debug_file, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=4)

            response_text = self._get_html_string(response_text)
            return response_text
        except Exception as e:
            print(f"Exception during AI call: {e}")
            return ""


class _TextElementCollector(HTMLParser):
    """HTML에서 텍스트 포함 리프 요소의 텍스트를 수집하는 파서"""
    LEAF_TAGS = {'td', 'th', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div', 'caption'}

    def __init__(self):
        super().__init__()
        self.elements = []  # list of (text, start_pos, end_pos)
        self._tag_stack = []
        self._current_text = []
        self._current_start = None

    def handle_starttag(self, tag, attrs):
        if tag in self.LEAF_TAGS:
            self._tag_stack.append(tag)
            self._current_text = []
            self._current_start = self.getpos()

    def handle_endtag(self, tag):
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
            text = ''.join(self._current_text).strip()
            if text:
                self.elements.append(text)
            self._current_text = []
            self._current_start = None

    def handle_data(self, data):
        if self._tag_stack:
            self._current_text.append(data)


def _has_math_or_formula(text: str) -> bool:
    """수식/LaTeX 마커 포함 여부 검사"""
    markers = ['\\frac', '\\sum', '\\int', '\\alpha', '\\beta', '\\gamma',
               '\\delta', '\\theta', '\\sigma', '\\mu', '\\lambda',
               '\\sqrt', '\\mathbb', '\\mathcal', '\\mathrm',
               '\\left', '\\right', '\\begin', '\\end',
               '$', '\\(', '\\)', '\\[', '\\]']
    text_lower = text.lower()
    return any(m in text_lower for m in markers)


def _find_best_ocr_match(llm_text: str, ocr_texts: list, used_indices: set,
                          threshold: float = 0.6, max_concat: int = 5) -> tuple:
    """LLM 텍스트에 대해 가장 유사한 OCR 텍스트를 찾음.

    Returns:
        (best_text, best_ratio, matched_indices) 또는 매칭 실패 시 (None, 0.0, set())
    """
    best_text = None
    best_ratio = 0.0
    best_indices = set()
    llm_norm = llm_text.strip()

    # 단일 OCR 항목 매칭
    for i, ocr_text in enumerate(ocr_texts):
        if i in used_indices:
            continue
        ratio = difflib.SequenceMatcher(None, llm_norm, ocr_text.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = ocr_text.strip()
            best_indices = {i}

    # 연속 OCR 항목 결합 매칭 (최대 max_concat개)
    for length in range(2, max_concat + 1):
        for start in range(len(ocr_texts) - length + 1):
            indices = set(range(start, start + length))
            if indices & used_indices:
                continue
            combined = ' '.join(ocr_texts[j].strip() for j in range(start, start + length))
            ratio = difflib.SequenceMatcher(None, llm_norm, combined).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = combined
                best_indices = indices

    if best_ratio >= threshold:
        return best_text, best_ratio, best_indices
    return None, 0.0, set()


def postprocess_with_ocr(html_text: str, ocr_texts: list, threshold: float = 0.6) -> str:
    """LLM 출력 HTML을 OCR 텍스트로 후보정.

    Args:
        html_text: LLM이 생성한 HTML 문자열
        ocr_texts: OCR에서 추출한 텍스트 리스트
        threshold: 유사도 임계값 (기본 0.6)

    Returns:
        후보정된 HTML 문자열
    """
    if not ocr_texts:
        return html_text

    # HTML에서 텍스트 요소 수집
    collector = _TextElementCollector()
    try:
        collector.feed(html_text)
    except Exception:
        return html_text

    llm_texts = collector.elements
    if not llm_texts:
        return html_text

    used_ocr_indices = set()
    replacements = []  # (llm_text, ocr_text) 쌍

    # 긴 텍스트부터 먼저 매칭 (부분 문자열 충돌 방지)
    sorted_texts = sorted(llm_texts, key=len, reverse=True)

    for llm_text in sorted_texts:
        if len(llm_text) <= 1:
            continue
        if _has_math_or_formula(llm_text):
            continue

        best_text, best_ratio, matched_indices = _find_best_ocr_match(
            llm_text, ocr_texts, used_ocr_indices, threshold
        )
        if best_text is not None:
            used_ocr_indices |= matched_indices
            if best_text != llm_text:
                replacements.append((llm_text, best_text))

    # HTML 내 텍스트 교체 (긴 것부터 교체하여 부분 매칭 방지)
    result = html_text
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for old_text, new_text in replacements:
        escaped_old = escape(old_text)
        escaped_new = escape(new_text)
        # 원본 텍스트로 교체 시도, 실패 시 escape 버전으로 시도
        if old_text in result:
            result = result.replace(old_text, new_text, 1)
        elif escaped_old in result:
            result = result.replace(escaped_old, escaped_new, 1)

    return result


def postprocess_with_llm(correction_caller, html_text: str, ocr_texts: list) -> str:
    """LLM을 사용하여 HTML 텍스트를 OCR 텍스트로 후보정.

    Args:
        correction_caller: 후보정용 LLMCaller 인스턴스
        html_text: VLM이 생성한 HTML 문자열
        ocr_texts: OCR에서 추출한 텍스트 리스트

    Returns:
        후보정된 HTML 문자열
    """
    if not ocr_texts:
        return html_text

    ocr_texts_str = "\n".join(f"- {text}" for text in ocr_texts)
    raw_data_list = {"original_html": html_text, "ocr_texts": ocr_texts_str}

    corrected = correction_caller("", prompt_correction, raw_data_list)
    if corrected:
        return corrected
    return html_text


def infer(params):
    llm_caller = params[0]
    file = params[1]
    text_info_list = params[2]
    zoom = params[3]
    index = params[4]
    post_data_dir = params[5]
    correction_caller = params[6]


    page_width = 0
    page_height = 0

    # 확장자별 처리
    ext = os.path.splitext(file)[1].lower()
    if ext == ".png":
        with Image.open(file) as img:
            page_width, page_height = img.size
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    elif ext == ".pdf":
        doc = fitz.open(file)
        page = doc.load_page(index)

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # base64 인코딩
        img_bytes = pix.tobytes("png")  # 파일 저장 없이 bytes
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        page_width = page.rect.width
        page_height = page.rect.height

    save_debug_file = None
    if post_data_dir:
        save_debug_file = f"{os.path.basename(file).replace('.pdf', '')}_page_{index}.json"
        save_debug_file = os.path.join(post_data_dir, save_debug_file)

    page_text_info = []
    raw_data_list = {}
    raw_data_list["bbox_scale"] = "1024"
    if text_info_list:
        for text_info in text_info_list:
            if text_info["page_no"] == index + 1:
                page_text_info = text_info["text_info"]
                break

        if page_text_info:
            # nomalize bbox to 0-1024
            for item in page_text_info:
                bbox = item["bbox"]
                x0 = int(bbox[0] * 1024 / page_width)
                y0 = int(bbox[1] * 1024 / page_height)
                x1 = int(bbox[2] * 1024 / page_width)
                y1 = int(bbox[3] * 1024 / page_height)
                item["bbox"] = [x0, y0, x1, y1]

        raw_data_list["ocr_info"] = page_text_info
    else:
        raw_data_list["ocr_info"] = []

    # LLM 호출: 이미지만 전송
    # raw_data_list = None
    response_text = llm_caller("", prompt_user_with_ocr, raw_data_list, [img_base64], save_debug_file=save_debug_file)

    # 후보정
    if response_text and page_text_info:
        ocr_texts = [item["text"] for item in page_text_info]
        if args.postprocess == "fuzzy":
            response_text = postprocess_with_ocr(response_text, ocr_texts)
        elif args.postprocess == "llm":
            response_text = postprocess_with_llm(correction_caller, response_text, ocr_texts)

    return response_text, file


def save_result(input_dir, output_dir: str, result_list: Any, one_html: bool = False):

    # input dir의 최하위 디렉토리명으로 output_filename 설정
    lowest_dir_name = os.path.basename(os.path.normpath(input_dir))
    output_filename = os.path.join(output_dir, lowest_dir_name + ".json")

    tatal_result = []
    for i, result in enumerate(result_list):
        raw_response, file_path = result
        # print(raw_response)

        # JSON 객체가 제대로 파싱되지 않으면 텍스트 전체를 raw로 저장
        record = {
            "file_path": file_path,
            "raw_response": raw_response,
        }

        tatal_result.append(record)

    # 동일한 file_path끼리 묶기
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for record in tatal_result:
        grouped_results[record['file_path']].append(record['raw_response'])
    tatal_result = []
    for file_path, responses in grouped_results.items():
        # combined_response = "\n".join(responses)
        record = {
            "file_path": file_path,
            "raw_response": responses,
        }
        tatal_result.append(record)


    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(tatal_result, f, ensure_ascii=False, indent=4)

    txt_filename = output_filename.replace(".json", ".txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        for record in tatal_result:
            f.write(f"File Path: {record['file_path']}\n")
            for page_idx, raw_response in enumerate(record['raw_response']):
                f.write(f"--- Page {page_idx} ---\n")
                f.write(raw_response + "\n")
            f.write("-" * 80 + "\n")

    # 파일별로 html 저장
    if one_html:
        all_html_contents = ""
        for record in tatal_result:
            file_path = record['file_path']
            raw_responses = record['raw_response']

            html_contents = ""
            for page_idx, raw_response in enumerate(raw_responses):
                html_contents += f"<h5>Page {page_idx}</h5>\n"
                html_contents += raw_response + "\n"
                html_contents += '<hr class="hr"/>\n'
            all_html_contents += f"<h2>File: {os.path.basename(file_path)}</h2>\n"
            all_html_contents += html_contents + "\n"
            all_html_contents += '<hr class="hr"/>\n'

        html_output = html_template.format(content=all_html_contents)
        all_html_filename = os.path.join(output_dir, lowest_dir_name + "_all.html")
        with open(all_html_filename, "w", encoding="utf-8") as f:
            f.write(html_output)

    else:
        for record in tatal_result:
            file_path = record['file_path']
            raw_responses = record['raw_response']

            html_contents = ""
            for page_idx, raw_response in enumerate(raw_responses):
                html_contents += f"<h5>Page {page_idx}</h5>\n"
                html_contents += raw_response + "\n"
                html_contents += '<hr class="hr"/>\n'

            html_output = html_template.format(content=html_contents)
            # html_output = html_template.replace("{content}", html_contents)

            # pdf, png 파일명 기준으로 html 저장
            html_filename = file_path.replace(".pdf", ".html").replace(".png", ".html")

            html_path = os.path.join(output_dir, os.path.basename(html_filename))
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_output)

doc_processor = DocumentProcessor()
async def process_document(doc_processor, input_path, **kwargs):
    print(input_path)  # 파일 경로 출력
    vectors = await doc_processor(input_path, **kwargs)
    return vectors

def main():

    system_cpu_count = multiprocessing.cpu_count()
    used_cpu_count = int(system_cpu_count * 1)
    if args.cpu > 0:
        used_cpu_count = min (args.cpu, used_cpu_count)

    print_env(system_cpu_count, used_cpu_count)

    os.makedirs(args.output_dir, exist_ok=True)

    zoom = 2.0  # 해상도(2.0이면 대략 144dpi 수준)

    post_data_dir = None
    if args.debug:
        post_data_dir = os.path.join(args.output_dir, "post_data")
        os.makedirs(post_data_dir, exist_ok=True)

    infer_url, infer_model, correction_url, correction_model = resolve_llm_config(args)
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else 10000
    print("- llm endpoint")
    print(f"  - target: {args.target if args.target else '(none)'}")
    print(f"  - url:    {infer_url}")
    print(f"  - model:  {infer_model}")
    print(f"  - max_tokens: {max_tokens}")
    if args.postprocess == "llm":
        print(f"  - correction_url:   {correction_url}")
        print(f"  - correction_model: {correction_model}")

    llm_caller = LLMCaller(infer_url, infer_model, _api_key_for_url(infer_url), max_tokens=max_tokens)
    correction_caller = (
        LLMCaller(correction_url, correction_model, _api_key_for_url(correction_url), max_tokens=max_tokens)
        if args.postprocess == "llm"
        else None
    )

    # input_dir 내의 모든 json 파일을 읽어서 contents 리스트 생성
    pdf_files = get_file_list(args.input_dir, extension=".pdf")
    png_files = get_file_list(args.input_dir, extension=".png")
    all_files = pdf_files + png_files

    begin = time.time()

    png_cache_path = get_png_params_cache_path(args.cache_dir)
    png_signature = build_png_input_signature(png_files)
    png_payload = load_cached_png_payload(png_cache_path, png_signature)

    if png_payload is None:
        png_payload = []
        if png_files:
            ocr_extractor = PaddleOCRExtractor()
            print("Extracting OCR info from PNG files...")
            for png_file in tqdm.tqdm(png_files):
                ocr_info = ocr_extractor.extract(png_file)
                text_info_list = [{"page_no": 1, "text_info": ocr_info}]
                png_payload.append(
                    {
                        "file": png_file,
                        "text_info_list": text_info_list,
                        "zoom": zoom,
                        "index": 0,
                    }
                )
            save_cached_png_payload(png_cache_path, png_signature, png_payload)

    png_params_list = build_params_list_from_png_payload(
        png_payload,
        llm_caller=llm_caller,
        post_data_dir=post_data_dir,
        correction_caller=correction_caller,
        default_zoom=zoom,
    )

    if png_params_list:
        print(f"Processing PNG files with LLM... {infer_model}")
        result_list = process_map(infer, png_params_list, max_workers=used_cpu_count, chunksize=1)
        save_result(args.input_dir, args.output_dir, result_list, one_html=True)

    # for pdf_file in pdf_files:
    for file in pdf_files:

        text_info_list = []
        text_info_list = asyncio.run(process_document(doc_processor, file))

        # pdf의 페이지수 가져오기
        doc = fitz.open(file)

        # page_count = min(doc.page_count, 20)  # 최대 20페이지만 처리
        page_count = doc.page_count

        params_list = []
        for i in range(page_count):
            params = []
            params.append(llm_caller)
            params.append(file)
            params.append(text_info_list)
            params.append(zoom)
            params.append(i)
            params.append(post_data_dir)
            params.append(correction_caller)
            params_list.append(params)

        result_list = process_map(infer, params_list, max_workers=used_cpu_count, chunksize=1)

        save_result(args.input_dir, args.output_dir, result_list)

    end = time.time()
    print(f"Total processing time: {end - begin:.2f} seconds")


if __name__ == '__main__':
    main()
