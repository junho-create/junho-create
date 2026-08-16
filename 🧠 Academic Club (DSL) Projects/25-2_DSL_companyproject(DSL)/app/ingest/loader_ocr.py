# -*- coding: utf-8 -*-
"""OCR API를 통한 PDF/이미지 텍스트 추출 모듈.

OCR API 응답을 TextBlock 리스트로 변환하여 기존 파이프라인과 호환되도록 함.

설정 방법:
1. .env 파일 생성: cp .env.example .env
2. .env 파일에 OCR_API_URL 설정
3. 또는 환경변수로 설정: export OCR_API_URL="..."
"""
import os
import tempfile
from pathlib import Path
from typing import List, Optional
import requests
from dotenv import load_dotenv
from app.ingest.loader_pdf import TextBlock

# .env 파일 로드 (프로젝트 루트에서)
env_path = Path(__file__).parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # .env 파일이 없어도 환경변수는 사용 가능
    load_dotenv()

# 설정 로드
# 우선순위: 1) 환경변수, 2) .env 파일, 3) None (에러 발생)
DEFAULT_OCR_URL = os.getenv("OCR_API_URL")
DEFAULT_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "60"))
MAX_PAGES_PER_OCR_REQUEST = 5  # 네이버 클로바 OCR API 최대 페이지 수


def call_ocr_api(
    file_path: str, 
    ocr_url: Optional[str] = None, 
    timeout: int = DEFAULT_TIMEOUT
) -> dict:
    """
    OCR API를 호출하여 파일에서 텍스트를 추출합니다.
    
    Args:
        file_path: 처리할 PDF/이미지 파일 경로
        ocr_url: OCR API 엔드포인트 URL (None이면 환경변수 또는 기본값 사용)
        timeout: 요청 타임아웃 (초)
    
    Returns:
        dict: OCR API 응답 JSON
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        requests.RequestException: API 호출 실패 시
        ValueError: 응답이 유효한 JSON이 아닌 경우
    """
    if ocr_url is None:
        ocr_url = DEFAULT_OCR_URL
    
    if not ocr_url:
        raise ValueError(
            "OCR API URL이 설정되지 않았습니다.\n"
            "설정 방법:\n"
            "1. .env 파일 생성: cp .env.example .env\n"
            "2. .env 파일에 OCR_API_URL=... 추가\n"
            "3. 또는 환경변수로 설정: export OCR_API_URL=...\n"
            "4. 또는 함수 호출 시 ocr_url 인자 제공"
        )
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # 파일 크기 확인
    file_size = file_path_obj.stat().st_size
    print(f"[DEBUG] OCR 요청 파일: {file_path_obj.name}, 크기: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # 파일 확장자에 따라 Content-Type 결정
    content_type = "application/pdf"
    if file_path_obj.suffix.lower() in [".jpg", ".jpeg"]:
        content_type = "image/jpeg"
    elif file_path_obj.suffix.lower() == ".png":
        content_type = "image/png"
    elif file_path_obj.suffix.lower() in [".tiff", ".tif"]:
        content_type = "image/tiff"
    
    # multipart/form-data로 파일 전송
    with open(file_path, "rb") as f:
        files = {
            "file": (
                file_path_obj.name, 
                f, 
                content_type
            )
        }
        try:
            response = requests.post(
                ocr_url, 
                files=files, 
                timeout=timeout
            )
            
            # HTTP 오류 시 상세 정보 로깅
            if not response.ok:
                error_detail = f"HTTP {response.status_code}"
                try:
                    error_body = response.text[:500]  # 최대 500자만
                    if error_body:
                        error_detail += f": {error_body}"
                except:
                    pass
                print(f"[ERROR] OCR API 오류 응답: {error_detail}")
            
            response.raise_for_status()  # HTTP 오류 시 예외 발생
            
            # JSON 파싱
            try:
                return response.json()
            except ValueError as e:
                raise ValueError(f"OCR API 응답이 유효한 JSON이 아닙니다: {e}")
        
        except requests.HTTPError as e:
            # HTTP 오류의 경우 응답 본문 포함
            error_msg = f"OCR API 호출 실패: {e}"
            try:
                if hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    if error_body:
                        error_msg += f"\n서버 응답: {error_body}"
            except:
                pass
            raise requests.RequestException(error_msg)
        except requests.RequestException as e:
            raise requests.RequestException(f"OCR API 호출 실패: {e}")


def ocr_response_to_blocks(ocr_json: dict) -> List[TextBlock]:
    """
    OCR API 응답을 TextBlock 리스트로 변환합니다.
    
    OCR 응답 구조:
    {
        "ok": true,
        "data": {
            "images": [{
                "fields": [{
                    "inferText": "텍스트",
                    "inferConfidence": 1,
                    "lineBreak": false/true
                }]
            }]
        }
    }
    
    Args:
        ocr_json: OCR API 응답 JSON
    
    Returns:
        List[TextBlock]: 변환된 텍스트 블록 리스트
    """
    blocks: List[TextBlock] = []
    
    if not isinstance(ocr_json, dict):
        return blocks
    
    # 응답이 성공인지 확인
    if not ocr_json.get("ok", False):
        return blocks
    
    # data.images 구조 확인
    data = ocr_json.get("data", {})
    if not isinstance(data, dict):
        return blocks
    
    images = data.get("images", [])
    if not isinstance(images, list):
        return blocks
    
    # 각 이미지(페이지) 처리
    for page_idx, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        
        fields = image.get("fields", [])
        if not isinstance(fields, list):
            continue
        
        # 페이지 번호 (1부터 시작)
        page_num = page_idx + 1
        
        # 필드를 순회하며 텍스트 라인 구성
        current_line_texts = []
        line_no = 1
        
        for field in fields:
            if not isinstance(field, dict):
                continue
            
            infer_text = field.get("inferText", "")
            if not infer_text:
                continue
            
            line_break = field.get("lineBreak", False)
            
            # 현재 라인에 텍스트 추가
            current_line_texts.append(infer_text)
            
            # 줄바꿈이면 블록 생성
            if line_break:
                line_text = " ".join(current_line_texts).strip()
                if line_text:  # 빈 줄은 스킵
                    blocks.append(TextBlock(
                        page=page_num,
                        line_no=line_no,
                        text=line_text
                    ))
                    line_no += 1
                current_line_texts = []
        
        # 마지막 라인 처리 (lineBreak가 없어서 끝나지 않은 경우)
        if current_line_texts:
            line_text = " ".join(current_line_texts).strip()
            if line_text:
                blocks.append(TextBlock(
                    page=page_num,
                    line_no=line_no,
                    text=line_text
                ))
    
    return blocks


def get_pdf_page_count(pdf_path: str) -> int:
    """
    PDF 파일의 페이지 수를 반환합니다.
    
    Args:
        pdf_path: PDF 파일 경로
    
    Returns:
        int: 페이지 수
    """
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)
    except ImportError:
        # PyPDF2가 없으면 pdfminer로 확인
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser
        
        with open(pdf_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            pages = list(PDFPage.create_pages(doc))
            return len(pages)
    except Exception as e:
        print(f"[WARN] PDF 페이지 수 확인 실패: {e}")
        return 0


def split_pdf_by_pages(
    pdf_path: str, 
    start_page: int, 
    end_page: int,
    output_path: Optional[str] = None
) -> str:
    """
    PDF 파일을 지정된 페이지 범위로 분할합니다.
    
    Args:
        pdf_path: 원본 PDF 파일 경로
        start_page: 시작 페이지 (1부터 시작)
        end_page: 종료 페이지 (포함)
        output_path: 출력 파일 경로 (None이면 임시 파일 생성)
    
    Returns:
        str: 분할된 PDF 파일 경로
    """
    import PyPDF2
    
    if output_path is None:
        # 임시 파일 생성
        temp_dir = tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".pdf", 
            delete=False, 
            dir=temp_dir
        )
        output_path = temp_file.name
        temp_file.close()
    
    with open(pdf_path, "rb") as input_file:
        pdf_reader = PyPDF2.PdfReader(input_file)
        pdf_writer = PyPDF2.PdfWriter()
        
        # 페이지 인덱스는 0부터 시작하므로 -1
        for page_num in range(start_page - 1, min(end_page, len(pdf_reader.pages))):
            pdf_writer.add_page(pdf_reader.pages[page_num])
        
        with open(output_path, "wb") as output_file:
            pdf_writer.write(output_file)
    
    return output_path


def load_pdf_via_ocr(
    pdf_path: str,
    ocr_url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_pages: int = MAX_PAGES_PER_OCR_REQUEST
) -> List[TextBlock]:
    """
    OCR API를 사용하여 PDF/이미지에서 텍스트 블록을 추출합니다.
    
    10페이지를 초과하는 PDF의 경우 자동으로 분할하여 처리합니다.
    
    Args:
        pdf_path: PDF/이미지 파일 경로
        ocr_url: OCR API 엔드포인트 URL (None이면 환경변수 또는 기본값 사용)
        timeout: 요청 타임아웃 (초)
        max_pages: OCR API 호출당 최대 페이지 수 (기본값: 10)
    
    Returns:
        List[TextBlock]: 추출된 텍스트 블록 리스트
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        requests.RequestException: API 호출 실패 시
        ValueError: 응답 파싱 실패 시
    """
    file_path_obj = Path(pdf_path)
    
    # PDF가 아닌 경우 (이미지 등) 바로 OCR 호출
    if file_path_obj.suffix.lower() != ".pdf":
        ocr_json = call_ocr_api(pdf_path, ocr_url, timeout)
        return ocr_response_to_blocks(ocr_json)
    
    # PDF 페이지 수 확인
    total_pages = get_pdf_page_count(pdf_path)
    
    if total_pages == 0:
        raise ValueError(f"PDF 파일을 읽을 수 없거나 페이지가 없습니다: {pdf_path}")
    
    print(f"[INFO] PDF 총 페이지 수: {total_pages}페이지")
    
    # 페이지 수가 max_pages 이하이면 바로 처리
    if total_pages <= max_pages:
        print(f"[INFO] 페이지 수가 {max_pages} 이하이므로 분할 없이 처리합니다.")
        ocr_json = call_ocr_api(pdf_path, ocr_url, timeout)
        return ocr_response_to_blocks(ocr_json)
    
    # 10페이지 초과 시 분할 처리
    print(f"[INFO] 페이지 수가 {max_pages}를 초과하므로 {max_pages}페이지 단위로 분할하여 처리합니다.")
    
    all_blocks: List[TextBlock] = []
    temp_files: List[str] = []  # 정리할 임시 파일 목록
    
    try:
        # 페이지를 max_pages 단위로 분할
        for start_page in range(1, total_pages + 1, max_pages):
            end_page = min(start_page + max_pages - 1, total_pages)
            print(f"[INFO] 페이지 {start_page}-{end_page} 처리 중... ({end_page - start_page + 1}페이지)")
            
            # PDF 분할
            split_pdf_path = split_pdf_by_pages(pdf_path, start_page, end_page)
            temp_files.append(split_pdf_path)
            
            # OCR 호출
            try:
                ocr_json = call_ocr_api(split_pdf_path, ocr_url, timeout)
                split_blocks = ocr_response_to_blocks(ocr_json)
                
                # 페이지 번호 조정 (원본 PDF의 페이지 번호로 매핑)
                for block in split_blocks:
                    # 분할된 PDF의 페이지 번호를 원본 PDF의 페이지 번호로 변환
                    original_page = start_page + block.page - 1
                    adjusted_block = TextBlock(
                        page=original_page,
                        line_no=block.line_no,
                        text=block.text,
                        bbox=block.bbox
                    )
                    all_blocks.append(adjusted_block)
                
                print(f"[INFO] 페이지 {start_page}-{end_page} 처리 완료: {len(split_blocks)}개 블록 추출")
                
            except Exception as e:
                print(f"[ERROR] 페이지 {start_page}-{end_page} OCR 처리 실패: {e}")
                # 일부 실패해도 계속 진행
        
        print(f"[INFO] 전체 처리 완료: 총 {len(all_blocks)}개 블록 추출")
        return all_blocks
    
    finally:
        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"[DEBUG] 임시 파일 삭제: {temp_file}")
            except Exception as e:
                print(f"[WARN] 임시 파일 삭제 실패 ({temp_file}): {e}")

