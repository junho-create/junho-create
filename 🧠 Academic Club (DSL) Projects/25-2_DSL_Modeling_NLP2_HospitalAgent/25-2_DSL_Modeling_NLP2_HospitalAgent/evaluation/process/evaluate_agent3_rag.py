import json
import pandas as pd
import os
import glob
import time
from typing import List, Dict, Any, Tuple

# RAGAS 및 데이터셋 관련 임포트
from datasets import Dataset 
from ragas import evaluate
# 💡 오류 해결 1: RAGAS의 실제 내부 모듈 경로를 직접 지정
from ragas.metrics._faithfulness import faithfulness 
from ragas.metrics._answer_relevance import answer_relevancy

# LLM 환경 설정 및 RAGAS LLM 래퍼 임포트
from dotenv import load_dotenv
# 💡 오류 해결 2 & 3: BaseRagasLLM 대신 LangchainLLMWrapper를 사용하고, llm 객체를 위치 인수로 전달
from ragas.llms.base import LangchainLLMWrapper 
from langchain_openai import ChatOpenAI


# =================================================================
# 1. 환경 설정 및 경로
# =================================================================

# 스크립트의 현재 디렉토리 (evaluation 폴더)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 (HospitalAgent 폴더)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
load_dotenv(os.path.join(PROJECT_ROOT, ".env")) # .env 파일 로드

# 개별 JSON 결과가 저장된 폴더 경로
INDIVIDUAL_RESULTS_DIR = os.path.join(CURRENT_DIR, "individual_results")
# 최종 CSV 결과 파일명
OUTPUT_CSV_FILE = os.path.join(CURRENT_DIR, "ragas_evaluation_results_final.csv")

# 💡 원본 문서 인덱스 파일 경로 (고객님 제공 경로 기반)
INDEX_FILE_PATH = os.path.join(PROJECT_ROOT, "rag_doctor_agent", "main", "data", "db_data", "index.json")


# =================================================================
# 2. 검색 ID -> 원본 콘텐츠 매핑 로직
# =================================================================

def load_retrieval_map(path: str) -> Dict[str, str]:
    """
    index.json 파일을 로드하여 retrieval evidence ID를 원본 텍스트 콘텐츠로 매핑합니다.
    """
    content_map = {}
    print(f"📄 원본 인덱스 파일 로드 중: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            
            if isinstance(index_data, dict):
                # index.json 파일 구조에 따라 ID를 키로, 콘텐츠를 값으로 매핑
                content_map = {k: v.get('content', str(v)) if isinstance(v, dict) else str(v) for k, v in index_data.items()}

    except FileNotFoundError:
        print(f"❌ 오류: 원본 인덱스 파일({path})을 찾을 수 없습니다. Faithfulness 평가에 영향을 줍니다.")
    except Exception as e:
        print(f"❌ 원본 인덱스 로드 중 오류 발생: {e}")
        
    return content_map

# =================================================================
# 3. RAGAS 데이터 로드 및 전처리
# =================================================================

def load_ragas_data() -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    개별 JSON 파일들을 읽어 RAGAS 데이터와 실행 시간을 로드하고,
    retrieval evidence ID를 원본 텍스트로 대체합니다.
    """
    
    # 원본 콘텐츠 매핑 로드
    content_map = load_retrieval_map(INDEX_FILE_PATH)
    
    ragas_data = []
    execution_times = []
    json_files = glob.glob(os.path.join(INDIVIDUAL_RESULTS_DIR, "sample_*.json"))
    
    if not json_files:
        print(f"❌ 오류: '{INDIVIDUAL_RESULTS_DIR}' 폴더에 JSON 결과 파일이 없습니다. 테스트를 먼저 실행하세요.")
        return [], []

    print(f"📄 총 {len(json_files)}개의 개별 테스트 결과를 로드 중...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                
                execution_times.append(entry.get("execution_time_seconds", 0.0))
                raw_result = entry.get("raw_rag_result", {})
                
                # 💡 contexts 추출 및 ID -> 콘텐츠 대체
                retrieval_ids = raw_result.get("retrieval_evidence", [])
                
                # ID를 content_map에서 조회하여 실제 텍스트를 contexts 리스트에 추가합니다.
                contexts = []
                for doc_id in retrieval_ids:
                    content = content_map.get(doc_id, f"Content Retrieval Failed for ID: {doc_id}")
                    # 컨텍스트가 너무 짧거나 오류 메시지가 아니면 추가
                    if len(content) > 20 and "Failed" not in content:
                         contexts.append(content)
                
                # RAGAS 데이터셋 구성
                ragas_data.append({
                    "test_file": entry.get("test_file"),
                    "execution_time_seconds": entry.get("execution_time_seconds"),
                    "question": entry.get("user_question", "No Question"),
                    "answer": entry.get("final_answer", "No Answer"),
                    "contexts": contexts if contexts else ["No Context Retrieved or Mapped"], 
                    "ground_truths": []
                })
                
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생 ({file_path}): {e}")
            continue
            
    print(f"✅ 총 {len(ragas_data)}개의 유효 데이터 로드 완료.")
    return ragas_data, execution_times


def evaluate_agent3_metrics():
    """Agent3의 Faithfulness와 Answer Relevancy를 RAGAS로 평가합니다."""
    
    start_time_total = time.time() # ⏱️ 총 분석 시간 측정 시작
    
    # 1. 데이터 로드 및 Dataset 변환
    ragas_data, execution_times = load_ragas_data()
    
    if not ragas_data:
        return

    df = pd.DataFrame(ragas_data)
    dataset = Dataset.from_pandas(df)

    # 2. LLM 설정: .env에서 API 키 로드
    openai_api_key = os.environ.get("OPENAI_API_KEY") 
    if not openai_api_key:
        print("🛑 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. RAGAS 평가 불가.")
        return

    try:
        # LLM 초기화 (gpt-4o-mini 사용)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key) 
        # 💡 LLM 래핑 오류 최종 해결: LangchainLLMWrapper에 llm 객체를 위치 인수로 전달
        ragas_llm = LangchainLLMWrapper(llm)
        
        # 메트릭에 LLM 할당
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm

    except Exception as e:
        print(f"🛑 LLM 초기화 오류: {e}. API 키와 모델명을 확인하세요.")
        return

    # 3. RAGAS 평가 실행 (LLM 호출)
    print("🚀 RAGAS 평가 실행 중... (Faithfulness, Answer Relevancy)")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,          
            answer_relevancy,      
        ],
    )
    
    end_time_total = time.time() # ⏱️ 총 분석 시간 측정 종료
    total_analysis_time = end_time_total - start_time_total

    # 4. 결과 출력 및 저장
    results_df = result.to_pandas()
    
    # 메타데이터 (실행 시간) 통합
    meta_df = df[['test_file', 'execution_time_seconds']].copy()
    results_df = pd.concat([meta_df, results_df], axis=1)

    print("\n--- 📊 RAGAS 평가 요약 결과 ---")
    print(result) 
    
    # ⏱️ 시간 분석 결과 계산 및 출력
    if execution_times:
        total_rag_exec_time = sum(execution_times)
        avg_rag_exec_time = total_rag_exec_time / len(execution_times)
        print("\n--- ⏱️ 시간 분석 결과 ---")
        print(f"평가 대상 테스트 수: {len(execution_times)}개")
        print(f"전체 RAG 실행 시간 합계: {total_rag_exec_time:.3f} 초")
        print(f"테스트 케이스 1개당 평균 실행 시간: {avg_rag_exec_time:.3f} 초")
    
    print("\n--- 📋 상세 평가 결과 (상위 5개) ---")
    print(results_df[['test_file', 'execution_time_seconds', 'faithfulness', 'answer_relevancy']].head())
    
    # CSV 파일로 저장
    results_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"\n✅ 상세 결과가 '{OUTPUT_CSV_FILE}'에 저장되었습니다.")
    print(f"⏱️ 총 RAGAS 분석 시간: {total_analysis_time:.3f} 초")


if __name__ == "__main__":
    evaluate_agent3_metrics()