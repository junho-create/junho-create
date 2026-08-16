import subprocess
import os
import glob
import sys
import json
import time # 시간 측정을 위해 time 모듈 추가

# =================================================================
# 1. 환경 설정
# =================================================================

# 프로젝트의 현재 디렉토리 (예: .../HospitalAgent/evaluation)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# PROJECT_ROOT(프로젝트 최상위): 현재 디렉토리의 상위 디렉토리로 설정
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

# 테스트 파일 디렉토리: PROJECT_ROOT/rag_doctor_agent/tests
TESTS_DIR = os.path.join(PROJECT_ROOT, "rag_doctor_agent", "tests")

# run_sample.py의 경로: PROJECT_ROOT/rag_doctor_agent/run_sample.py
RUN_SAMPLE_SCRIPT = os.path.join(PROJECT_ROOT, "rag_doctor_agent", "run_sample.py")

# 개별 JSON 결과 저장 경로
INDIVIDUAL_RESULTS_DIR = os.path.join(CURRENT_DIR, "individual_results")


# =================================================================
# 2. RAGAS 외부 로깅 함수 (개별 파일 저장)
# =================================================================

def save_individual_test_result(
    test_filename: str, 
    json_file_path: str, 
    raw_rag_result: dict, 
    execution_time: float
):
    """
    STDOUT에서 캡처한 원시 RAG 결과를 개별 JSON 파일에 저장합니다.
    """
    
    # 2-1. 원본 테스트 JSON에서 질문(symptoms + info)을 재구성
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            symptoms = test_data.get("symptoms", [])
            other_info_list = test_data.get("other_info", [])
            user_question = f"증상: {', '.join(symptoms)}. {', '.join(other_info_list)}"
    except Exception:
        user_question = f"Test File Error: {test_filename}"
        
    # 2-2. 최종 답변 텍스트 구성
    answer_parts = []
    if "top_k_suggestions" in raw_rag_result:
        for sug in raw_rag_result["top_k_suggestions"]:
            if sug.get("의료진명") and sug.get("진료과"):
                answer_parts.append(f"{sug['진료과']}의 {sug['의료진명']} (이유: {sug.get('이유', '정보 없음')})")
    
    final_answer = f"추천 진료과: {raw_rag_result.get('dept', '불명')}. 추천 의료진: {'; '.join(answer_parts[:3])}"
    if not answer_parts:
        final_answer = f"추천 진료과: {raw_rag_result.get('dept', '불명')} - 상세 의료진 정보 없음."

    # 2-3. 최종 로그 데이터 구성 (시간 포함)
    log_data = {
        "test_file": test_filename,
        "execution_time_seconds": round(execution_time, 3), # 실행 시간 (소수점 셋째 자리까지)
        "user_question": user_question,
        "final_answer": final_answer,
        # 원본 JSON 출력 전체를 저장하여 RAGAS 평가를 위한 contexts 추출에 사용
        "raw_rag_result": raw_rag_result
    }
    
    # 2-4. JSON 파일에 저장
    os.makedirs(INDIVIDUAL_RESULTS_DIR, exist_ok=True)
    output_filepath = os.path.join(INDIVIDUAL_RESULTS_DIR, test_filename)
    
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"🗃️ 결과 저장 완료: {output_filepath}")
            
    except Exception as e:
        print(f"⚠️ 개별 JSON 저장 실패 ({test_filename}): {e}")


# =================================================================
# 3. 테스트 실행 함수
# =================================================================

def run_all_tests():
    # 1. run_sample.py 파일 존재 여부 확인
    if not os.path.exists(RUN_SAMPLE_SCRIPT):
        print(f"❌ 오류: run_sample.py를 찾을 수 없습니다.")
        print(f"  예상 경로: {RUN_SAMPLE_SCRIPT}")
        sys.exit(1)
    
    # PYTHONPATH 설정
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([PROJECT_ROOT] + env.get('PYTHONPATH', '').split(os.pathsep))
        
    # 2. tests/sample_*.json 파일을 모두 찾습니다.
    test_files = glob.glob(os.path.join(TESTS_DIR, "sample_*.json"))
    
    if not test_files:
        print(f"⚠️ 테스트 파일이 '{TESTS_DIR}' 폴더에 없습니다. 먼저 generate_all_rag_tests.py를 실행해주세요.")
        return

    print(f"🚀 총 {len(test_files)}개의 RAG 테스트를 시작합니다...")
    
    # 이전 결과 폴더 삭제 (깨끗한 테스트를 위해)
    if os.path.exists(INDIVIDUAL_RESULTS_DIR):
        import shutil
        shutil.rmtree(INDIVIDUAL_RESULTS_DIR)
        print(f"🗑️ 이전 결과 폴더 삭제 완료: {INDIVIDUAL_RESULTS_DIR}")
    
    for file_path in sorted(test_files):
        filename = os.path.basename(file_path)
        absolute_file_path = os.path.abspath(file_path)

        print(f"\n--- 🧪 Testing: {filename} ---")
        
        start_time = time.time() # ⏱️ 시간 측정 시작
        
        try:
            # subprocess를 사용하여 run_sample.py 실행 (STDOUT 캡처)
            result = subprocess.run(
                ["python", RUN_SAMPLE_SCRIPT, absolute_file_path],
                capture_output=True,
                text=True,
                encoding="utf-8", 
                env=env,
                cwd=PROJECT_ROOT, 
                check=True 
            )
            
            end_time = time.time() # ⏱️ 시간 측정 종료
            execution_time = end_time - start_time
            
            # STDOUT 캡처 및 개별 JSON 파일로 저장
            if result.stdout.strip():
                try:
                    # run_sample.py의 출력을 JSON 객체로 로드
                    raw_rag_result = json.loads(result.stdout)
                    
                    # 개별 파일 저장 및 시간 정보 포함
                    save_individual_test_result(filename, absolute_file_path, raw_rag_result, execution_time)
                    print(f"✅ {filename} 테스트 성공 (시간: {execution_time:.3f}s)")
                    
                except json.JSONDecodeError:
                    print(f"⚠️ {filename} 경고: STDOUT이 유효한 JSON이 아닙니다. 저장을 건너뜁니다.")
                    print("--- STDOUT (ERROR) ---")
                    print(result.stdout)
                    
            else:
                 print(f"❌ {filename} 테스트 실패 (STDOUT 출력 없음)")
                 
            if result.stderr:
                 print("--- STDERR ---")
                 print(result.stderr)
            
        except subprocess.CalledProcessError as e:
            # subprocess 실행 중 오류 발생 시
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"❌ {filename} 테스트 실패 (Return Code: {e.returncode}, 시간: {execution_time:.3f}s)")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
        except FileNotFoundError:
            print(f"❌ Python 실행 파일을 찾을 수 없습니다. PATH 설정을 확인해주세요.")
            break
            
    print("\n===============================")
    print(f"✨ 모든 RAG 테스트 실행 완료. 개별 결과 저장 경로: {INDIVIDUAL_RESULTS_DIR}")
    
if __name__ == "__main__":
    run_all_tests()
