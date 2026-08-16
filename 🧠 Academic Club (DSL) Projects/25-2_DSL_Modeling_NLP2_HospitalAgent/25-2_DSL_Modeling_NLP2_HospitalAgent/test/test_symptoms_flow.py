#!/usr/bin/env python3
"""
증상 수집 플로우 테스트
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_symptoms_flow():
    """증상 수집 플로우 테스트"""
    try:
        from main.langgraph_workflow import run_hospital_reservation, run_hospital_reservation_with_session_data
        
        print("🧪 증상 수집 플로우 테스트 시작")
        print("=" * 60)
        
        # 1단계: 예약 요청
        print("1️⃣ 예약 요청")
        result1 = run_hospital_reservation("내일 예약 돼?", "test_session")
        print(f"✅ 성공: {result1.get('success', False)}")
        print(f"💬 응답: {result1.get('response', '응답 없음')}")
        print()
        
        # 2단계: 환자 정보 제공 (세션 데이터 포함)
        print("2️⃣ 환자 정보 제공")
        session_data = {
            "context": {"previous_intent": "reservation"},
            "reservation_info": {}
        }
        result2 = run_hospital_reservation_with_session_data("박 세현, 01024675848", "test_session", session_data)
        print(f"✅ 성공: {result2.get('success', False)}")
        print(f"💬 응답: {result2.get('response', '응답 없음')}")
        print()
        
        # 3단계: 증상 제공
        print("3️⃣ 증상 제공")
        result3 = run_hospital_reservation("무릎이 아파요", "test_session")
        print(f"✅ 성공: {result3.get('success', False)}")
        print(f"💬 응답: {result3.get('response', '응답 없음')}")
        print()
        
        print("✅ 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_symptoms_flow()
