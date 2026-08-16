#!/usr/bin/env python3
"""
세션 상태 관리 개선 테스트
연속적인 예약 플로우에서 상태가 제대로 유지되는지 확인
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main.langgraph_workflow import run_continuous_reservation_flow

def test_session_flow():
    """세션 상태 관리 테스트"""
    print("🧪 세션 상태 관리 테스트 시작")
    print("=" * 60)
    
    try:
        # 연속적인 예약 플로우 테스트
        user_queries = [
            "내일 예약하고 싶어요",
            "박 세현, 01024675848", 
            "무릎이 아파요",
            "양재혁 의사로 예약하고 싶어요"
        ]
        
        print(f"📋 테스트 시나리오: {len(user_queries)}단계")
        print("1. 예약 요청")
        print("2. 환자 정보 제공")
        print("3. 증상 제공")
        print("4. 의료진 선택 및 예약 확인")
        print()
        
        # 연속 플로우 실행
        result = run_continuous_reservation_flow(user_queries, "test_session")
        
        # 결과 분석
        print("\n📊 테스트 결과:")
        print(f"✅ 성공: {result.get('success', False)}")
        print(f"💬 최종 응답: {result.get('response', '')}")
        
        # 세션 상태 확인
        reservation_info = result.get('reservation_info', {})
        print(f"\n🔍 최종 세션 상태:")
        print(f"• 환자명: {reservation_info.get('환자명', 'N/A')}")
        print(f"• 전화번호: {reservation_info.get('전화번호', 'N/A')}")
        print(f"• 증상: {reservation_info.get('symptoms', [])}")
        print(f"• 추천 진료과: {reservation_info.get('recommended_department', 'N/A')}")
        print(f"• 추천 의료진 수: {len(reservation_info.get('recommended_doctors', []))}")
        print(f"• 가용일정 수: {len(reservation_info.get('available_schedules', []))}")
        
        # 예약 결과 확인
        reservation_result = result.get('reservation_result', {})
        if reservation_result:
            print(f"\n📅 예약 처리 결과:")
            print(f"• 상태: {reservation_result.get('status', 'N/A')}")
            print(f"• 성공: {reservation_result.get('success', False)}")
            print(f"• 메시지: {reservation_result.get('message', 'N/A')}")
        
        # 오류 확인
        if result.get('error'):
            print(f"\n❌ 오류 발생:")
            print(f"• 오류: {result.get('error')}")
            print(f"• 오류 단계: {result.get('error_step')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_steps():
    """개별 단계별 테스트 (세션 데이터 전달)"""
    print("\n🧪 개별 단계별 테스트")
    print("=" * 40)
    
    try:
        from main.langgraph_workflow import run_hospital_reservation, run_hospital_reservation_with_session_data
        
        # 1단계: 예약 요청
        print("1️⃣ 예약 요청")
        result1 = run_hospital_reservation("내일 예약하고 싶어요", "test_session")
        print(f"✅ 성공: {result1.get('success', False)}")
        print(f"💬 응답: {result1.get('response', '')}")
        
        # 2단계: 환자 정보 제공 (세션 데이터 포함)
        print("\n2️⃣ 환자 정보 제공")
        session_data = {
            "context": result1.get('context', {}),
            "reservation_info": result1.get('reservation_info', {})
        }
        result2 = run_hospital_reservation_with_session_data("박 세현, 01024675848", "test_session", session_data)
        print(f"✅ 성공: {result2.get('success', False)}")
        print(f"💬 응답: {result2.get('response', '')}")
        
        # 3단계: 증상 제공 (세션 데이터 포함)
        print("\n3️⃣ 증상 제공")
        session_data = {
            "context": result2.get('context', {}),
            "reservation_info": result2.get('reservation_info', {})
        }
        result3 = run_hospital_reservation_with_session_data("무릎이 아파요", "test_session", session_data)
        print(f"✅ 성공: {result3.get('success', False)}")
        print(f"💬 응답: {result3.get('response', '')}")
        
        # 4단계: 의료진 선택 (세션 데이터 포함)
        print("\n4️⃣ 의료진 선택")
        session_data = {
            "context": result3.get('context', {}),
            "reservation_info": result3.get('reservation_info', {})
        }
        result4 = run_hospital_reservation_with_session_data("양재혁 의사로 예약하고 싶어요", "test_session", session_data)
        print(f"✅ 성공: {result4.get('success', False)}")
        print(f"💬 응답: {result4.get('response', '')}")
        
        # 최종 상태 확인
        final_reservation_info = result4.get('reservation_info', {})
        print(f"\n🔍 최종 상태:")
        print(f"• 환자명: {final_reservation_info.get('환자명', 'N/A')}")
        print(f"• 전화번호: {final_reservation_info.get('전화번호', 'N/A')}")
        print(f"• 증상: {final_reservation_info.get('symptoms', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 개별 단계 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏥 바른마디병원 예약 시스템 - 세션 상태 관리 테스트")
    print("=" * 60)
    
    # 연속 플로우 테스트
    print("🔄 연속 플로우 테스트")
    success1 = test_session_flow()
    
    # 개별 단계 테스트
    print("\n" + "=" * 60)
    print("🔄 개별 단계 테스트")
    success2 = test_individual_steps()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약:")
    print(f"• 연속 플로우: {'✅ 성공' if success1 else '❌ 실패'}")
    print(f"• 개별 단계: {'✅ 성공' if success2 else '❌ 실패'}")
    
    if success1 and success2:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 일부 테스트에서 문제가 발생했습니다.")
        sys.exit(1)
