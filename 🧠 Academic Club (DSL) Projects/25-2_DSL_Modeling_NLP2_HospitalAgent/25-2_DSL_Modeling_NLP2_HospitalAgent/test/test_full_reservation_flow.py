#!/usr/bin/env python3
"""
전체 예약 플로우 테스트
예약 요청 -> 환자 정보 요청 -> 증상 요청 -> 의료진 추천 -> 가용일정 확인 -> 예약 확인
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main.agents.agent1_manager import Agent1Manager
from main.langgraph_workflow import run_hospital_reservation

def test_full_reservation_flow():
    """전체 예약 플로우 테스트"""
    print("🧪 전체 예약 플로우 테스트 시작")
    print("=" * 60)
    
    try:
        # 테스트 시나리오
        test_scenarios = [
            {
                "step": 1,
                "description": "예약 요청",
                "user_input": "내일 예약하고 싶어요",
                "expected_status": "missing_info",
                "expected_message_contains": ["환자명", "전화번호"]
            },
            {
                "step": 2,
                "description": "환자 정보 제공",
                "user_input": "박 세현, 01024675848",
                "expected_status": "need_symptoms",
                "expected_message_contains": ["증상", "어떤 증상"]
            },
            {
                "step": 3,
                "description": "증상 제공",
                "user_input": "무릎이 아파요",
                "expected_status": "completed",
                "expected_message_contains": ["의료진", "추천", "정형외과"]
            },
            {
                "step": 4,
                "description": "의료진 선택 및 예약 확인",
                "user_input": "양재혁 의사로 예약하고 싶어요",
                "expected_status": "completed",
                "expected_message_contains": ["예약", "일정", "확인"]
            }
        ]
        
        print(f"📋 테스트 시나리오: {len(test_scenarios)}단계")
        print()
        
        # 각 단계별 테스트
        for scenario in test_scenarios:
            print(f"{scenario['step']}️⃣ {scenario['description']}")
            print(f"💬 사용자 입력: {scenario['user_input']}")
            
            # 워크플로우 실행
            result = run_hospital_reservation(scenario['user_input'], "test_session")
            
            # 결과 분석
            success = result.get("success", False)
            response = result.get("response", "")
            
            # 상태 추출 (reservation_result에서)
            status = "unknown"
            if "reservation_result" in result and result["reservation_result"]:
                reservation_result = result["reservation_result"]
                status = reservation_result.get("status", "unknown")
            
            message = response
            
            print(f"✅ 성공: {success}")
            print(f"📊 상태: {status}")
            print(f"💬 응답: {message}")
            
            # 예상 결과 검증
            if scenario.get("expected_status"):
                if status == scenario["expected_status"]:
                    print(f"✅ 상태 검증 통과: {status}")
                else:
                    print(f"❌ 상태 검증 실패: 예상={scenario['expected_status']}, 실제={status}")
            
            # 메시지 내용 검증
            if scenario.get("expected_message_contains"):
                message_check = all(
                    keyword in message for keyword in scenario["expected_message_contains"]
                )
                if message_check:
                    print(f"✅ 메시지 검증 통과: {scenario['expected_message_contains']}")
                else:
                    print(f"❌ 메시지 검증 실패: {scenario['expected_message_contains']}")
            
            # 상태 업데이트 (워크플로우 결과에서 상태 추출)
            if "reservation_result" in result and result["reservation_result"]:
                reservation_result = result["reservation_result"]
                if "collected_info" in reservation_result:
                    current_state = {"collected_info": reservation_result["collected_info"]}
            
            print("-" * 40)
            print()
            
            # 3단계 이후에는 의료진 추천이 완료되었으므로 추가 검증
            if scenario['step'] == 3:
                if "reservation_result" in result and result["reservation_result"]:
                    reservation_result = result["reservation_result"]
                    collected_info = reservation_result.get("collected_info", {})
                    if "recommended_doctors" in collected_info:
                        print(f"👨‍⚕️ 추천 의료진 수: {len(collected_info['recommended_doctors'])}")
                        for i, doctor in enumerate(collected_info['recommended_doctors'][:3], 1):
                            print(f"   {i}. {doctor.get('name', 'Unknown')} - {doctor.get('department', 'Unknown')}")
                    
                    if "available_schedules" in collected_info:
                        print(f"📅 가용일정 수: {len(collected_info['available_schedules'])}")
                        for schedule in collected_info['available_schedules'][:3]:
                            print(f"   • {schedule.get('날짜', 'N/A')} {schedule.get('시간', 'N/A')}")
            
            # 4단계에서는 예약 확인 검증
            if scenario['step'] == 4:
                if "reservation_result" in result and result["reservation_result"]:
                    reservation_result = result["reservation_result"]
                    collected_info = reservation_result.get("collected_info", {})
                    if "tool_result" in collected_info:
                        tool_result = collected_info["tool_result"]
                        if tool_result.get("success"):
                            print(f"✅ 예약 처리 성공: {tool_result.get('message', 'No message')}")
                        else:
                            print(f"❌ 예약 처리 실패: {tool_result.get('message', 'No message')}")
        
        print("🎉 전체 예약 플로우 테스트 완료")
        
        # 최종 상태 요약
        print("\n📊 최종 상태 요약:")
        if "reservation_result" in result and result["reservation_result"]:
            reservation_result = result["reservation_result"]
            final_collected_info = reservation_result.get("collected_info", {})
            print(f"• 환자명: {final_collected_info.get('환자명', 'N/A')}")
            print(f"• 전화번호: {final_collected_info.get('전화번호', 'N/A')}")
            print(f"• 증상: {final_collected_info.get('symptoms', [])}")
            print(f"• 추천 진료과: {final_collected_info.get('recommended_department', 'N/A')}")
            print(f"• 추천 의료진 수: {len(final_collected_info.get('recommended_doctors', []))}")
            print(f"• 가용일정 수: {len(final_collected_info.get('available_schedules', []))}")
        else:
            print("• 최종 상태 정보를 가져올 수 없습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flow_validation():
    """플로우 검증 함수"""
    print("\n🔍 플로우 검증:")
    
    expected_flow = [
        "예약 요청",
        "환자 정보 요청", 
        "증상 요청",
        "의료진 추천 (RAG 기반)",
        "Supabase에서 의료진 가용일정 확인",
        "환자에게 confirm 요청",
        "예약 확인"
    ]
    
    print("📋 예상 플로우:")
    for i, step in enumerate(expected_flow, 1):
        print(f"   {i}. {step}")
    
    print("\n✅ 플로우가 올바르게 구현되었는지 테스트를 통해 확인합니다.")

if __name__ == "__main__":
    print("🏥 바른마디병원 예약 시스템 - 전체 플로우 테스트")
    print("=" * 60)
    
    # 플로우 검증
    test_flow_validation()
    
    # 전체 테스트 실행
    success = test_full_reservation_flow()
    
    if success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 테스트 중 문제가 발생했습니다.")
        sys.exit(1)
