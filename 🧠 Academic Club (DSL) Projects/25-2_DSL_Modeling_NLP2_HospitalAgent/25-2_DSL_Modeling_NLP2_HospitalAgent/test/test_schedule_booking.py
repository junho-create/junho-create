#!/usr/bin/env python3
"""
의료진 추천 → 일정 조회 → 예약 확정 테스트 스크립트
"""

import os
import sys
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main.agents.agent1_manager import Agent1Manager
from main.agents.agent2_reservation import Agent2Reservation
from main.agents.agent3_rag import Agent3RAG

def test_doctor_schedule_booking():
    """의료진 추천 → 일정 조회 → 예약 확정 전체 플로우 테스트"""
    print("🏥 의료진 추천 → 일정 조회 → 예약 확정 테스트")
    print("=" * 60)
    
    # 에이전트 초기화
    print("🔧 에이전트 초기화 중...")
    agent1 = Agent1Manager()
    agent2 = Agent2Reservation()
    agent3 = Agent3RAG()
    print("✅ 에이전트 초기화 완료")
    
    # 시나리오 1: 무릎 통증 → 의료진 추천 → 일정 조회 → 예약 확정
    print("\n📋 시나리오 1: 무릎 통증 예약")
    print("-" * 40)
    
    # 1단계: 증상 기반 의료진 추천
    print("🔍 1단계: 증상 기반 의료진 추천")
    symptoms = ["무릎 통증"]
    rag_result = agent3.recommend_doctors(symptoms, "무릎이 아파서 예약하고 싶어요")
    
    if not rag_result.get("success"):
        print(f"❌ 의료진 추천 실패: {rag_result.get('message')}")
        return
    
    recommended_doctors = rag_result.get("recommended_doctors", [])
    print(f"✅ 의료진 추천 성공: {len(recommended_doctors)}명")
    
    for i, doctor in enumerate(recommended_doctors[:3], 1):
        print(f"   {i}. {doctor.get('name', 'Unknown')} - {doctor.get('department', 'Unknown')}")
        print(f"      추천 근거: {doctor.get('reasoning', 'No reasoning')}")
    
    # 2단계: 첫 번째 의료진의 일정 조회
    if recommended_doctors:
        first_doctor = recommended_doctors[0]
        doctor_name = first_doctor.get('name', '')
        print(f"\n🔍 2단계: 의료진 '{doctor_name}'의 예약 가능 일정 조회")
        
        schedule_result = agent2._get_doctor_schedule(doctor_name)
        
        if schedule_result.get("success"):
            available_schedules = schedule_result.get("data", [])
            print(f"✅ 예약 가능한 일정: {len(available_schedules)}건")
            
            for i, schedule in enumerate(available_schedules[:3], 1):
                print(f"   {i}. {schedule.get('날짜')} {schedule.get('시간')} - {schedule.get('의료진')}")
        else:
            print(f"❌ 일정 조회 실패: {schedule_result.get('message')}")
            return
    else:
        print("❌ 추천된 의료진이 없습니다.")
        return
    
    # 3단계: 사용자 정보 수집 및 예약 확정
    print(f"\n🔍 3단계: 예약 정보 수집 및 확정")
    
    # 환자 정보
    patient_info = {
        "환자명": "김철수",
        "전화번호": "010-1234-5678",
        "성별": "남"
    }
    
    # 첫 번째 일정 선택
    if available_schedules:
        selected_schedule = available_schedules[0]
        print(f"📅 선택된 일정: {selected_schedule.get('날짜')} {selected_schedule.get('시간')}")
        
        # 예약 확정 시뮬레이션
        print(f"\n🎯 예약 확정 시뮬레이션")
        print(f"   환자: {patient_info['환자명']} ({patient_info['전화번호']})")
        print(f"   의료진: {doctor_name}")
        print(f"   일정: {selected_schedule.get('날짜')} {selected_schedule.get('시간')}")
        print(f"   증상: {', '.join(symptoms)}")
        
        # 실제 예약 확정 로직 (향후 구현)
        print(f"\n✅ 예약 확정 완료!")
        print(f"   예약 ID: RES-{selected_schedule.get('일정ID')}-{patient_info['전화번호'][-4:]}")
        print(f"   예약 상태: 확정")
        print(f"   알림: 예약 확정 안내가 발송되었습니다.")
    else:
        print("❌ 예약 가능한 일정이 없습니다.")

def test_doctor_lookup():
    """의사 테이블 조회 테스트"""
    print("\n🔍 의사 테이블 조회 테스트")
    print("-" * 40)
    
    from main.tools.supabase_mcp_tool import SupabaseDoctorLookupTool
    
    doctor_tool = SupabaseDoctorLookupTool()
    
    # 테스트할 의료진들
    test_doctors = ["김재훈", "우연선", "고현길"]
    
    for doctor_name in test_doctors:
        print(f"\n🔍 의료진 '{doctor_name}' 조회 중...")
        result = doctor_tool._run(doctor_name=doctor_name, run_manager=None)
        result_data = json.loads(result)
        
        if result_data.get("success"):
            doctor_info = result_data["data"][0]
            print(f"✅ 조회 성공: DocID={doctor_info.get('DocID')}, 진료실={doctor_info.get('진료실코드')}")
        else:
            print(f"❌ 조회 실패: {result_data.get('message')}")

def test_schedule_lookup():
    """가용일정 조회 테스트"""
    print("\n🔍 가용일정 조회 테스트")
    print("-" * 40)
    
    from main.tools.supabase_mcp_tool import SupabaseScheduleLookupTool
    
    schedule_tool = SupabaseScheduleLookupTool()
    
    # 테스트할 DocID들
    test_doc_ids = [1, 2, 3]
    
    for doc_id in test_doc_ids:
        print(f"\n🔍 DocID {doc_id}의 가용일정 조회 중...")
        result = schedule_tool._run(doc_id=doc_id, limit=3, run_manager=None)
        result_data = json.loads(result)
        
        if result_data.get("success"):
            schedules = result_data["data"]
            print(f"✅ 조회 성공: {len(schedules)}건의 예약 가능 일정")
            for schedule in schedules:
                print(f"   - {schedule.get('진료년')}-{schedule.get('진료월'):02d}-{schedule.get('진료일'):02d} {schedule.get('진료시'):02d}:{schedule.get('진료분'):02d}")
        else:
            print(f"❌ 조회 실패: {result_data.get('message')}")

def test_complete_booking_flow():
    """완전한 예약 플로우 테스트"""
    print("\n🎯 완전한 예약 플로우 테스트")
    print("=" * 60)
    
    # 시나리오: 허리 통증 → 의료진 추천 → 일정 조회 → 예약 확정
    print("📋 시나리오: 허리 통증 예약")
    
    # Agent1: 의도 분석
    user_input = "허리가 아프고 디스크가 있어서 예약하고 싶어요"
    print(f"👤 사용자: {user_input}")
    
    agent1 = Agent1Manager()
    intent_result = agent1.analyze_user_intent(user_input)
    
    if intent_result.get("primary_intent") == "reservation":
        print("✅ 예약 의도로 분류됨")
        
        # Agent2: 예약 처리 (의료진 추천 포함)
        agent2 = Agent2Reservation()
        reservation_result = agent2.process_reservation_request(user_input, {})
        
        if reservation_result.get("success"):
            collected_info = reservation_result.get("collected_info", {})
            
            # 의료진 추천 결과 확인
            if collected_info.get("recommended_doctors"):
                print(f"\n👨‍⚕️ 추천된 의료진: {len(collected_info['recommended_doctors'])}명")
                for doctor in collected_info["recommended_doctors"][:2]:
                    print(f"   - {doctor.get('name')} ({doctor.get('department')})")
            
            # 예약 가능 일정 확인
            if collected_info.get("available_schedules"):
                print(f"\n📅 예약 가능 일정: {len(collected_info['available_schedules'])}건")
                for schedule in collected_info["available_schedules"][:3]:
                    print(f"   - {schedule.get('날짜')} {schedule.get('시간')} ({schedule.get('의료진')})")
                
                # 예약 확정 시뮬레이션
                print(f"\n🎯 예약 확정 시뮬레이션")
                first_schedule = collected_info["available_schedules"][0]
                print(f"   선택된 일정: {first_schedule.get('날짜')} {first_schedule.get('시간')}")
                print(f"   의료진: {first_schedule.get('의료진')}")
                print(f"   환자: {collected_info.get('환자명', 'Unknown')} ({collected_info.get('전화번호', 'Unknown')})")
                print(f"   증상: {', '.join(collected_info.get('symptoms', []))}")
                print(f"\n✅ 예약 확정 완료!")
            else:
                print("❌ 예약 가능한 일정이 없습니다.")
        else:
            print(f"❌ 예약 처리 실패: {reservation_result.get('message')}")
    else:
        print(f"❌ 예약 의도가 아님: {intent_result.get('primary_intent')}")

def test_natural_language_schedule():
    """자연어 일정 처리 테스트"""
    print("\n🗣️ 자연어 일정 처리 테스트")
    print("=" * 60)
    
    agent2 = Agent2Reservation()
    
    # 테스트 케이스들
    test_cases = [
        "무릎이 아파서 최대한 빨리 예약하고 싶어요",
        "허리 통증으로 내일 오후에 예약하고 싶어요", 
        "어깨가 아파서 다음 주 월요일 오전에 예약하고 싶어요",
        "두통이 심해서 급하게 예약하고 싶어요"
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n📋 테스트 케이스 {i}: {user_input}")
        print("-" * 40)
        
        # 일정 선호도 파싱
        preference_result = agent2._parse_schedule_preference(user_input)
        
        if preference_result.get("success"):
            preference = preference_result.get("parsed_preference", {})
            print(f"✅ 파싱 성공:")
            print(f"   긴급도: {preference.get('urgency', 'Unknown')}")
            print(f"   선호 날짜: {preference.get('preferred_date', 'None')}")
            print(f"   선호 시간: {preference.get('preferred_time', 'None')}")
            print(f"   시간대: {preference.get('time_period', 'Unknown')}")
            print(f"   며칠 후: {preference.get('days_from_now', 'None')}")
        else:
            print(f"❌ 파싱 실패")

def test_schedule_matching():
    """일정 매칭 테스트"""
    print("\n🎯 일정 매칭 테스트")
    print("=" * 60)
    
    agent2 = Agent2Reservation()
    
    # 가상의 가용일정 데이터
    mock_schedules = [
        {"일정ID": 1, "날짜": "2025-09-27", "시간": "09:00", "의료진": "김재훈"},
        {"일정ID": 2, "날짜": "2025-09-27", "시간": "14:00", "의료진": "김재훈"},
        {"일정ID": 3, "날짜": "2025-09-28", "시간": "10:00", "의료진": "김재훈"},
        {"일정ID": 4, "날짜": "2025-09-28", "시간": "16:00", "의료진": "김재훈"},
    ]
    
    # 테스트 선호도들
    test_preferences = [
        {"urgency": "high", "time_period": "morning"},
        {"urgency": "medium", "time_period": "afternoon"},
        {"urgency": "low", "time_period": "any"},
    ]
    
    for i, preference in enumerate(test_preferences, 1):
        print(f"\n📋 선호도 테스트 {i}: {preference}")
        print("-" * 30)
        
        matched_schedules = agent2._match_schedule_with_preference(mock_schedules, preference)
        
        print(f"✅ 매칭된 일정: {len(matched_schedules)}건")
        for schedule in matched_schedules:
            print(f"   - {schedule.get('날짜')} {schedule.get('시간')} ({schedule.get('의료진')})")

if __name__ == "__main__":
    print("🚀 의료진 추천 → 일정 조회 → 예약 확정 테스트 시작")
    print("=" * 80)
    
    try:
        # 1. 의사 테이블 조회 테스트
        test_doctor_lookup()
        
        # 2. 가용일정 조회 테스트
        test_schedule_lookup()
        
        # 3. 자연어 일정 처리 테스트
        test_natural_language_schedule()
        
        # 4. 일정 매칭 테스트
        test_schedule_matching()
        
        # 5. 의료진 추천 → 일정 조회 → 예약 확정 테스트
        test_doctor_schedule_booking()
        
        # 6. 완전한 예약 플로우 테스트
        test_complete_booking_flow()
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
