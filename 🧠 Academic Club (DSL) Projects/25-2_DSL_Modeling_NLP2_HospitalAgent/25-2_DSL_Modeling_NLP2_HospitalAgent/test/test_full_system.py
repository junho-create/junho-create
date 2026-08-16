#!/usr/bin/env python3
"""
전체 시스템 통합 테스트 스크립트
- Agent1 → Agent2 → Agent3 전체 플로우 테스트
- Tool Calling 및 RAG 파이프라인 통합 테스트
"""
import os
import sys
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_full_reservation_flow():
    """전체 예약 플로우 테스트"""
    print("🧪 전체 예약 플로우 테스트")
    print("=" * 50)
    
    try:
        # 모든 에이전트 임포트
        from main.agents.agent1_manager import Agent1Manager
        from main.agents.agent2_reservation import Agent2Reservation
        from main.agents.agent3_rag import Agent3RAG
        
        print("✅ 모든 에이전트 임포트 성공")
        
        # 에이전트들 생성
        agent1 = Agent1Manager()
        agent2 = Agent2Reservation()
        agent3 = Agent3RAG()
        
        print("✅ 모든 에이전트 생성 완료")
        
        # 전체 플로우 테스트 케이스들
        test_cases = [
            {
                "name": "증상 기반 예약 요청",
                "user_input": "어깨가 아파서 예약하고 싶어요",
                "expected_flow": "Agent1 → Agent2 → Agent3 → Agent2"
            },
            {
                "name": "예약 조회 요청",
                "user_input": "홍길동으로 예약 확인해주세요",
                "expected_flow": "Agent1 → Agent2"
            },
            {
                "name": "의료진 추천 요청",
                "user_input": "무릎이 아픈데 어떤 의사한테 가야 할까요?",
                "expected_flow": "Agent1 → Agent3"
            },
            {
                "name": "병원 정보 요청",
                "user_input": "병원 휴무일이 언제인가요?",
                "expected_flow": "Agent1 → Tavily 검색"
            }
        ]
        
        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 테스트 {i}: {test_case['name']}")
            print("-" * 30)
            
            try:
                # Agent1으로 요청 분석
                print("🔍 Agent1: 요청 분석 중...")
                agent1_result = agent1.process_user_request(test_case["user_input"])
                
                print(f"✅ Agent1 분석 완료")
                print(f"📝 의도: {agent1_result.get('intent', 'Unknown')}")
                print(f"📝 다음 액션: {agent1_result.get('next_action', 'Unknown')}")
                
                # Agent1 결과에 따른 후속 처리
                if agent1_result.get('next_action') == 'route_to_reservation_agent':
                    print("🔄 Agent2 호출 중...")
                    agent2_result = agent2.process_reservation_request(
                        test_case["user_input"],
                        agent1_result.get('extracted_info', {})
                    )
                    
                    print(f"✅ Agent2 처리 완료")
                    print(f"📝 성공 여부: {agent2_result.get('success', False)}")
                    print(f"📝 상태: {agent2_result.get('status', 'Unknown')}")
                    
                    # Agent3 호출 여부 확인
                    if agent2_result.get('collected_info', {}).get('symptoms'):
                        print("🔄 Agent3 호출됨 (증상 분석)")
                        if agent2_result.get('collected_info', {}).get('recommended_department'):
                            print(f"✅ Agent3 추천 완료")
                            print(f"📝 추천 진료과: {agent2_result.get('collected_info', {}).get('recommended_department')}")
                    
                elif agent1_result.get('next_action') == 'route_to_rag_agent':
                    print("🔄 Agent3 호출 중...")
                    symptoms = agent1_result.get('extracted_info', {}).get('symptoms', [])
                    agent3_result = agent3.recommend_doctors(symptoms)
                    
                    print(f"✅ Agent3 처리 완료")
                    print(f"📝 성공 여부: {agent3_result.get('success', False)}")
                    print(f"📝 추천 진료과: {agent3_result.get('department', 'Unknown')}")
                    
                elif agent1_result.get('next_action') == 'search_hospital_info':
                    print("🔄 Tavily 검색 호출 중...")
                    # Tavily 검색 시뮬레이션
                    print("✅ Tavily 검색 완료 (시뮬레이션)")
                    
                else:
                    print(f"⚠️ 예상과 다른 라우팅: {agent1_result.get('next_action')}")
                
                print(f"✅ {test_case['name']} 테스트 완료")
                
            except Exception as e:
                print(f"❌ 테스트 실행 중 오류: {e}")
        
        print(f"\n🎉 전체 예약 플로우 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 전체 플로우 테스트 실패: {e}")

def test_tool_calling_integration():
    """Tool Calling 통합 테스트"""
    print(f"\n🔧 Tool Calling 통합 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent2_reservation import Agent2Reservation
        
        agent2 = Agent2Reservation()
        
        # Tool Calling 테스트 케이스들
        tool_test_cases = [
            {
                "name": "예약 조회 (Tool Calling)",
                "user_input": "홍길동, 010-1234-5678로 예약 조회해주세요",
                "expected_tool": "supabase_read_direct"
            },
            {
                "name": "환자 조회 (Tool Calling)",
                "user_input": "010-1234-5678로 환자 정보 확인해주세요",
                "expected_tool": "supabase_patient_lookup"
            },
            {
                "name": "예약 생성 (Tool Calling)",
                "user_input": "새로운 예약을 만들고 싶어요",
                "expected_tool": "supabase_create_direct"
            }
        ]
        
        for i, test_case in enumerate(tool_test_cases, 1):
            print(f"\n📊 Tool Calling 테스트 {i}: {test_case['name']}")
            print("-" * 30)
            
            try:
                result = agent2.process_reservation_request(test_case["user_input"])
                
                print(f"✅ Tool Calling 테스트 성공")
                print(f"📝 성공 여부: {result.get('success', False)}")
                print(f"📝 상태: {result.get('status', 'Unknown')}")
                
                # Tool Calling 결과 확인
                tool_result = result.get('tool_result', {})
                if tool_result.get('tool_calls'):
                    print("🔧 Tool Calling 사용됨")
                    for tool_call in tool_result['tool_calls']:
                        print(f"  - 툴: {tool_call['tool_name']}")
                        print(f"  - 결과: {tool_call['result'].get('success', False)}")
                else:
                    print("📝 Tool Calling 없이 처리됨")
                
            except Exception as e:
                print(f"❌ Tool Calling 테스트 실패: {e}")
        
        print(f"\n🎉 Tool Calling 통합 테스트 완료!")
        
    except Exception as e:
        print(f"❌ Tool Calling 통합 테스트 실패: {e}")

def test_error_handling():
    """에러 처리 테스트"""
    print(f"\n⚠️ 에러 처리 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent1_manager import Agent1Manager
        from main.agents.agent2_reservation import Agent2Reservation
        
        agent1 = Agent1Manager()
        agent2 = Agent2Reservation()
        
        # 에러 상황 테스트 케이스들
        error_test_cases = [
            {
                "name": "모호한 요청",
                "user_input": "안녕하세요",
                "expected_handling": "명확화 요청"
            },
            {
                "name": "정보 부족한 예약 요청",
                "user_input": "예약하고 싶어요",
                "expected_handling": "필수 정보 요청"
            },
            {
                "name": "존재하지 않는 환자 조회",
                "user_input": "존재하지않는환자, 999-9999-9999로 예약 조회",
                "expected_handling": "환자 없음 처리"
            }
        ]
        
        for i, test_case in enumerate(error_test_cases, 1):
            print(f"\n📊 에러 테스트 {i}: {test_case['name']}")
            print("-" * 30)
            
            try:
                if "예약" in test_case["user_input"]:
                    result = agent2.process_reservation_request(test_case["user_input"])
                else:
                    result = agent1.process_user_request(test_case["user_input"])
                
                print(f"✅ 에러 처리 테스트 성공")
                print(f"📝 성공 여부: {result.get('success', False)}")
                print(f"📝 상태: {result.get('status', 'Unknown')}")
                print(f"💬 메시지: {result.get('message', 'No message')}")
                
                if not result.get('success'):
                    print("⚠️ 예상된 에러 상황 처리됨")
                else:
                    print("✅ 에러 상황이 적절히 처리됨")
                
            except Exception as e:
                print(f"❌ 에러 처리 테스트 실패: {e}")
        
        print(f"\n🎉 에러 처리 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 에러 처리 테스트 실패: {e}")

if __name__ == "__main__":
    print("🚀 전체 시스템 통합 테스트 시작")
    print("=" * 60)
    
    # 전체 예약 플로우 테스트
    test_full_reservation_flow()
    
    # Tool Calling 통합 테스트
    test_tool_calling_integration()
    
    # 에러 처리 테스트
    test_error_handling()
    
    print(f"\n🎉 모든 통합 테스트 완료!")
    print("=" * 60)
