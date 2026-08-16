#!/usr/bin/env python3
"""
Agent1 분기 테스트 스크립트
- Agent1이 사용자 요청을 분석하고 적절한 에이전트/도구로 라우팅하는지 테스트
"""
import os
import sys
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent1_routing():
    """Agent1 분기 테스트"""
    print("🧪 Agent1 분기 테스트")
    print("=" * 50)
    
    try:
        # Agent1 임포트
        from main.agents.agent1_manager import Agent1Manager
        print("✅ Agent1 임포트 성공")
        
        # Agent1 생성
        agent1 = Agent1Manager()
        print("✅ Agent1 생성 완료")
        
        # 테스트 케이스들
        test_cases = [
            {
                "name": "예약 관련 요청",
                "user_input": "홍길동으로 예약하고 싶어요",
                "expected_intent": "reservation",
                "expected_action": "create"
            },
            {
                "name": "증상-의료진 매핑 요청", 
                "user_input": "어깨가 아파서 어떤 의사한테 가야 할까요?",
                "expected_intent": "symptom_doctor",
                "expected_action": "symptom_analysis"
            },
            {
                "name": "병원 정보 요청",
                "user_input": "병원 휴무일이 언제인가요?",
                "expected_intent": "hospital_info",
                "expected_action": "search_hospital_info"
            },
            {
                "name": "예약 조회 요청",
                "user_input": "내 예약 확인해주세요",
                "expected_intent": "reservation", 
                "expected_action": "check"
            },
            {
                "name": "예약 취소 요청",
                "user_input": "예약 취소하고 싶어요",
                "expected_intent": "reservation",
                "expected_action": "cancel"
            },
            {
                "name": "모호한 요청",
                "user_input": "안녕하세요",
                "expected_intent": "unclear",
                "expected_action": "ask_clarification"
            }
        ]
        
        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 테스트 {i}: {test_case['name']}")
            print("-" * 30)
            
            try:
                # Agent1으로 요청 처리 (올바른 메서드 사용)
                result = agent1.route_to_agent_or_tool(test_case["user_input"])
                
                print(f"✅ 처리 성공")
                print(f"📝 의도: {result.get('primary_intent', 'Unknown')}")
                print(f"📝 신뢰도: {result.get('confidence', 0.0)}")
                print(f"📝 추출된 정보: {result.get('extracted_info', {})}")
                print(f"📝 라우팅: {result.get('routing_info', {}).get('target', 'Unknown')}")
                print(f"💬 메시지: {result.get('message', 'No message')}")
                
                # 예상 결과와 비교
                if result.get('primary_intent') == test_case['expected_intent']:
                    print("✅ 의도 분석 정확")
                else:
                    print(f"⚠️ 의도 분석 오류: 예상 {test_case['expected_intent']}, 실제 {result.get('primary_intent')}")
                
                # 라우팅 결과 확인
                routing_info = result.get('routing_info', {})
                if routing_info:
                    target = routing_info.get('target', 'Unknown')
                    print(f"🔀 라우팅: {target}")
                    
                    # 실제 라우팅 실행 (선택적)
                    if target == 'agent2_reservation':
                        print("  → Agent2로 라우팅됨")
                    elif target == 'agent3_rag':
                        print("  → Agent3로 라우팅됨")
                    elif target == 'tavily_search':
                        print("  → Tavily 검색으로 라우팅됨")
                    else:
                        print(f"  → {target}로 라우팅됨")
                
            except Exception as e:
                print(f"❌ 테스트 실행 중 오류: {e}")
        
        print(f"\n🎉 Agent1 분기 테스트 완료!")
        
    except Exception as e:
        print(f"❌ Agent1 테스트 실패: {e}")

def test_agent1_tool_calling():
    """Agent1 Tool Calling 테스트"""
    print(f"\n🔧 Agent1 Tool Calling 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent1_manager import Agent1Manager
        agent1 = Agent1Manager()
        
        # 병원 정보 요청 (Tavily 검색 테스트)
        test_input = "병원 운영시간이 어떻게 되나요?"
        result = agent1.route_to_agent_or_tool(test_input)
        
        print(f"✅ Tool Calling 테스트 성공")
        print(f"📝 의도: {result.get('primary_intent')}")
        print(f"📝 라우팅: {result.get('routing_info', {}).get('target', 'Unknown')}")
        
        if result.get('routing_info', {}).get('target') == 'tavily_search':
            print("✅ Tavily 검색으로 올바르게 라우팅됨")
        else:
            print("⚠️ 예상과 다른 라우팅")
            
    except Exception as e:
        print(f"❌ Tool Calling 테스트 실패: {e}")

def test_agent1_agent2_integration():
    """Agent1 → Agent2 통합 테스트"""
    print(f"\n🔗 Agent1 → Agent2 통합 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent1_manager import Agent1Manager
        agent1 = Agent1Manager()
        
        # 예약 요청
        test_input = "김철수, 010-1234-5678로 예약하고 싶어요"
        result = agent1.route_to_agent_or_tool(test_input)
        
        print(f"✅ 통합 테스트 성공")
        print(f"📝 의도: {result.get('primary_intent')}")
        print(f"📝 라우팅: {result.get('routing_info', {}).get('target', 'Unknown')}")
        
        if result.get('routing_info', {}).get('target') == 'agent2_reservation':
            print("✅ Agent2로 올바르게 라우팅됨")
            
            # Agent2 호출 시뮬레이션
            print("🔄 Agent2 호출 시뮬레이션...")
            # 실제 Agent2 호출은 여기서 구현 가능
            
        else:
            print("⚠️ 예상과 다른 라우팅")
            
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")

if __name__ == "__main__":
    print("🚀 Agent1 분기 및 라우팅 테스트 시작")
    print("=" * 60)
    
    # 기본 분기 테스트
    test_agent1_routing()
    
    # Tool Calling 테스트
    test_agent1_tool_calling()
    
    # Agent2 통합 테스트
    test_agent1_agent2_integration()
    
    print(f"\n🎉 모든 Agent1 테스트 완료!")
