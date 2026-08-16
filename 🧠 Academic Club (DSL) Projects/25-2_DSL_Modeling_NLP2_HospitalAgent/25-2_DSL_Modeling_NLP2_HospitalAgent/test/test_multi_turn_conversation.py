#!/usr/bin/env python3
"""
다중 턴 대화형 예약 테스트 스크립트
- 여러 턴에 걸쳐 사용자와 대화하며 정보 수집
- 예약 생성까지 완전한 시나리오 테스트
"""

import os
import sys
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multi_turn_reservation():
    """다중 턴 예약 대화 테스트"""
    print("🚀 다중 턴 대화형 예약 테스트 시작")
    print("=" * 60)
    
    try:
        # Agent1과 Agent2 임포트
        from main.agents.agent1_manager import Agent1Manager
        from main.agents.agent2_reservation import Agent2Reservation
        
        # 에이전트 초기화
        agent1 = Agent1Manager()
        agent2 = Agent2Reservation()
        
        print("✅ 에이전트 초기화 완료")
        print()
        
        # 대화 시나리오 정의
        conversation_scenarios = [
            # {
            #     "name": "시나리오 1: 기본 예약 (정보 누락 → 추가 요청 → 완료)",
            #     "turns": [
            #         {
            #             "user_input": "홍길동으로 예약하고 싶어요",
            #             "expected_intent": "reservation",
            #             "expected_missing": ["전화번호"],
            #             "description": "1턴: 이름만 제공, 전화번호 누락"
            #         },
            #         {
            #             "user_input": "010-1234-5678",
            #             "expected_intent": "reservation", 
            #             "expected_missing": [],
            #             "description": "2턴: 전화번호 제공, 예약 완료"
            #         }
            #     ]
            # },
            # {
            #     "name": "시나리오 2: 증상 포함 예약 (증상 → 의료진 추천 → 예약)",
            #     "turns": [
            #         {
            #             "user_input": "어깨가 아파서 예약하고 싶어요",
            #             "expected_intent": "reservation",
            #             "expected_missing": ["환자명", "전화번호"],
            #             "description": "1턴: 증상만 제공, 기본 정보 누락"
            #         },
            #         {
            #             "user_input": "김철수, 010-5678-9012",
            #             "expected_intent": "reservation",
            #             "expected_missing": [],
            #             "description": "2턴: 기본 정보 제공, 예약 완료"
            #         }
            #     ]
            # },
            # {
            #     "name": "시나리오 3: 단계별 정보 수집",
            #     "turns": [
            #         {
            #             "user_input": "예약하고 싶어요",
            #             "expected_intent": "reservation",
            #             "expected_missing": ["환자명", "전화번호"],
            #             "description": "1턴: 예약 의도만 표현"
            #         },
            #         {
            #             "user_input": "이름은 박영희",
            #             "expected_intent": "reservation",
            #             "expected_missing": ["전화번호"],
            #             "description": "2턴: 이름만 제공"
            #         },
            #         {
            #             "user_input": "전화번호는 010-9876-5432",
            #             "expected_intent": "reservation",
            #             "expected_missing": [],
            #             "description": "3턴: 전화번호 제공, 예약 완료"
            #         }
            #     ]
            # }
            
            # 새로운 고도화된 시나리오들
            {
                "name": "시나리오 1: 완전한 증상-의료진 매핑 예약",
                "turns": [
                    {
                        "user_input": "무릎이 아파서 예약하고 싶어요",
                        "expected_intent": "reservation",
                        "expected_missing": ["환자명", "전화번호"],
                        "description": "1턴: 증상만 제공, 기본 정보 누락"
                    },
                    {
                        "user_input": "이름은 김철수입니다",
                        "expected_intent": "reservation",
                        "expected_missing": ["전화번호"],
                        "description": "2턴: 이름 제공"
                    },
                    {
                        "user_input": "전화번호는 010-1234-5678",
                        "expected_intent": "reservation",
                        "expected_missing": [],
                        "description": "3턴: 전화번호 제공, 의료진 추천 및 예약 완료"
                    }
                ]
            },
            {
                "name": "시나리오 2: 복합 증상 의료진 매핑",
                "turns": [
                    {
                        "user_input": "허리가 아프고 디스크가 있어서 예약하고 싶어요",
                        "expected_intent": "reservation",
                        "expected_missing": ["환자명", "전화번호"],
                        "description": "1턴: 복합 증상 제공"
                    },
                    {
                        "user_input": "박영희, 010-5678-9012",
                        "expected_intent": "reservation",
                        "expected_missing": [],
                        "description": "2턴: 기본 정보 제공, 척추 전문의 추천 및 예약"
                    }
                ]
            },
            {
                "name": "시나리오 3: 신경과 증상 매핑",
                "turns": [
                    {
                        "user_input": "두통과 어지럼증이 심해서 예약하고 싶어요",
                        "expected_intent": "reservation",
                        "expected_missing": ["환자명", "전화번호"],
                        "description": "1턴: 신경과 증상 제공"
                    },
                    {
                        "user_input": "이름은 이민호, 전화번호는 010-9876-5432",
                        "expected_intent": "reservation",
                        "expected_missing": [],
                        "description": "2턴: 기본 정보 제공, 신경과 전문의 추천 및 예약"
                    }
                ]
            },
            {
                "name": "시나리오 4: 응급 상황 매핑",
                "turns": [
                    {
                        "user_input": "급성 통증이 심해서 응급실에 가야 할 것 같아요",
                        "expected_intent": "reservation",
                        "expected_missing": ["환자명", "전화번호"],
                        "description": "1턴: 응급 상황 증상 제공"
                    },
                    {
                        "user_input": "정수진, 010-1111-2222",
                        "expected_intent": "reservation",
                        "expected_missing": [],
                        "description": "2턴: 기본 정보 제공, 응급의학과 추천 및 예약"
                    }
                ]
            },
            {
                "name": "시나리오 5: 단계별 증상 상세화",
                "turns": [
                    {
                        "user_input": "어깨가 아파요",
                        "expected_intent": "symptom_doctor",
                        "expected_missing": [],
                        "description": "1턴: 증상만 제공, 의료진 추천 요청"
                    },
                    {
                        "user_input": "그럼 예약하고 싶어요",
                        "expected_intent": "reservation",
                        "expected_missing": ["환자명", "전화번호"],
                        "description": "2턴: 예약 의도 표현"
                    },
                    {
                        "user_input": "최지훈, 010-3333-4444",
                        "expected_intent": "reservation",
                        "expected_missing": [],
                        "description": "3턴: 기본 정보 제공, 정형외과 전문의 추천 및 예약"
                    }
                ]
            },
            {
                "name": "시나리오 6: 모호한 증상 → 일반의 추천",
                "turns": [
                    {
                        "user_input": "몸이 아프고 구체적으로 뭐가 아픈지 모르겠어요",
                        "expected_intent": "reservation",
                        "expected_missing": ["환자명", "전화번호"],
                        "description": "1턴: 모호한 증상 제공"
                    },
                    {
                        "user_input": "김영희, 010-5555-6666",
                        "expected_intent": "reservation",
                        "expected_missing": [],
                        "description": "2턴: 기본 정보 제공, 일반의 추천 및 예약"
                    }
                ]
            }
        ]
        
        # 각 시나리오 테스트
        for scenario_idx, scenario in enumerate(conversation_scenarios, 1):
            print(f"📋 {scenario['name']}")
            print("-" * 50)
            
            # 대화 컨텍스트 초기화
            conversation_context = {}
            collected_info = {}
            
            for turn_idx, turn in enumerate(scenario["turns"], 1):
                print(f"\n🔄 턴 {turn_idx}: {turn['description']}")
                print(f"👤 사용자: {turn['user_input']}")
                
                # Agent1으로 의도 분석 및 라우팅 (컨텍스트 포함)
                routing_result = agent1.route_to_agent_or_tool(turn['user_input'], conversation_context)
                
                if routing_result.get("primary_intent") == "reservation":
                    print(f"🤖 Agent1: 예약 관련 요청으로 인식")
                    
                    # Agent2로 예약 처리 (기존 컨텍스트 전달)
                    reservation_result = agent2.process_reservation_request(
                        turn['user_input'], 
                        collected_info
                    )
                    
                    # 결과 분석
                    if reservation_result.get("success"):
                        print(f"✅ 예약 처리 성공")
                        print(f"📝 수집된 정보: {reservation_result.get('collected_info', {})}")
                        print(f"💬 응답: {reservation_result.get('message', 'No message')}")
                        
                        # 컨텍스트 업데이트
                        collected_info.update(reservation_result.get('collected_info', {}))
                        conversation_context.update(collected_info)
                        
                    else:
                        print(f"⚠️ 예약 처리 실패")
                        print(f"📝 상태: {reservation_result.get('status', 'Unknown')}")
                        print(f"💬 응답: {reservation_result.get('message', 'No message')}")
                        
                        # 누락된 정보가 있으면 컨텍스트에 추가
                        if reservation_result.get("collected_info"):
                            collected_info.update(reservation_result.get("collected_info"))
                            conversation_context.update(collected_info)
                
                elif routing_result.get("primary_intent") == "symptom_doctor":
                    print(f"🤖 Agent1: 증상-의료진 매핑 요청으로 인식")
                    
                    # Agent3 RAG로 의료진 추천
                    from main.agents.agent3_rag import Agent3RAG
                    agent3 = Agent3RAG()
                    
                    # 증상 추출
                    symptoms = routing_result.get("extracted_info", {}).get("symptoms", [])
                    if not symptoms:
                        # 사용자 입력에서 증상 추출 시도
                        symptoms = [turn['user_input']]
                    
                    print(f"🔍 추출된 증상: {symptoms}")
                    
                    # RAG로 의료진 추천
                    rag_result = agent3.recommend_doctors(symptoms, turn['user_input'])
                    
                    if rag_result.get("success"):
                        print(f"✅ 의료진 추천 성공")
                        print(f"📝 추천 진료과: {rag_result.get('department', 'Unknown')}")
                        print(f"👨‍⚕️ 추천 의료진 수: {len(rag_result.get('recommended_doctors', []))}")
                        print(f"📊 신뢰도: {rag_result.get('confidence', 0.0)}")
                        print(f"💭 추천 근거: {rag_result.get('reasoning', 'No reasoning')}")
                        
                        # 추천 의료진 상세 정보 출력
                        for i, doctor in enumerate(rag_result.get('recommended_doctors', [])[:3], 1):
                            print(f"   {i}. {doctor.get('name', 'Unknown')} - {doctor.get('department', 'Unknown')}")
                            print(f"      추천 근거: {doctor.get('reasoning', 'No reasoning')}")
                        
                        # 컨텍스트에 의료진 정보 추가
                        conversation_context.update({
                            "recommended_doctors": rag_result.get('recommended_doctors', []),
                            "recommended_department": rag_result.get('department', ''),
                            "rag_confidence": rag_result.get('confidence', 0.0)
                        })
                        
                    else:
                        print(f"⚠️ 의료진 추천 실패")
                        print(f"💬 응답: {rag_result.get('message', 'No message')}")
                
                else:
                    print(f"❌ 예상과 다른 의도: {routing_result.get('primary_intent')}")
                    print(f"💬 응답: {routing_result.get('message', 'No message')}")
                
                print("-" * 30)
            
            # 시나리오 완료 요약
            print(f"\n📊 시나리오 {scenario_idx} 완료")
            print(f"📝 최종 수집 정보: {collected_info}")
            print(f"🔍 누락된 정보: {agent2._check_missing_information(collected_info)}")
            print("=" * 60)
        
        print("\n🎉 모든 다중 턴 대화 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

def test_conversation_flow():
    """대화 흐름 테스트"""
    print("\n🔄 대화 흐름 테스트")
    print("=" * 40)
    
    try:
        from main.agents.agent2_reservation import Agent2Reservation
        
        agent2 = Agent2Reservation()
        
        # 시뮬레이션 대화
        conversation = [
            "홍길동으로 예약하고 싶어요",
            "010-1234-5678",
            "어깨가 아파요",
            "내일 오후 2시에 가능한가요?"
        ]
        
        collected_info = {}
        
        for i, user_input in enumerate(conversation, 1):
            print(f"\n🔄 턴 {i}: {user_input}")
            
            result = agent2.process_reservation_request(user_input, collected_info)
            
            print(f"📝 상태: {result.get('status', 'Unknown')}")
            print(f"💬 응답: {result.get('message', 'No message')}")
            
            if result.get("collected_info"):
                collected_info.update(result["collected_info"])
                print(f"📋 수집된 정보: {collected_info}")
            
            if result.get("success"):
                print("✅ 예약 처리 완료!")
                break
            else:
                missing = agent2._check_missing_information(collected_info)
                if missing:
                    print(f"⚠️ 누락된 정보: {missing}")
                else:
                    print("🔄 추가 정보 수집 중...")
        
    except Exception as e:
        print(f"❌ 대화 흐름 테스트 오류: {e}")

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n🧪 엣지 케이스 테스트")
    print("=" * 40)
    
    try:
        from main.agents.agent1_manager import Agent1Manager
        from main.agents.agent2_reservation import Agent2Reservation
        
        agent1 = Agent1Manager()
        agent2 = Agent2Reservation()
        
        edge_cases = [
            {
                "input": "안녕하세요",
                "description": "인사말",
                "expected_intent": "greeting"
            },
            {
                "input": "예약하고 싶은데...",
                "description": "불완전한 요청",
                "expected_intent": "reservation"
            },
            {
                "input": "홍길동, 010-1234-5678, 어깨가 아파요",
                "description": "모든 정보를 한 번에 제공 (증상 포함)",
                "expected_intent": "symptom_doctor"
            },
            {
                "input": "예약 취소하고 싶어요",
                "description": "예약 취소 요청",
                "expected_intent": "reservation"
            },
            {
                "input": "무릎이 아프고 허리도 아파요",
                "description": "복합 증상 의료진 추천",
                "expected_intent": "symptom_doctor"
            },
            {
                "input": "병원 휴무일이 언제인가요?",
                "description": "병원 정보 요청",
                "expected_intent": "hospital_info"
            }
        ]
        
        for case in edge_cases:
            print(f"\n🔍 테스트: {case['description']}")
            print(f"👤 입력: {case['input']}")
            
            # Agent1 라우팅
            routing_result = agent1.route_to_agent_or_tool(case['input'])
            actual_intent = routing_result.get('primary_intent', 'Unknown')
            expected_intent = case.get('expected_intent', 'Unknown')
            
            print(f"🤖 의도: {actual_intent}")
            
            # 의도 검증
            if actual_intent == expected_intent:
                print(f"✅ 의도 분석 정확")
            else:
                print(f"❌ 의도 분석 오류: 예상 {expected_intent}, 실제 {actual_intent}")
            
            # 의도별 처리
            if actual_intent == "reservation":
                # Agent2 처리
                result = agent2.process_reservation_request(case['input'])
                print(f"📝 상태: {result.get('status', 'Unknown')}")
                print(f"💬 응답: {result.get('message', 'No message')}")
                
            elif actual_intent == "symptom_doctor":
                # Agent3 RAG 처리
                from main.agents.agent3_rag import Agent3RAG
                agent3 = Agent3RAG()
                
                symptoms = routing_result.get("extracted_info", {}).get("symptoms", [case['input']])
                rag_result = agent3.recommend_doctors(symptoms, case['input'])
                
                if rag_result.get("success"):
                    print(f"✅ 의료진 추천 성공")
                    print(f"📝 추천 진료과: {rag_result.get('department', 'Unknown')}")
                    print(f"👨‍⚕️ 추천 의료진 수: {len(rag_result.get('recommended_doctors', []))}")
                    
                    # 상위 3명 의료진 출력
                    for i, doctor in enumerate(rag_result.get('recommended_doctors', [])[:3], 1):
                        print(f"   {i}. {doctor.get('name', 'Unknown')} - {doctor.get('department', 'Unknown')}")
                else:
                    print(f"⚠️ 의료진 추천 실패")
                    print(f"💬 응답: {rag_result.get('message', 'No message')}")
                    
            elif actual_intent == "hospital_info":
                # 병원 정보 검색
                print(f"🏥 병원 정보 검색 중...")
                print(f"💬 응답: {routing_result.get('message', 'No message')}")
                
            else:
                print(f"💬 응답: {routing_result.get('message', 'No message')}")
            
            print("-" * 30)
        
    except Exception as e:
        print(f"❌ 엣지 케이스 테스트 오류: {e}")

def main():
    """메인 테스트 실행"""
    print("🚀 다중 턴 대화형 예약 시스템 테스트")
    print("=" * 60)
    
    # 환경 변수 확인
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    if not os.getenv('SUPABASE_URL'):
        print("⚠️ SUPABASE_URL이 설정되지 않았습니다.")
        return
    
    # 테스트 실행
    test_multi_turn_reservation()
    test_conversation_flow()
    test_edge_cases()
    
    print("\n🎉 모든 테스트 완료!")

if __name__ == "__main__":
    main()

