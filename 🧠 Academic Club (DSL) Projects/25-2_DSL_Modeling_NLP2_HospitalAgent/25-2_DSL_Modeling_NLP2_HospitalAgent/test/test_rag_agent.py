#!/usr/bin/env python3
"""
RAG Agent 테스트 스크립트
- Agent3 (RAG)가 증상을 분석하고 적절한 의료진을 추천하는지 테스트
"""
import os
import sys
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rag_agent():
    """RAG Agent 기본 테스트"""
    print("🧪 RAG Agent 테스트")
    print("=" * 50)
    
    try:
        # Agent3 임포트
        from main.agents.agent3_rag import Agent3RAG
        print("✅ Agent3 임포트 성공")
        
        # Agent3 생성
        agent3 = Agent3RAG()
        print("✅ Agent3 생성 완료")
        
        # 테스트 케이스들
        test_cases = [
            {
                "name": "관절 관련 증상",
                "symptoms": ["무릎 통증", "관절 부종"],
                "additional_info": "운동 후 통증이 심해짐",
                "expected_department": "관절센터"
            },
            {
                "name": "척추 관련 증상",
                "symptoms": ["허리 통증", "디스크"],
                "additional_info": "장시간 앉아있으면 통증",
                "expected_department": "척추센터"
            },
            {
                "name": "소화기 관련 증상",
                "symptoms": ["복통", "소화불량"],
                "additional_info": "식후 복부 불편감",
                "expected_department": "내과"
            },
            {
                "name": "신경 관련 증상",
                "symptoms": ["두통", "어지럼증"],
                "additional_info": "스트레스로 인한 두통",
                "expected_department": "뇌신경센터"
            },
            {
                "name": "응급 상황",
                "symptoms": ["급성 통증", "외상"],
                "additional_info": "사고로 인한 급성 통증",
                "expected_department": "응급의학센터"
            }
        ]
        
        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 테스트 {i}: {test_case['name']}")
            print("-" * 30)
            
            try:
                # RAG Agent로 의료진 추천 요청
                result = agent3.recommend_doctors(
                    symptoms=test_case["symptoms"],
                    additional_info=test_case["additional_info"]
                )
                
                print(f"✅ 처리 성공")
                print(f"📝 성공 여부: {result.get('success', False)}")
                print(f"📝 추천 의료진: {len(result.get('recommended_doctors', []))}명")
                print(f"📝 추천 진료과: {result.get('department', 'Unknown')}")
                print(f"📝 신뢰도: {result.get('confidence', 0.0)}")
                print(f"📝 추천 근거: {result.get('reasoning', 'No reasoning')}")
                
                # 추천 의료진 상세 정보
                if result.get('recommended_doctors'):
                    print("👨‍⚕️ 추천 의료진:")
                    for j, doctor in enumerate(result.get('recommended_doctors', []), 1):
                        print(f"  {j}. {doctor.get('name', 'Unknown')} - {doctor.get('specialty', 'Unknown')}")
                        print(f"     진료과: {doctor.get('department', 'Unknown')}")
                        print(f"     추천 근거: {doctor.get('reasoning', 'No reasoning')}")
                
                # 예상 결과와 비교
                if result.get('department') == test_case['expected_department']:
                    print("✅ 진료과 추천 정확")
                else:
                    print(f"⚠️ 진료과 추천 오류: 예상 {test_case['expected_department']}, 실제 {result.get('department')}")
                
                # 대안 진료과 확인
                if result.get('alternative_options'):
                    print(f"🔄 대안 진료과: {', '.join(result.get('alternative_options', []))}")
                
            except Exception as e:
                print(f"❌ 테스트 실행 중 오류: {e}")
        
        print(f"\n🎉 RAG Agent 테스트 완료!")
        
    except Exception as e:
        print(f"❌ RAG Agent 테스트 실패: {e}")

def test_rag_pipeline_integration():
    """RAG 파이프라인 통합 테스트"""
    print(f"\n🔗 RAG 파이프라인 통합 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent3_rag import Agent3RAG
        agent3 = Agent3RAG()
        
        # 복합 증상 테스트
        complex_symptoms = ["무릎 통증", "허리 통증", "두통"]
        additional_info = "다양한 증상이 동시에 발생"
        
        print(f"📝 복합 증상: {', '.join(complex_symptoms)}")
        print(f"📝 추가 정보: {additional_info}")
        
        result = agent3.recommend_doctors(
            symptoms=complex_symptoms,
            additional_info=additional_info
        )
        
        print(f"✅ 통합 테스트 성공")
        print(f"📝 추천 진료과: {result.get('department', 'Unknown')}")
        print(f"📝 추천 의료진 수: {len(result.get('recommended_doctors', []))}")
        
        if result.get('success'):
            print("✅ RAG 파이프라인 정상 작동")
        else:
            print("⚠️ RAG 파이프라인 오류")
            
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")

def test_rag_fallback():
    """RAG 폴백 테스트"""
    print(f"\n🔄 RAG 폴백 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent3_rag import Agent3RAG
        agent3 = Agent3RAG()
        
        # 모호한 증상 테스트
        vague_symptoms = ["몸이 아파요"]
        additional_info = "구체적인 증상을 모르겠음"
        
        print(f"📝 모호한 증상: {', '.join(vague_symptoms)}")
        print(f"📝 추가 정보: {additional_info}")
        
        result = agent3.recommend_doctors(
            symptoms=vague_symptoms,
            additional_info=additional_info
        )
        
        print(f"✅ 폴백 테스트 성공")
        print(f"📝 성공 여부: {result.get('success', False)}")
        print(f"📝 추천 진료과: {result.get('department', 'Unknown')}")
        
        if result.get('success'):
            print("✅ 폴백 로직 정상 작동")
        else:
            print("⚠️ 폴백 로직 오류")
            
    except Exception as e:
        print(f"❌ 폴백 테스트 실패: {e}")

def test_agent2_agent3_integration():
    """Agent2 → Agent3 통합 테스트"""
    print(f"\n🔗 Agent2 → Agent3 통합 테스트")
    print("-" * 30)
    
    try:
        from main.agents.agent2_reservation import Agent2Reservation
        agent2 = Agent2Reservation()
        
        # 증상이 포함된 예약 요청
        test_input = "어깨가 아파서 예약하고 싶어요"
        collected_info = {
            "환자명": "김철수",
            "전화번호": "010-1234-5678",
            "symptoms": "어깨 통증"
        }
        
        print(f"📝 예약 요청: {test_input}")
        print(f"📝 수집된 정보: {collected_info}")
        
        result = agent2.process_reservation_request(test_input, collected_info)
        
        print(f"✅ 통합 테스트 성공")
        print(f"📝 성공 여부: {result.get('success', False)}")
        print(f"📝 상태: {result.get('status', 'Unknown')}")
        
        # Agent3 호출 결과 확인
        if result.get('collected_info', {}).get('recommended_department'):
            print(f"✅ Agent3 호출 성공")
            print(f"📝 추천 진료과: {result.get('collected_info', {}).get('recommended_department')}")
            print(f"📝 추천 의료진: {result.get('collected_info', {}).get('recommended_doctor')}")
        else:
            print("⚠️ Agent3 호출 실패 또는 증상 없음")
            
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")

if __name__ == "__main__":
    print("🚀 RAG Agent 테스트 시작")
    print("=" * 60)
    
    # 기본 RAG 테스트
    test_rag_agent()
    
    # RAG 파이프라인 통합 테스트
    test_rag_pipeline_integration()
    
    # RAG 폴백 테스트
    test_rag_fallback()
    
    # Agent2 → Agent3 통합 테스트
    test_agent2_agent3_integration()
    
    print(f"\n🎉 모든 RAG 테스트 완료!")
