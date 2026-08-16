#!/usr/bin/env python3
"""
RAG 에이전트 간단 테스트
"""

import os
import sys
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# RAG 에이전트 경로 추가
rag_agent_path = os.path.join(os.path.dirname(__file__), "rag_doctor_agent")
if rag_agent_path not in sys.path:
    sys.path.insert(0, rag_agent_path)

def test_rag_agent():
    """RAG 에이전트 직접 테스트"""
    try:
        print("🚀 RAG 에이전트 직접 테스트")
        print("=" * 50)
        
        # RAG 에이전트 임포트
        from rag_doctor_agent.main.agent.graph import build_and_run_agent
        from rag_doctor_agent.main.agent.output_enforcer import OutputSchema
        
        print("✅ RAG 에이전트 임포트 성공")
        
        # 테스트 입력
        test_input = {
            "symptoms": ["어깨 통증"],
            "additional_info": "어깨가 아파서 예약하고 싶어요",
            "query": "증상: 어깨 통증. 어깨가 아파서 예약하고 싶어요"
        }
        
        print(f"📝 테스트 입력: {test_input}")
        
        # RAG 에이전트 실행
        result = build_and_run_agent(test_input)
        
        print("✅ RAG 에이전트 실행 성공")
        print(f"📝 결과: {result}")
        
        # 결과 저장
        os.makedirs("out", exist_ok=True)
        with open("out/rag_test_result.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2))
        
        print("✅ 결과 저장 완료: out/rag_test_result.json")
        
    except Exception as e:
        print(f"❌ RAG 에이전트 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_agent()
