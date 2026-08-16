#!/usr/bin/env python3
"""
Tool Calling 테스트 스크립트
"""
import os
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def test_tool_calling():
    """Tool Calling 테스트"""
    print("🧪 Tool Calling 테스트")
    print("=" * 50)
    
    try:
        # 에이전트2 임포트
        from main.agents.agent2_reservation import Agent2Reservation
        
        print("✅ 에이전트2 임포트 성공")
        
        # 에이전트 생성
        agent = Agent2Reservation()
        print("✅ 에이전트2 생성 완료")
        
        # Tool Calling 설정 확인
        if agent.llm_with_tools:
            print("✅ Tool Calling 설정 완료")
            print(f"📋 바인딩된 도구 수: {len(agent.tools)}")
            
            # 도구 목록 출력
            for i, tool in enumerate(agent.tools):
                print(f"  {i+1}. {tool.name}: {tool.description}")
        else:
            print("❌ Tool Calling 설정 실패")
            print("📝 LLM 클라이언트와 도구 설정을 확인해주세요")
            return
        
        # 테스트 케이스들
        test_cases = [
            {
                "name": "예약 조회 테스트",
                "input": "내 예약을 확인해주세요",
                "info": {
                    "patient_name": "홍길동",
                    "patient_phone": "010-1234-5678"
                }
            },
            {
                "name": "예약 생성 테스트",
                "input": "예약하고 싶어요",
                "info": {
                    "patient_name": "김철수",
                    "patient_gender": "남",
                    "patient_phone": "010-9876-5432",
                    "symptoms": "무릎 통증"
                }
            },
            {
                "name": "예약 수정 테스트",
                "input": "예약 시간을 변경하고 싶어요",
                "info": {
                    "patient_name": "이영희",
                    "patient_phone": "010-5555-6666"
                }
            }
        ]
        
        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases):
            print(f"\n📊 테스트 {i+1}: {test_case['name']}")
            print("-" * 30)
            
            try:
                # Tool Calling을 사용한 처리
                result = agent.process_reservation_request(
                    test_case["input"],
                    test_case["info"]
                )
                
                if result.get("success"):
                    print("✅ 처리 성공")
                    print(f"📝 상태: {result.get('status')}")
                    print(f"💬 메시지: {result.get('message')}")
                    
                    # Tool Calling 결과 확인
                    tool_result = result.get("tool_result", {})
                    if tool_result.get("tool_calls"):
                        print("🔧 Tool Calling 사용됨")
                        for tool_call in tool_result["tool_calls"]:
                            print(f"  - 툴: {tool_call['tool_name']}")
                            print(f"  - 인수: {tool_call['tool_args']}")
                            print(f"  - 결과: {tool_call['result'].get('success', False)}")
                    else:
                        print("📝 Tool Calling 없이 처리됨")
                else:
                    print("❌ 처리 실패")
                    print(f"📝 오류: {result.get('error')}")
                    print(f"💬 메시지: {result.get('message')}")
                    
            except Exception as e:
                print(f"❌ 테스트 실행 중 오류: {e}")
        
        # Tool Calling 기능 테스트
        print(f"\n🔧 Tool Calling 기능 테스트")
        print("-" * 30)
        
        try:
            # 직접 Tool Calling 테스트
            test_input = "예약을 조회해주세요"
            test_info = {
                "patient_name": "테스트환자",
                "patient_phone": "010-0000-0000"
            }
            
            tool_result = agent._execute_with_tool_calling(test_input, test_info)
            
            if tool_result.get("success"):
                print("✅ Tool Calling 실행 성공")
                if tool_result.get("tool_calls"):
                    print("🔧 툴 호출됨")
                    for tool_call in tool_result["tool_calls"]:
                        print(f"  - 툴: {tool_call['tool_name']}")
                        print(f"  - 결과: {tool_call['result'].get('success', False)}")
                else:
                    print("📝 툴 호출 없음 (텍스트 응답)")
            else:
                print("❌ Tool Calling 실행 실패")
                print(f"📝 오류: {tool_result.get('error')}")
                
        except Exception as e:
            print(f"❌ Tool Calling 테스트 오류: {e}")
        
        print(f"\n🎉 Tool Calling 테스트 완료!")
        
    except ImportError as e:
        print(f"❌ 임포트 실패: {e}")
        print("📝 필요한 의존성을 설치해주세요:")
        print("pip install langchain-mcp-adapters")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")

def test_tool_binding():
    """도구 바인딩 테스트"""
    print("\n🔗 도구 바인딩 테스트")
    print("=" * 30)
    
    try:
        from main.tools.supabase_mcp_tool import get_supabase_tools_for_binding
        
        # 도구 가져오기
        tools = get_supabase_tools_for_binding()
        print(f"✅ {len(tools)}개 도구 가져오기 성공")
        
        # 각 도구 정보 출력
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool.name}")
            print(f"     설명: {tool.description}")
            print(f"     타입: {type(tool).__name__}")
        
        # LLM 바인딩 테스트
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                model="gpt-4o-mini",
                temperature=0.1
            )
            
            # 도구 바인딩
            llm_with_tools = llm.bind_tools(tools)
            print("✅ LLM 도구 바인딩 성공")
            
            # 간단한 테스트
            response = llm_with_tools.invoke("예약을 조회해주세요")
            print(f"📝 LLM 응답: {response.content[:100]}...")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print("🔧 툴 호출 감지됨")
                for tool_call in response.tool_calls:
                    if hasattr(tool_call, 'name'):
                        print(f"  - 툴: {tool_call.name}")
                        print(f"  - 인수: {tool_call.args}")
                    else:
                        print(f"  - 툴: {tool_call.get('name', 'Unknown')}")
                        print(f"  - 인수: {tool_call.get('args', {})}")
            else:
                print("📝 툴 호출 없음")
                
        except Exception as e:
            print(f"❌ LLM 바인딩 테스트 실패: {e}")
        
    except Exception as e:
        print(f"❌ 도구 바인딩 테스트 오류: {e}")

if __name__ == "__main__":
    test_tool_calling()
    test_tool_binding()
