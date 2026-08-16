#!/usr/bin/env python3
"""
LangChain MCP Adapters 테스트 스크립트
"""
import os
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def test_langchain_mcp_adapters():
    """LangChain MCP Adapters 테스트"""
    print("🧪 LangChain MCP Adapters 테스트")
    print("=" * 50)
    
    try:
        # 직접 Supabase 도구 임포트
        from main.tools.supabase_mcp_tool import (
            SupabaseDirectTool,
            SupabaseReadTool,
            SupabaseCreateTool,
            SupabaseUpdateTool,
            SupabaseDeleteTool,
            create_supabase_direct_tools
        )
        
        print("✅ LangChain MCP Adapters 도구 임포트 성공")
        
        # 도구 생성 (bind_tools용)
        from main.tools.supabase_mcp_tool import get_supabase_tools_for_binding
        tools = get_supabase_tools_for_binding()
        print(f"✅ {len(tools)}개 도구 생성 완료")
        
        # 기본 도구 테스트
        basic_tool = tools[0]
        print(f"✅ 기본 도구: {basic_tool.name}")
        
        # Supabase 연결 테스트
        print("\n🔗 Supabase 연결 테스트")
        if basic_tool.test_connection():
            print("✅ Supabase 연결 성공")
        else:
            print("❌ Supabase 연결 실패")
            print("📝 환경 변수 확인: SUPABASE_URL, SUPABASE_ANON_KEY")
        
        # 사용 가능한 작업 목록
        available_operations = basic_tool.get_available_operations()
        print(f"📋 사용 가능한 작업: {available_operations}")
        
        # 읽기 도구 테스트
        print("\n📖 읽기 도구 테스트")
        read_tool = tools[0]  # SupabaseReadTool
        try:
            result = read_tool._run(
                table="예약정보",
                filters={},
                run_manager=None
            )
            result_data = json.loads(result)
            if result_data.get('success'):
                print("✅ 읽기 도구 테스트 성공")
                print(f"📊 조회된 레코드 수: {len(result_data.get('data', []))}")
            else:
                print(f"❌ 읽기 도구 테스트 실패: {result_data.get('error')}")
        except Exception as e:
            print(f"❌ 읽기 도구 테스트 오류: {e}")
        
        print("\n🎉 LangChain MCP Adapters 테스트 완료!")
        
    except ImportError as e:
        print(f"❌ LangChain MCP Adapters 임포트 실패: {e}")
        print("📝 다음 명령어로 설치해주세요:")
        print("pip install langchain-mcp-adapters")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    test_langchain_mcp_adapters()
