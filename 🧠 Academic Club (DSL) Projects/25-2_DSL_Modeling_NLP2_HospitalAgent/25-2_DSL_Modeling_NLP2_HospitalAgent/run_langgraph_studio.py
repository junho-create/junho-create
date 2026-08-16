#!/usr/bin/env python3
"""
LangGraph Studio 실행 스크립트
Medical Reservation Agent를 LangGraph Studio에서 실행하기 위한 설정
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """LangGraph Studio 실행"""
    print("🏥 Medical Reservation Agent - LangGraph Studio")
    print("=" * 60)
    
    # 현재 디렉토리 확인
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    print("🔧 LangGraph Studio 설정:")
    print("   - Config: langgraph.json")
    print("   - Graph: medical_reservation")
    print("   - Entry Point: ./main/langgraph_workflow.py:create_hospital_reservation_workflow")
    print("=" * 60)
    
    # .env 파일 확인
    env_file = current_dir / ".env"
    if not env_file.exists():
        print("⚠️  .env 파일이 없습니다!")
        print("   env_example_langgraph.txt 파일을 참고하여 .env 파일을 생성해주세요.")
        print("   특히 OPENAI_API_KEY는 필수입니다.")
        print()
    
    try:
        # LangGraph Studio 실행
        print("🚀 LangGraph Studio 시작 중...")
        print("   브라우저에서 http://localhost:8123 으로 접속하세요.")
        print("   종료하려면 Ctrl+C를 누르세요.")
        print()
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['LANGGRAPH_API_PORT'] = '8123'
        
        subprocess.run([
            "langgraph", "dev", "--port", "8123", "--allow-blocking"
        ], check=True, env=env)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ LangGraph Studio 실행 오류: {e}")
        print()
        print("해결 방법:")
        print("1. langgraph CLI가 설치되어 있는지 확인:")
        print("   pip install langgraph-cli")
        print()
        print("2. 의존성이 설치되어 있는지 확인:")
        print("   pip install -r requirements.txt")
        print()
        print("3. .env 파일이 올바르게 설정되어 있는지 확인")
        
    except KeyboardInterrupt:
        print("\n🛑 LangGraph Studio가 중지되었습니다")

if __name__ == "__main__":
    main()
