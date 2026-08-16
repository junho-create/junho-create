#!/usr/bin/env python3
"""
실시간 채팅 인터페이스
바른마디병원 예약 시스템과 실시간으로 대화할 수 있는 터미널 기반 채팅 인터페이스
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main.langgraph_workflow import run_hospital_reservation, run_hospital_reservation_with_session_data

class ChatSession:
    """채팅 세션 관리 클래스"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.session_data = {
            "context": {},
            "reservation_info": {},
            "user_preferences": {}
        }
        self.start_time = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """대화 기록 추가"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 정보 반환"""
        return {
            "session_id": self.session_id,
            "duration": str(datetime.now() - self.start_time),
            "message_count": len(self.conversation_history),
            "context": self.session_data["context"],
            "reservation_info": self.session_data["reservation_info"]
        }

class ChatInterface:
    """실시간 채팅 인터페이스"""
    
    def __init__(self):
        self.session = ChatSession()
        self.running = True
        self.commands = {
            "/help": self._show_help,
            "/status": self._show_status,
            "/history": self._show_history,
            "/clear": self._clear_history,
            "/save": self._save_session,
            "/load": self._load_session,
            "/quit": self._quit,
            "/exit": self._quit
        }
        
    def _print_banner(self):
        """배너 출력"""
        print("=" * 80)
        print("🏥 바른마디병원 AI 예약 어시스턴트")
        print("=" * 80)
        print("💬 실시간 채팅 인터페이스")
        print(f"🆔 세션 ID: {self.session.session_id}")
        print(f"⏰ 시작 시간: {self.session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("💡 도움말을 보려면 '/help'를 입력하세요")
        print("🚪 종료하려면 '/quit' 또는 '/exit'를 입력하세요")
        print("=" * 80)
        print()
    
    def _show_help(self, args: list = None):
        """도움말 표시"""
        help_text = """
📚 사용 가능한 명령어:

💬 **일반 대화**
  • 예약하고 싶어요
  • 무릎이 아파서 의사를 추천해주세요
  • 병원 휴무일이 언제인가요?
  • 안녕하세요

🔧 **시스템 명령어**
  /help     - 이 도움말 표시
  /status   - 현재 세션 상태 확인
  /history  - 대화 기록 보기
  /clear    - 대화 기록 지우기
  /save     - 세션 저장
  /load     - 세션 불러오기
  /quit     - 프로그램 종료
  /exit     - 프로그램 종료

📋 **예약 예시**
  • "예약하고 싶어요"
  • "홍길동입니다. 010-1234-5678이에요"
  • "무릎이 아파요"
  • "최대한 빨리 예약하고 싶어요"

🩺 **의료진 추천 예시**
  • "허리가 아파요"
  • "디스크가 있어서 어떤 의사를 봐야 할까요?"
  • "두통과 어지러움이 있어요"

🏥 **병원 정보 예시**
  • "병원 휴무일이 언제인가요?"
  • "진료시간을 알려주세요"
  • "병원 전화번호가 뭐에요?"
        """
        print(help_text)
    
    def _show_status(self, args: list = None):
        """세션 상태 표시"""
        summary = self.session.get_session_summary()
        print(f"\n📊 **세션 상태**")
        print(f"🆔 세션 ID: {summary['session_id']}")
        print(f"⏱️  진행 시간: {summary['duration']}")
        print(f"💬 메시지 수: {summary['message_count']}")
        
        if summary['context']:
            print(f"🧠 컨텍스트: {json.dumps(summary['context'], ensure_ascii=False, indent=2)}")
        
        if summary['reservation_info']:
            print(f"📅 예약 정보: {json.dumps(summary['reservation_info'], ensure_ascii=False, indent=2)}")
        print()
    
    def _show_history(self, args: list = None):
        """대화 기록 표시"""
        print(f"\n📜 **대화 기록** ({len(self.session.conversation_history)}개 메시지)")
        print("-" * 60)
        
        for i, msg in enumerate(self.session.conversation_history, 1):
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            print(f"{i:2d}. {timestamp} {role_emoji} {msg['role'].upper()}: {msg['content']}")
            
            if msg['metadata']:
                print(f"    📋 메타데이터: {json.dumps(msg['metadata'], ensure_ascii=False)}")
        print("-" * 60)
        print()
    
    def _clear_history(self, args: list = None):
        """대화 기록 지우기"""
        self.session.conversation_history = []
        self.session.session_data = {
            "context": {},
            "reservation_info": {},
            "user_preferences": {}
        }
        print("✅ 대화 기록이 지워졌습니다.")
    
    def _save_session(self, args: list = None):
        """세션 저장"""
        filename = f"chat_session_{self.session.session_id[:8]}.json"
        session_data = {
            "session_id": self.session.session_id,
            "start_time": self.session.start_time.isoformat(),
            "conversation_history": self.session.conversation_history,
            "session_data": self.session.session_data
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 세션이 '{filename}'에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 세션 저장 실패: {e}")
    
    def _load_session(self, args: list = None):
        """세션 불러오기"""
        if not args:
            print("❌ 사용법: /load <파일명>")
            return
        
        filename = args[0]
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.session.session_id = session_data['session_id']
            self.session.conversation_history = session_data['conversation_history']
            self.session.session_data = session_data['session_data']
            self.session.start_time = datetime.fromisoformat(session_data['start_time'])
            
            print(f"✅ 세션 '{filename}'을 불러왔습니다.")
            print(f"📊 {len(self.session.conversation_history)}개 메시지, {len(self.session.session_data['context'])}개 컨텍스트")
        except FileNotFoundError:
            print(f"❌ 파일 '{filename}'을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ 세션 불러오기 실패: {e}")
    
    def _quit(self, args: list = None):
        """프로그램 종료"""
        print("\n👋 바른마디병원 AI 어시스턴트를 이용해주셔서 감사합니다!")
        print("💡 세션을 저장하려면 '/save' 명령어를 사용하세요.")
        self.running = False
    
    def _process_user_input(self, user_input: str) -> Optional[str]:
        """사용자 입력 처리"""
        user_input = user_input.strip()
        
        if not user_input:
            return None
        
        # 명령어 처리
        if user_input.startswith('/'):
            parts = user_input[1:].split()
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            if command in self.commands:
                self.commands[command](args)
                return None
            else:
                print(f"❌ 알 수 없는 명령어: /{command}")
                print("💡 사용 가능한 명령어: /help")
                return None
        
        return user_input
    
    def _call_agent(self, user_input: str) -> Dict[str, Any]:
        """에이전트 호출"""
        try:
            # 세션 데이터가 있으면 기존 컨텍스트와 함께 호출
            if self.session.session_data["context"] or self.session.session_data["reservation_info"]:
                result = run_hospital_reservation_with_session_data(
                    user_input,
                    self.session.session_id,
                    self.session.session_data
                )
            else:
                result = run_hospital_reservation(user_input, self.session.session_id)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "response": f"❌ 시스템 오류가 발생했습니다: {str(e)}",
                "error": str(e)
            }
    
    def _update_session_data(self, agent_result: Dict[str, Any]):
        """세션 데이터 업데이트"""
        # 컨텍스트 업데이트
        if agent_result.get("context"):
            self.session.session_data["context"].update(agent_result["context"])
        
        # 예약 정보 업데이트
        if agent_result.get("reservation_info"):
            self.session.session_data["reservation_info"].update(agent_result["reservation_info"])
        
        # 관리자 결과에서 이전 대화 의도 정보 저장
        if agent_result.get("manager_result"):
            manager_result = agent_result["manager_result"]
            if manager_result.get("extracted_info") and manager_result["extracted_info"].get("action"):
                # 이전 대화의 의도 정보를 세션에 저장
                self.session.session_data["context"]["previous_intent"] = manager_result["extracted_info"]["action"]
                print(f"🔍 이전 대화 의도를 세션에 저장: {manager_result['extracted_info']['action']}")
            
            # 라우팅 정보도 저장
            if manager_result.get("routing_info"):
                self.session.session_data["context"]["routing_info"] = manager_result["routing_info"]
        
        # RAG 결과에서 추천 의료진 정보도 세션에 저장
        if agent_result.get("rag_result"):
            rag_result = agent_result["rag_result"]
            if rag_result.get("success") and rag_result.get("recommended_doctors"):
                # RAG 결과를 세션 컨텍스트에 저장
                self.session.session_data["context"]["recommended_doctors"] = rag_result["recommended_doctors"]
                self.session.session_data["context"]["recommended_department"] = rag_result.get("department")
                self.session.session_data["context"]["rag_confidence"] = rag_result.get("confidence", 0.0)
                print(f"🔍 RAG 결과를 세션에 저장: {len(rag_result['recommended_doctors'])}명 의료진")
        
        # 예약 결과에서 수집된 정보도 세션에 저장
        if agent_result.get("reservation_result"):
            reservation_result = agent_result["reservation_result"]
            if reservation_result.get("collected_info"):
                collected_info = reservation_result["collected_info"]
                # 증상 정보가 있으면 세션에 저장
                if collected_info.get("symptoms"):
                    self.session.session_data["context"]["symptoms"] = collected_info["symptoms"]
                    print(f"🔍 증상 정보를 세션에 저장: {collected_info['symptoms']}")
                
                # 재예약 컨텍스트 정보가 있으면 세션에 저장
                if collected_info.get("rebooking_context"):
                    self.session.session_data["context"]["rebooking_context"] = collected_info["rebooking_context"]
                    print(f"🔍 재예약 컨텍스트를 세션에 저장: {collected_info['rebooking_context']}")
                
                # 추천 의료진 정보가 있으면 세션에 저장
                if collected_info.get("recommended_doctors"):
                    self.session.session_data["context"]["recommended_doctors"] = collected_info["recommended_doctors"]
                    print(f"🔍 추천 의료진을 세션에 저장: {len(collected_info['recommended_doctors'])}명")
                
                # 기타 수집된 정보들도 저장
                for key in ["환자명", "전화번호", "preferred_doctor", "recommended_department"]:
                    if collected_info.get(key):
                        self.session.session_data["context"][key] = collected_info[key]
    
    def run(self):
        """채팅 인터페이스 실행"""
        self._print_banner()
        
        while self.running:
            try:
                # 사용자 입력 받기
                user_input = input("👤 사용자: ").strip()
                
                # 입력 처리
                processed_input = self._process_user_input(user_input)
                
                if processed_input is None:
                    continue  # 명령어 처리됨
                
                # 사용자 메시지 기록
                self.session.add_message("user", processed_input)
                
                # 에이전트 호출
                print("🤖 어시스턴트: ", end="", flush=True)
                agent_result = self._call_agent(processed_input)
                
                # 응답 처리
                if agent_result.get("success", False):
                    response = agent_result.get("response", "응답을 생성할 수 없습니다.")
                else:
                    response = agent_result.get("response", "오류가 발생했습니다.")
                
                # 응답 출력 (타이핑 효과)
                self._type_message(response)
                
                # 어시스턴트 메시지 기록
                metadata = {
                    "routing_info": agent_result.get("routing_info", {}),
                    "reservation_info": agent_result.get("reservation_info", {}),
                    "context": agent_result.get("context", {}),
                    "error": agent_result.get("error")
                }
                self.session.add_message("assistant", response, metadata)
                
                # 세션 데이터 업데이트
                self._update_session_data(agent_result)
                
                print()  # 빈 줄 추가
                
            except KeyboardInterrupt:
                print("\n\n⚠️  프로그램이 중단되었습니다.")
                print("💡 정상 종료하려면 '/quit' 명령어를 사용하세요.")
                continue
            except EOFError:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {e}")
                continue
    
    def _type_message(self, message: str, delay: float = 0.02):
        """타이핑 효과로 메시지 출력"""
        import time
        
        for char in message:
            print(char, end="", flush=True)
            time.sleep(delay)
        print()

def main():
    """메인 함수"""
    try:
        # 환경 변수 확인
        required_env_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
            print("💡 .env 파일을 확인하거나 환경 변수를 설정해주세요.")
            return
        
        # 채팅 인터페이스 시작
        chat = ChatInterface()
        chat.run()
        
    except Exception as e:
        print(f"❌ 프로그램 시작 오류: {e}")
        print("💡 환경 설정을 확인해주세요.")

if __name__ == "__main__":
    main()
