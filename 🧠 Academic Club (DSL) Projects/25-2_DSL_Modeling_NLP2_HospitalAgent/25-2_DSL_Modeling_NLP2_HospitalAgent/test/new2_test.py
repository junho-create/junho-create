#!/usr/bin/env python3
"""
🧩 에이전트 맥락 추적 및 전환 테스트 (확장판)
- 총 15개 시나리오
- 의도 전환, 기억력, 복합 증상, 예외 처리 등 종합 테스트
"""

import os, sys, json, traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🚀 에이전트 맥락 전환·기억력 테스트 시작")
    print("=" * 80)

    try:
        from main.agents.agent1_manager import Agent1Manager
        from main.agents.agent2_reservation import Agent2Reservation
        from main.agents.agent3_rag import Agent3RAG
    except Exception as e:
        print(f"❌ 에이전트 임포트 오류: {e}")
        traceback.print_exc()
        return

    agent1, agent2, agent3 = Agent1Manager(), Agent2Reservation(), Agent3RAG()
    all_logs = []

    def run_turns(title, turns, ctx=None, info=None):
        ctx = ctx or {}
        info = info or {}
        print(f"\n🧩 [{title}]")
        print("=" * 60)
        logs = []
        for i, msg in enumerate(turns, 1):
            print(f"\n🔄 턴 {i}: {msg}")
            try:
                res = agent1.route_to_agent_or_tool(msg, ctx)
                intent = res.get("primary_intent", "Unknown")
                log = {"turn": i, "input": msg, "intent": intent}

                if intent == "reservation":
                    result = agent2.process_reservation_request(msg, info)
                    info.update(result.get("collected_info", {}))
                    log.update({
                        "type": "reservation",
                        "message": result.get("message", ""),
                        "success": result.get("success", False)
                    })
                    print(f"📅 예약 응답: {result.get('message')}")
                elif intent == "symptom_doctor":
                    rag = agent3.recommend_doctors([msg], msg)
                    log.update({
                        "type": "symptom_mapping",
                        "department": rag.get("department", ""),
                        "confidence": rag.get("confidence", 0.0)
                    })
                    print(f"💡 진료과 추천: {rag.get('department')} ({rag.get('confidence', 0.0):.2f})")
                    ctx["recommended_dept"] = rag.get("department", "")
                elif intent == "hospital_info":
                    log["type"] = "hospital_info"
                    log["message"] = res.get("message", "")
                    print(f"🏥 병원 정보: {res.get('message')}")
                else:
                    print(f"⚠️ 의도 불명: {intent}")
                logs.append(log)
                ctx.update(info)
            except Exception as e:
                logs.append({"turn": i, "input": msg, "error": str(e)})
                print(f"❌ 예외 발생: {e}")
        all_logs.extend(logs)

    # ============================
    # 15개 시나리오 정의
    # ============================
    scenarios = [
        ("Scenario 1: 예약 → 증상 → 예약 완료",
         ["예약하고 싶어요", "무릎이 아파요", "이름은 김민수예요", "전화번호는 010-4444-5555", "내일 오전 9시에요"]),
        ("Scenario 2: 복합 증상 + 예약 연속성",
         ["허리랑 어깨가 둘 다 아파요", "그럼 예약하고 싶어요", "이름은 최지훈, 010-9999-8888이에요"]),
        ("Scenario 3: 의도 전환 (예약→정보요청→재예약)",
         ["예약하려고요", "병원 점심시간은 언제예요?", "그럼 점심 이후로 예약하고 싶어요", "이름은 홍길동, 010-3333-2222"]),
        ("Scenario 4: 예외 입력 처리",
         ["", " ", "asdfgh", "12345", "....", "응"]),
        ("Scenario 5: 재예약 요청 (이전 정보 기억)",
         ["지난주 예약했던 사람인데 내일로 변경 가능할까요?", "김민수예요"]),
        ("Scenario 6: 증상 반복 업데이트",
         ["어제는 무릎이 아팠는데 오늘은 허리도 아파요", "그럼 어떤 진료과 가야 해요?"]),
        ("Scenario 7: 예약 중 취소 요청",
         ["예약하려고요", "이름은 박철수예요", "아니요, 취소하고 싶어요"]),
        ("Scenario 8: 병원 정보만 문의",
         ["오늘 진료 가능한가요?", "토요일 진료하나요?", "진료시간 알려주세요"]),
        ("Scenario 9: 연속 증상 후 예약",
         ["기침이 심하고 열도 있어요", "그럼 내과로 예약할게요", "이름은 이수연이에요"]),
        ("Scenario 10: 부서 추천 후 전화번호 누락",
         ["어깨 통증이 있어요", "예약하고 싶어요", "이름은 김민지예요"]),
        ("Scenario 11: 외국인 사용자 입력",
         ["I want to make an appointment", "My knee hurts", "Name is John Kim"]),
        ("Scenario 12: 불명확한 입력",
         ["음...", "그거 있잖아요", "그럼 그렇게 해주세요"]),
        ("Scenario 13: 기존 예약 확인 요청",
         ["제가 어제 예약했는데 확인 가능할까요?", "김민수예요"]),
        ("Scenario 14: 중복 정보 입력",
         ["예약할게요", "이름은 홍길동이에요", "홍길동입니다", "전화번호는 010-2222-3333이에요"]),
        ("Scenario 15: 최종 맥락 복원 및 응답 일관성",
         ["지난번처럼 예약하고 싶어요", "같은 시간으로 가능할까요?", "이름은 김지현이에요"])
    ]

    for title, turns in scenarios:
        run_turns(title, turns)

    save_logs(all_logs)
    print("\n📝 전체 로그 저장 완료!")
    print("=" * 80)
    print("🎉 15개 시나리오 테스트 종료!")

# -----------------------------------------------------
def save_logs(results: list):
    """테스트 결과를 logs 폴더에 저장"""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("logs", f"context_flow_results_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"🗂 로그 파일 저장: {path}")

# -----------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY 미설정")
    if not os.getenv("SUPABASE_URL"):
        print("⚠️ SUPABASE_URL 미설정")
    main()
