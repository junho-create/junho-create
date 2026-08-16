#!/usr/bin/env python3
"""
📘 통합 에이전트 테스트 스위트 (로그 저장 포함)
- Agent1, Agent2, Agent3 전체를 대상으로 다중 시나리오 자동 테스트
- 모든 실행 결과를 logs/test_results_{timestamp}.json 파일로 자동 저장
"""

import os, sys, json, traceback
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드 및 경로 설정
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================
# 🧠 Agent 테스트 통합 진입점
# =====================================================
def main():
    print("🚀 통합 에이전트 테스트 스위트 시작")
    print("=" * 70)

    try:
        from main.agents.agent1_manager import Agent1Manager
        from main.agents.agent2_reservation import Agent2Reservation
        from main.agents.agent3_rag import Agent3RAG
    except Exception as e:
        print(f"❌ 에이전트 임포트 오류: {e}")
        traceback.print_exc()
        return

    agent1, agent2, agent3 = Agent1Manager(), Agent2Reservation(), Agent3RAG()
    test_results = []  # 🔹 결과 누적용 리스트

    # =====================================================
    # 1️⃣ 의도 분류 테스트
    # =====================================================
    print("\n🧩 [TEST 1] Agent1 의도 분류 정확도")
    print("=" * 60)
    intent_cases = [
        ("안녕하세요", "greeting"),
        ("예약하고 싶어요", "reservation"),
        ("어깨가 아파요", "symptom_doctor"),
        ("예약 취소하고 싶어요", "reservation"),
        ("병원 진료시간 알려주세요", "hospital_info"),
        ("의사 누구 있어요?", "symptom_doctor"),
        ("몸이 아파요", "symptom_doctor"),
        ("지금 여는 병원인가요?", "hospital_info"),
    ]
    for text, expected in intent_cases:
        res = agent1.route_to_agent_or_tool(text)
        actual = res.get("primary_intent", "Unknown")
        success = actual == expected
        test_results.append({
            "test": "intent_classification",
            "input": text,
            "expected": expected,
            "actual": actual,
            "success": success
        })
        mark = "✅" if success else "❌"
        print(f"{mark} '{text}' → {actual} (예상: {expected})")

    # =====================================================
    # 2️⃣ 예약 정보 누락 테스트
    # =====================================================
    print("\n🧾 [TEST 2] Agent2 누락 정보 단계별 보완")
    print("=" * 60)
    collected = {}
    steps = [
        "예약하고 싶어요",
        "이름은 김철수예요",
        "전화번호는 010-1234-5678",
        "내일 오전 10시요",
        "무릎이 아파요"
    ]
    for i, msg in enumerate(steps, 1):
        result = agent2.process_reservation_request(msg, collected)
        collected.update(result.get("collected_info", {}))
        missing = agent2._check_missing_information(collected)
        test_results.append({
            "test": "missing_info_progress",
            "turn": i,
            "input": msg,
            "response": result.get("message", ""),
            "missing": missing,
            "success": result.get("success", False)
        })
        print(f"턴{i} | 입력: {msg}")
        print(f"💬 응답: {result.get('message','No message')}")
        print(f"📋 누락: {missing}")
        if result.get("success"):
            print("✅ 예약 완료!\n")
            break

    # =====================================================
    # 3️⃣ 증상 → 진료과 매핑 테스트
    # =====================================================
    print("\n🧠 [TEST 3] Agent3 RAG 진료과 추천")
    print("=" * 60)
    symptoms = [
        ("무릎 통증", "정형외과"),
        ("두통과 어지럼", "신경과"),
        ("가슴이 답답해요", "순환기내과"),
        ("피부가 가려워요", "피부과"),
        ("소화가 잘 안돼요", "내과"),
        ("불면증이 심해요", "정신건강의학과"),
        ("허리통증과 다리저림", "정형외과"),
    ]
    for sym, expected in symptoms:
        res = agent3.recommend_doctors([sym], sym)
        dept = res.get("department", "Unknown")
        conf = res.get("confidence", 0.0)
        success = res.get("success", False)
        test_results.append({
            "test": "rag_mapping",
            "input": sym,
            "expected_department": expected,
            "predicted_department": dept,
            "confidence": conf,
            "success": success
        })
        if success:
            mark = "✅" if expected in dept else "⚠️"
            print(f"{mark} '{sym}' → {dept} (예상: {expected}) | 신뢰도 {conf:.2f}")
        else:
            print(f"❌ 실패: {sym} | {res.get('message','No msg')}")

    # =====================================================
    # 4️⃣ 응급 증상 테스트
    # =====================================================
    print("\n🚨 [TEST 4] 응급 증상 라우팅")
    print("=" * 60)
    emergencies = [
        "숨이 막히고 가슴이 답답해요",
        "피가 계속 나요",
        "의식이 희미해요",
        "교통사고가 났어요"
    ]
    for e in emergencies:
        r = agent1.route_to_agent_or_tool(e)
        test_results.append({
            "test": "emergency_routing",
            "input": e,
            "detected_intent": r.get("primary_intent", "Unknown"),
            "message": r.get("message", "")
        })
        print(f"입력: {e} → {r.get('primary_intent','Unknown')}")

    # =====================================================
    # 5️⃣ 병원 정보 테스트
    # =====================================================
    print("\n🏥 [TEST 5] 병원 운영 정보 요청")
    print("=" * 60)
    infos = [
        "오늘 진료하나요?",
        "점심시간이 언제인가요?",
        "토요일도 여나요?",
        "병원 위치 알려주세요"
    ]
    for q in infos:
        res = agent1.route_to_agent_or_tool(q)
        test_results.append({
            "test": "hospital_info",
            "input": q,
            "intent": res.get("primary_intent", ""),
            "message": res.get("message", "")
        })
        print(f"{q} → {res.get('primary_intent','Unknown')}")

    # =====================================================
    # 6️⃣ 불완전 입력 대응
    # =====================================================
    print("\n💬 [TEST 6] 불완전 입력 대응")
    print("=" * 60)
    short_inputs = ["예약...", "음...", "어깨가...", "그...", "저기요"]
    for s in short_inputs:
        res = agent1.route_to_agent_or_tool(s)
        test_results.append({
            "test": "incomplete_sentence",
            "input": s,
            "intent": res.get("primary_intent", "Unknown"),
            "message": res.get("message", "")
        })
        print(f"입력: '{s}' → {res.get('primary_intent','Unknown')}")

    # =====================================================
    # 🔟 로그 저장
    # =====================================================
    save_results(test_results)
    print("\n📝 로그 저장 완료!")
    print("=" * 70)
    print("🎉 모든 테스트 종료!")

# =====================================================
# 로그 저장 함수
# =====================================================
def save_results(results: list):
    """테스트 결과를 JSON 파일로 저장"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(log_dir, f"test_results_{ts}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"🗂 로그 파일 저장: {file_path}")

# =====================================================
# 실행
# =====================================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
    if not os.getenv("SUPABASE_URL"):
        print("⚠️ SUPABASE_URL이 설정되지 않았습니다.")
    main()
