#!/usr/bin/env python3
"""
모든 테스트 파일을 순차 실행하고 구조화된 JSON 로그를 자동으로 저장
"""
import os, sys, subprocess, json, re
from datetime import datetime
from dotenv import load_dotenv

# ✅ .env 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(f"🔑 현재 적용된 OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

# ✅ UTF-8 인코딩 강제
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

# ✅ 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

TEST_DIR = os.path.join(ROOT_DIR, "test")
OUT_JSON = os.path.join(ROOT_DIR, "evaluation", "test_results_detailed.json")

test_files = [f for f in os.listdir(TEST_DIR) if f.startswith("test_") and f.endswith(".py")]


def parse_stdout(stdout_text: str):
    """테스트 stdout을 정규식 기반으로 파싱하여 JSON 구조로 변환"""
    cases = []
    current_case = None

    clean_text = stdout_text.encode("utf-8", "ignore").decode("utf-8")

    for line in clean_text.splitlines():
        line = line.strip()

        # --- 케이스 시작 ---
        if re.search(r"테스트\s*\d*[:：]", line):
            if current_case:
                cases.append(current_case)
            name = re.sub(r"📊|테스트\s*\d*[:：]", "", line).strip()
            current_case = {"name": name}
            continue

        # --- 의도 ---
        if re.match(r"📝?\s*의도[:：]", line):
            if current_case is not None:
                current_case["intent"] = line.split(":")[-1].strip()
            continue

        # --- 신뢰도 ---
        if re.match(r"📝?\s*신뢰도[:：]", line):
            if current_case is not None:
                nums = re.findall(r"[0-9.]+", line)
                if nums:
                    current_case["confidence"] = float(nums[0])
            continue

        # --- 라우팅 ---
        if re.match(r"📝?\s*라우팅[:：]", line):
            if current_case is not None:
                current_case["routing"] = line.split(":")[-1].strip()
            continue

        # --- 입력 문장 ---
        if re.search(r"(입력|user_input)", line):
            if current_case is not None:
                val = line.split(":")[-1].strip()
                current_case["input"] = val
            continue

        # --- 결과 판정 ---
        if "의도 분석 정확" in line:
            if current_case is not None:
                current_case["status"] = "success"
            continue
        if "의도 분석 오류" in line or "❌" in line or "오류" in line:
            if current_case is not None:
                current_case["status"] = "fail"
            continue

    if current_case:
        cases.append(current_case)

    return cases


def run_test(file_name):
    """각 테스트 실행 후 stdout 파싱"""
    print(f"\n🚀 실행 중: {file_name}")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = ROOT_DIR
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # ✅ 강제 주입 (LangGraph override 방지)
    env["LANGGRAPH_API_KEY"] = ""     # ✅ 추가
    env["LANGSMITH_API_KEY"] = ""     # ✅ 추가
    result = subprocess.run(
        [sys.executable, os.path.join(TEST_DIR, file_name)],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
        env=env,
        encoding="utf-8",  # ✅ 디코딩 강제
        errors="replace"   # ✅ 깨지는 문자 대체
    )

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    parsed_cases = parse_stdout(stdout)

    print(stdout)
    if stderr:
        print("⚠️ 오류 발생:")
        print(stderr)
    print("=" * 60)

    return {
        "test_name": file_name,
        "timestamp": datetime.now().isoformat(),
        "success": result.returncode == 0,
        "case_count": len(parsed_cases),
        "cases": parsed_cases,
        "stderr": stderr,
    }


def main():
    print("🧪 병원 AI 시스템 전체 테스트 시작")
    print("=" * 80)

    all_results = []

    for file in test_files:
        res = run_test(file)
        all_results.append(res)

    out_data = {
        "run_timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n📝 구조화된 JSON 로그가 '{OUT_JSON}'에 저장되었습니다.")
    print("🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()
