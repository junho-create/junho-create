#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 추출 결과와 정답 파일을 비교하는 스크립트.

사용법:
  python compare_results_llm.py sample1
  python compare_results_llm.py --all
  python compare_results_llm.py --all --detailed

- 비교 대상 actual:
    outputs/llm_json/*.json   (cli_llm.py 결과)
- 비교 대상 expected:
    expected/*.json           (기존 정답 파일 그대로 재사용)

주의:
- LLM 결과의 필드명 중 일부는 대장 필드와 이름이 다르기 때문에,
  아래와 같이 매핑해서 비교합니다.

  CNT_EXEC_FLAG        → CNT_CONCLUDED
  CNT_AUTO_RNW_TERM_NUM → CNT_AUTO_RNW_TERM_AMT

- TEMP_KEY는 LLM이 만들지 않으므로 비교에서 제외합니다.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# LLM 결과 디렉터리 (cli_llm.py가 쓰는 경로와 맞춰야 함)
OUTPUTS_DIR = Path(__file__).parent / "outputs" / "llm"
EXPECTED_DIR = Path(__file__).parent / "expected"
SAMPLES_DIR = Path(__file__).parent / "samples"

# LLM 결과 ↔ 기존 대장 필드명 매핑
FIELD_NAME_MAP = {
    "CNT_EXEC_FLAG": "CNT_CONCLUDED",          # 체결 여부
    "CNT_AUTO_RNW_TERM_NUM": "CNT_AUTO_RNW_TERM_AMT",  # 자동갱신 기간 숫자
}

# LLM 쪽에서는 안 뽑는 필드 → 비교에서 제외
IGNORE_FIELDS = {"TEMP_KEY"}


def load_json(path: Path) -> Dict[str, Any]:
    """JSON 파일을 로드합니다."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERR] Invalid JSON in {path}: {e}")
        return {}


def _normalize_actual_keys(actual_raw: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 결과의 필드명을 정답 파일 쪽 이름과 맞춰서 반환."""
    normalized: Dict[str, Any] = {}
    for k, v in actual_raw.items():
        canonical = FIELD_NAME_MAP.get(k, k)
        normalized[canonical] = v
    return normalized


def compare_values(actual: Any, expected: Any) -> Tuple[bool, str]:
    """실제 값과 예상 값을 비교합니다.

    Returns:
        (is_match, message) 튜플
    """
    # expected가 None이면: actual도 None이면 OK
    if expected is None:
        if actual is None or (isinstance(actual, dict) and actual.get("value") is None):
            return True, "✓ (null)"
        return False, f"✗ Expected null, got {actual}"

    # 배열 처리 (여러 가능한 정답)
    if isinstance(expected, list):
        if actual is None or (isinstance(actual, dict) and actual.get("value") is None):
            return False, f"✗ Expected one of {expected}, got null"
        actual_value = actual if not isinstance(actual, dict) else actual.get("value")
        if actual_value in expected:
            return True, f"✓ Matched: {actual_value}"
        return False, f"✗ Expected one of {expected}, got {actual_value}"

    # 단일 값 비교
    actual_value = actual if not isinstance(actual, dict) else actual.get("value")

    if actual_value == expected:
        return True, f"✓ {actual_value}"
    return False, f"✗ Expected {expected}, got {actual_value}"


def compare_result(actual_path: Path, expected_path: Path, detailed: bool = False) -> Dict[str, Any]:
    """단일 결과 파일을 비교합니다."""
    actual_raw = load_json(actual_path)
    expected = load_json(expected_path)

    if not expected:
        return {
            "file": actual_path.stem,
            "status": "no_expected",
            "matches": 0,
            "mismatches": 0,
            "missing_fields": 0,
            "details": []
        }

    # LLM 결과 필드명을 expected 쪽 이름으로 맞추기
    actual = _normalize_actual_keys(actual_raw)

    matches = 0
    mismatches = 0
    missing_fields = 0
    details: List[Dict[str, Any]] = []

    # 모든 필드 비교 (일부는 IGNORE_FIELDS로 스킵)
    all_fields = set(actual.keys()) | set(expected.keys())

    for field in sorted(all_fields):
        if field in IGNORE_FIELDS:
            continue  # TEMP_KEY 등 LLM이 안 뽑는 필드는 비교에서 제외

        actual_val = actual.get(field)
        expected_val = expected.get(field)

        if expected_val is None:
            # 정답 JSON에서 해당 필드가 None이면, 그냥 스킵하거나
            # 필요하면 비교하도록 수정 가능
            continue

        if field not in actual:
            missing_fields += 1
            details.append({
                "field": field,
                "status": "missing",
                "expected": expected_val,
                "message": "✗ Field missing in actual LLM result"
            })
            continue

        is_match, message = compare_values(actual_val, expected_val)

        if is_match:
            matches += 1
        else:
            mismatches += 1

        if detailed or not is_match:
            details.append({
                "field": field,
                "status": "match" if is_match else "mismatch",
                "expected": expected_val,
                "actual": actual_val if not isinstance(actual_val, dict) else actual_val.get("value"),
                "message": message
            })

    total = matches + mismatches + missing_fields
    accuracy = (matches / total * 100) if total > 0 else 0

    return {
        "file": actual_path.stem,
        "status": "perfect" if mismatches == 0 and missing_fields == 0 else "partial" if matches > 0 else "failed",
        "matches": matches,
        "mismatches": mismatches,
        "missing_fields": missing_fields,
        "total": total,
        "accuracy": accuracy,
        "details": details
    }


def print_comparison(result: Dict[str, Any], detailed: bool = False):
    """비교 결과를 출력합니다."""
    file_name = result["file"]
    status = result["status"]
    matches = result["matches"]
    mismatches = result["mismatches"]
    missing_fields = result["missing_fields"]
    accuracy = result["accuracy"]

    # 상태 아이콘
    status_icon = {
        "perfect": "✅",
        "partial": "⚠️",
        "failed": "❌",
        "no_expected": "⏭️"
    }.get(status, "❓")

    print(f"\n{status_icon} {file_name}")
    print(f"   정확도: {accuracy:.1f}% ({matches}/{result['total']} 일치)")

    if mismatches > 0:
        print(f"   불일치: {mismatches}개")
    if missing_fields > 0:
        print(f"   누락: {missing_fields}개")

    if status == "no_expected":
        print(f"   ⚠️  정답 파일이 없습니다: expected/{file_name}.json")
        return

    # 상세 정보 출력
    if detailed and result["details"]:
        print("   상세:")
        for detail in result["details"]:
            if detail["status"] != "match":
                print(f"      {detail['field']}: {detail['message']}")
    elif result["details"]:
        # 불일치만 출력
        mismatches_only = [d for d in result["details"] if d["status"] != "match"]
        if mismatches_only:
            print("   불일치 필드:")
            for detail in mismatches_only[:5]:  # 최대 5개만
                print(f"      {detail['field']}: {detail['message']}")
            if len(mismatches_only) > 5:
                print(f"      ... 외 {len(mismatches_only) - 5}개")


def main():
    parser = argparse.ArgumentParser(
        description="LLM 추출 결과와 정답 파일을 비교합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s sample1                    # sample1 비교
  %(prog)s --all                      # outputs/llm_json의 모든 샘플 비교
  %(prog)s --all --detailed           # 상세 출력 포함
        """
    )

    parser.add_argument(
        "sample",
        nargs="?",
        help="비교할 샘플 이름 (예: sample1, 생략 시 --all 필요)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="outputs/llm_json의 모든 샘플 비교"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="모든 필드 상세 정보 출력"
    )

    args = parser.parse_args()

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        # 모든 샘플 비교
        actual_files = sorted(OUTPUTS_DIR.glob("*.json"))
        if not actual_files:
            print(f"[ERR] {OUTPUTS_DIR} 에 LLM 결과 파일이 없습니다.")
            return

        results = []
        for actual_path in actual_files:
            expected_path = EXPECTED_DIR / actual_path.name
            result = compare_result(actual_path, expected_path, args.detailed)
            results.append(result)
            print_comparison(result, args.detailed)

        # 전체 통계
        print("\n" + "=" * 60)
        print("전체 통계")
        print("=" * 60)

        total_matches = sum(r["matches"] for r in results)
        total_mismatches = sum(r["mismatches"] for r in results)
        total_missing = sum(r["missing_fields"] for r in results)
        total_fields = sum(r["total"] for r in results)

        perfect_count = sum(1 for r in results if r["status"] == "perfect")
        partial_count = sum(1 for r in results if r["status"] == "partial")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        no_expected_count = sum(1 for r in results if r["status"] == "no_expected")

        print(f"총 샘플 수: {len(results)}")
        print(f"  ✅ 완벽 일치: {perfect_count}개")
        print(f"  ⚠️  부분 일치: {partial_count}개")
        print(f"  ❌ 실패: {failed_count}개")
        print(f"  ⏭️  정답 없음: {no_expected_count}개")
        print()
        print(f"전체 정확도: {(total_matches / total_fields * 100) if total_fields > 0 else 0:.1f}%")
        print(f"  일치: {total_matches}개")
        print(f"  불일치: {total_mismatches}개")
        print(f"  누락: {total_missing}개")

    elif args.sample:
        # 단일 샘플 비교
        actual_path = OUTPUTS_DIR / f"{args.sample}.json"
        expected_path = EXPECTED_DIR / f"{args.sample}.json"

        if not actual_path.exists():
            print(f"[ERR] LLM 결과 파일이 없습니다: {actual_path}")
            return

        result = compare_result(actual_path, expected_path, args.detailed)
        print_comparison(result, args.detailed)

        if args.detailed and result["details"] and result["status"] != "no_expected":
            print("\n모든 필드 상세:")
            for detail in result["details"]:
                print(f"  {detail['field']}: {detail['message']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
