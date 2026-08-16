
from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

from .retriever import get_shared_retriever
from .rules import AdminRules
from .llm import LLMClient
from .augmentation import expand_symptoms
from .output_enforcer import enforce_output, OutputSchema
from .utils import normalize_text

# Try LangGraph; else fallback
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False

def _clean_label(val: str, prefer_korean: bool = True) -> str:
    if not val:
        return val
    parts = re.split(r"[\|/]+", val)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return val.strip()
    if prefer_korean:
        ko = [p for p in parts if re.search(r"[가-힣]", p)]
        if ko:
            return ko[0]
    parts_sorted = sorted(parts, key=lambda x: (-len(re.findall(r"[가-힣A-Za-z]", x)), -len(x)))
    return parts_sorted[0] if parts_sorted else parts[0]

def _extract_preferred_doctors_from_other_info(other_info: List[str]) -> List[str]:
    """other_info에서 희망 의사명을 추출합니다."""
    if not other_info:
        return []
    
    preferred_doctors = []
    
    # 의사명 추출 패턴들
    patterns = [
        r'([가-힣]{2,4})\s*(?:원장|교수|전문의|의사)(?:님)?(?:께|에게|한테)?\s*(?:진료|치료|상담|예약|만나|보고\s*싶|받고\s*싶)',
        r'([가-힣]{2,4})\s*(?:원장|교수|전문의|의사)(?:님)?(?:과|와|랑)?\s*(?:진료|치료|상담|예약)',
        r'([가-힣]{2,4})\s*(?:원장|교수|전문의|의사)(?:님)?(?:을|를)?\s*(?:찾|원해|희망)',
        r'([가-힣]{2,4})\s*(?:원장|교수|전문의|의사)(?:님)?(?:이|가)?\s*(?:좋|괜찮)',
    ]
    
    for info_text in other_info:
        if not isinstance(info_text, str):
            continue
            
        for pattern in patterns:
            matches = re.findall(pattern, info_text, re.IGNORECASE)
            for match in matches:
                doctor_name = match.strip()
                if doctor_name and len(doctor_name) >= 2:
                    preferred_doctors.append(doctor_name)
    
    # 중복 제거하면서 순서 유지
    return list(dict.fromkeys(preferred_doctors))

def _select_with_rules(query: Dict[str, Any], retrieved: List[Dict[str, Any]], rules: AdminRules) -> Dict[str, Any]:
    syms = query.get("symptoms", [])
    syms_norm = [s.strip() for s in syms if s and isinstance(s, str)]
    aug_syms = expand_symptoms(syms_norm)
    weights = rules.get_weights()
    top_k = int(os.getenv("TOP_K", str(rules.get_top_k())))
    
    # other_info에서 희망 의사 추출
    other_info = query.get("other_info", [])
    preferred_doctors = _extract_preferred_doctors_from_other_info(other_info)
    has_single_preferred_doctor = len(preferred_doctors) == 1

    candidates = []
    for h in retrieved:
        meta = h.get("meta", {})
        name = meta.get("doctor_name") or meta.get("의료진명") or ""
        dept = meta.get("dept") or meta.get("진료과") or ""
        title = meta.get("title") or meta.get("직함") or ""
        doc_id = meta.get("doctor_id") or meta.get("DocID") or meta.get("DocID_응급실포함") or ""
        text = h.get("text", "")
        if name and dept:
            score = 0.0
            if any(d in text for d in [dept]):
                score += weights.get("dept_exact", 1.2)
            else:
                score += weights.get("dept_close", 1.0)
            if any(sym.lower() in text.lower() for sym in aug_syms):
                score += weights.get("symptom_match", 1.0)
            score += rules.title_priority_score(title) * weights.get("title_weight", 0.2)
            candidates.append({
                "doctor_name": name, "dept": dept, "title": title, "score": score,
                "evidence_id": h.get("id"), "evidence_text": text,
                "matched_symptoms": [s for s in aug_syms if s.lower() in text.lower()][:5],
                "doctor_id": doc_id
            })

    if not candidates:
        fbmap = rules.get_fallback_depts_map()
        dept_score = {}
        for s in aug_syms:
            for key, depts in fbmap.items():
                if key in s:
                    for d in depts:
                        dept_score[d] = dept_score.get(d, 0) + 1
        sel_dept = sorted(dept_score.items(), key=lambda x: -x[1])[0][0] if dept_score else ""
        candidates.append({"doctor_name": "", "dept": sel_dept, "title": "", "score": 0.1, "evidence_id": "", "evidence_text": "", "matched_symptoms": aug_syms[:3]})

    candidates.sort(key=lambda x: -x["score"])
    top_k = max(1, top_k)
    
    # 희망 의사가 1명인 경우 특별 처리
    if has_single_preferred_doctor:
        preferred_doctor = preferred_doctors[0]
        
        # 1. 먼저 증상 관련 후보군을 필터링 (점수 기준)
        symptom_related_candidates = [c for c in candidates if c["score"] > 0.5 or c["matched_symptoms"]]
        
        # 2. 증상 관련 후보군에서 희망 의사를 찾기
        preferred_candidate = None
        for c in symptom_related_candidates:
            if preferred_doctor in c["doctor_name"]:
                preferred_candidate = c
                break
        
        # 3. 희망 의사가 증상 관련 후보군에 있으면 우선 선택
        if preferred_candidate:
            selected = [preferred_candidate]
            seen_doctors = {preferred_candidate["doctor_name"]}
            
            # 나머지 슬롯은 기존 로직으로 채움
            for c in candidates:
                doctor_name = c["doctor_name"]
                if doctor_name not in seen_doctors and len(selected) < top_k:
                    selected.append(c)
                    seen_doctors.add(doctor_name)
        else:
            # 4. 희망 의사가 증상 관련 후보군에 없으면 강제 매핑
            # 전체 후보군에서 희망 의사를 찾기
            forced_candidate = None
            for c in candidates:
                if preferred_doctor in c["doctor_name"]:
                    forced_candidate = c
                    forced_candidate = forced_candidate.copy()  # 복사본 생성
                    forced_candidate["forced_mapping"] = True  # 강제 매핑 표시
                    break
            
            if forced_candidate:
                selected = [forced_candidate]
                seen_doctors = {forced_candidate["doctor_name"]}
                
                # 나머지 슬롯은 기존 로직으로 채움
                for c in candidates:
                    doctor_name = c["doctor_name"]
                    if doctor_name not in seen_doctors and len(selected) < top_k:
                        selected.append(c)
                        seen_doctors.add(doctor_name)
            else:
                # 희망 의사를 전혀 찾을 수 없는 경우 기존 로직 사용
                selected = []
                seen_doctors = set()
                for c in candidates:
                    doctor_name = c["doctor_name"]
                    if doctor_name not in seen_doctors:
                        selected.append(c)
                        seen_doctors.add(doctor_name)
                        if len(selected) >= top_k:
                            break
    else:
        # 희망 의사가 없거나 2명 이상인 경우 기존 로직 사용
        selected = []
        seen_doctors = set()
        
        for c in candidates:
            doctor_name = c["doctor_name"]
            if doctor_name not in seen_doctors:
                selected.append(c)
                seen_doctors.add(doctor_name)
                if len(selected) >= top_k:
                    break
    
    # 서로 다른 의료진이 부족한 경우, 나머지는 NULL로 채움
    while len(selected) < top_k:
        selected.append({
            "doctor_name": None, "dept": None, "title": None, "score": 0.0,
            "evidence_id": None, "evidence_text": "", "matched_symptoms": []
        })

    # 첫 번째 유효 의료진을 primary로 선택(Null 제안은 건너뛴다)
    primary_valid = next((c for c in selected if c.get("doctor_name") not in (None, "")), None)
    if primary_valid is None:
        # 진료과는 후보/룰 기반 추정값을 그대로 사용(편향 기본값 제거)
        dept_guess = _clean_label((selected[0].get("dept") if selected else "") or "", prefer_korean=True)
        primary = {"dept": dept_guess or "", "doctor_name": ""}
    else:
        primary = primary_valid
    dept_final = _clean_label(primary.get("dept", ""), prefer_korean=True) if primary.get("dept") else ""
    doctor_final = _clean_label(primary.get("doctor_name", "") or "", prefer_korean=True)

    suggestions = []
    first_non_null_doctor: str = ""
    for c in selected:
        if c["doctor_name"] is None:
            # NULL 항목 추가
            suggestions.append({
                "의료진명": None,
                "진료과": None,
                "환자의 구체적인 증상": None,
                "이유": None
            })
        else:
            reason = []
            
            # 강제 매핑된 경우 특별한 이유 추가
            if c.get("forced_mapping"):
                reason.append("환자 희망 의사이나 주 증상과 전문분야 불일치로 인한 강제 매핑")
            
            if c.get("title"):
                reason.append(f"직함:{c['title']}")
            if c["matched_symptoms"]:
                reason.append(f"증상매칭:{', '.join(c['matched_symptoms'][:3])}")
            if c.get("evidence_id"):
                reason.append(f"근거:{c['evidence_id']}")
            if not reason:
                reason.append("증상과의 연관성 기반 추천")
            suggestions.append({
                "의료진명": _clean_label(c["doctor_name"] or "(미상)", prefer_korean=True),
                "진료과": _clean_label(c["dept"], prefer_korean=True),
                "환자의 구체적인 증상": list(dict.fromkeys(c["matched_symptoms"]))[:5],
                "이유": "; ".join(reason),
                "doctor_id": c.get("doctor_id", "")
            })
            if not first_non_null_doctor and c.get("doctor_name"):
                first_non_null_doctor = _clean_label(c["doctor_name"], prefer_korean=True)

    # 최종 보정: primary에서 doctor_name이 비어있는 경우, 첫 유효 제안의 의료진명을 사용
    if not doctor_final and first_non_null_doctor:
        doctor_final = first_non_null_doctor

    evidence_ids = [c["evidence_id"] for c in selected if c.get("evidence_id")]

    out = {
        "patient_name": query.get("patient_name", ""),
        "patient_gender": query.get("patient_gender", ""),
        "phone_num": query.get("phone_num", ""),
        "chat_start_date": query.get("chat_start_date", ""),
        "symptoms": syms_norm,
        "visit_type": query.get("visit_type", ""),
        "preference_datetime": query.get("preference_datetime", []),
        "dept": dept_final,
        "doctor_name": doctor_final,
        "top_k_suggestions": suggestions,
        "retrieval_evidence": evidence_ids
    }
    return out

def build_and_run_agent(input_json: Dict[str, Any]) -> OutputSchema:
    """Prefer LangGraph when available; otherwise fallback pipeline."""
    if HAS_LANGGRAPH:
        graph = build_langgraph_agent()
        if graph is not None:
            app = graph.compile()
            state = {"input_json": input_json, "retrieved": [], "retrieval_ok": False, "llm_ok": True, "draft_output": {}, "output": {}, "logs": [], "errors": []}
            final = app.invoke(state)
            return enforce_output(final["output"])
    # Fallback path
    retriever = get_shared_retriever()
    rules = AdminRules()
    top_k = int(os.getenv("TOP_K", str(rules.get_top_k())))
    retrieved = retriever.retrieve(input_json.get("symptoms", []), top_k=max(12, top_k*3))
    llm = LLMClient()
    llm_output = llm.structured_select(input_json, retrieved, rules)
    if llm_output:
        payload = {
            "patient_name": input_json.get("patient_name", ""),
            "patient_gender": input_json.get("patient_gender", ""),
            "phone_num": input_json.get("phone_num", ""),
            "chat_start_date": input_json.get("chat_start_date", ""),
            "symptoms": input_json.get("symptoms", []),
            "visit_type": input_json.get("visit_type", ""),
            "preference_datetime": input_json.get("preference_datetime", []),
            "dept": llm_output.get("dept", ""),
            "doctor_name": llm_output.get("doctor_name", ""),
            "top_k_suggestions": llm_output.get("top_k_suggestions", []),
            "retrieval_evidence": llm_output.get("retrieval_evidence", []),
        }
        return enforce_output(payload)
    payload = _select_with_rules(input_json, retrieved, rules)
    return enforce_output(payload)

def build_langgraph_agent():
    if not HAS_LANGGRAPH:
        return None
    from typing import TypedDict

    class State(TypedDict, total=False):
        input_json: Dict[str, Any]
        retrieved: List[Dict[str, Any]]
        retrieval_ok: bool
        llm_ok: bool
        draft_output: Dict[str, Any]
        output: Dict[str, Any]
        logs: List[str]
        errors: List[str]

    graph = StateGraph(State)
    rules = AdminRules()
    retriever = get_shared_retriever()
    llm = LLMClient()

    def log(state, msg: str):
        logs = state.get("logs", [])
        logs.append(msg)
        return {"logs": logs}

    def node_prepare(state):
        # Assume OPENAI_API_KEY is set; LLMClient was constructed
        out = {"llm_ok": True}
        out.update(log(state, "prepare"))
        return out

    def node_retrieve(state):
        top_k = int(os.getenv("TOP_K", str(rules.get_top_k())))
        try:
            hits = retriever.retrieve(state["input_json"].get("symptoms", []), top_k=max(12, top_k*3))
        except Exception as e:
            errs = state.get("errors", []); errs.append(f"retrieve_error:{type(e).__name__}")
            out = {"retrieved": [], "retrieval_ok": False, "errors": errs}
            out.update(log(state, "retrieve_fail"))
            return out
        provider_hits = [h for h in hits if (h.get("meta", {}).get("doctor_name") and h.get("meta", {}).get("dept"))]
        out = {"retrieved": hits, "retrieval_ok": len(provider_hits) > 0}
        out.update(log(state, f"retrieve_ok:{len(hits)}"))
        return out

    def branch_select(state):
        if state.get("llm_ok", False):
            return "llm"
        return "rules"

    def node_select_llm(state):
        try:
            res = llm.structured_select(state["input_json"], state.get("retrieved", []), rules)
            if not res:
                # LLM이 결과를 반환하지 않은 경우 (API 키 문제, 모델 응답 없음 등)
                # 자동으로 규칙 기반(rules-based) 선택 로직으로 fallback 수행
                #print("⚠️  LLM 선택 실패: 결과 없음 → 규칙 기반 로직으로 자동 전환")
                errs = state.get("errors", []); errs.append("llm_select_none_fallback_to_rules")
                res = _select_with_rules(state["input_json"], state.get("retrieved", []), rules)
                out = {"draft_output": res, "llm_ok": False, "errors": errs}
                out.update(log(state, "select_llm_fail_fallback_rules"))
                return out
            
            # LLM 결과를 완전한 형태로 보완
            input_json = state["input_json"]
            complete_res = {
                "patient_name": input_json.get("patient_name", ""),
                "patient_gender": input_json.get("patient_gender", ""),
                "phone_num": input_json.get("phone_num", ""),
                "chat_start_date": input_json.get("chat_start_date", ""),
                "symptoms": input_json.get("symptoms", []),
                "visit_type": input_json.get("visit_type", ""),
                "preference_datetime": input_json.get("preference_datetime", []),
                "dept": res.get("dept", ""),
                "doctor_name": res.get("doctor_name", ""),
                "top_k_suggestions": res.get("top_k_suggestions", []),
                "retrieval_evidence": res.get("retrieval_evidence", [])
            }
            
            # LLM이 성공적으로 결과를 반환한 경우
            #print("✅ LLM 기반 의료진 선택 완료")
            out = {"draft_output": complete_res}
            out.update(log(state, "select_llm_ok"))
            return out
        except Exception as e:
            # LLM 호출 중 예외 발생 (네트워크 오류, API 키 오류, 모델 오류 등)
            # 자동으로 규칙 기반(rules-based) 선택 로직으로 fallback 수행
            # print(f"⚠️  LLM 선택 오류 ({type(e).__name__}): {str(e)[:100]} → 규칙 기반 로직으로 자동 전환")
            errs = state.get("errors", []); errs.append(f"llm_error_{type(e).__name__}_fallback_to_rules")
            res = _select_with_rules(state["input_json"], state.get("retrieved", []), rules)
            out = {"draft_output": res, "llm_ok": False, "errors": errs}
            out.update(log(state, "select_llm_error_fallback_rules"))
            return out

    def node_select_rules(state):
        # 처음부터 규칙 기반 선택을 사용하는 경우 (LLM 비활성화 상태)
        # print("🔧 규칙 기반 의료진 선택 수행 (LLM 미사용)")
        res = _select_with_rules(state["input_json"], state.get("retrieved", []), rules)
        out = {"draft_output": res}
        out.update(log(state, "select_rules_ok"))
        return out

    def branch_validate(state):
        draft = state.get("draft_output", {})
        try:
            enforce_output(draft)
            return "ok"
        except Exception:
            return "repair"

    def node_validate(state):
        draft = state.get("draft_output", {})
        validated = enforce_output(draft)
        out = {"output": json.loads(validated.model_dump_json(by_alias=True))}
        out.update(log(state, "validate_ok"))
        return out

    def node_repair_with_rules(state):
        res = _select_with_rules(state["input_json"], state.get("retrieved", []), rules)
        validated = enforce_output(res)
        out = {"output": json.loads(validated.model_dump_json(by_alias=True))}
        out.update(log(state, "repair_with_rules"))
        return out

    graph.add_node("prepare", node_prepare)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("select_llm", node_select_llm)
    graph.add_node("select_rules", node_select_rules)
    graph.add_node("validate", node_validate)
    graph.add_node("repair_with_rules", node_repair_with_rules)

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "retrieve")
    graph.add_conditional_edges("retrieve", branch_select, {"llm": "select_llm", "rules": "select_rules"})
    graph.add_edge("select_rules", "validate")
    graph.add_edge("select_llm", "validate")
    graph.add_conditional_edges("validate", branch_validate, {"ok": END, "repair": "repair_with_rules"})
    graph.add_edge("repair_with_rules", END)

    return graph
