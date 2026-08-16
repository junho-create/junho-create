from __future__ import annotations
from typing import List, Dict, Any
import os, yaml, re
from .augmentation import canonical_title
from .utils import normalize_text, normalize_text_preserve_symbols, translate_symbols_to_text, normalize_keyboard_symbols_only

RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")

class AdminRules:
    def __init__(self, path: str = RULES_PATH):
        self.path = path
        self.rules = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return {"priority": {"title_order": []}, "weights": {}, "top_k": 3, "fallback_depts_map": {}, "doctor_availability": {}}
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.rules, f, allow_unicode=True, sort_keys=False)

    def get_title_order(self) -> List[str]:
        return self.rules.get("priority", {}).get("title_order", [])

    def get_weights(self) -> Dict[str, float]:
        return self.rules.get("weights", {})

    def get_top_k(self) -> int:
        # 환경변수 TOP_K가 설정되어 있으면 우선 사용
        env_top_k = os.getenv("TOP_K")
        if env_top_k:
            try:
                return int(env_top_k)
            except ValueError:
                pass
        return int(self.rules.get("top_k", 3))

    def get_fallback_depts_map(self) -> Dict[str, List[str]]:
        return self.rules.get("fallback_depts_map", {})

    def get_doctor_availability(self) -> Dict[str, bool]:
        """의료진별 예약 가능 상태를 반환합니다. True=예약가능, False=예약불가"""
        return self.rules.get("doctor_availability", {})

    def set_doctor_availability(self, doctor_name: str, available: bool) -> bool:
        """의료진의 예약 가능 상태를 설정합니다."""
        self.rules.setdefault("doctor_availability", {})
        self.rules["doctor_availability"][doctor_name] = available
        self.save()
        return True

    def set_weights(self, weights: Dict[str, float]) -> bool:
        """가중치를 설정합니다."""
        self.rules.setdefault("weights", {})
        self.rules["weights"].update(weights)
        self.save()
        return True

    def set_fallback_depts_map(self, mapping: Dict[str, List[str]]) -> bool:
        """증상별 진료과 매핑을 설정합니다."""
        self.rules.setdefault("fallback_depts_map", {})
        self.rules["fallback_depts_map"].update(mapping)
        self.save()
        return True

    def execute_llm_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM이 해석한 명령을 실행합니다."""
        command_type = command_data.get("command_type")
        action = command_data.get("action")
        parameters = command_data.get("parameters", {})
        
        try:
            if command_type == "doctor_availability":
                doctor_name = parameters.get("doctor_name")
                available = parameters.get("available", True)
                if doctor_name:
                    self.set_doctor_availability(doctor_name, available)
                    status = "재개" if available else "중단"
                    return {
                        "updated": True, 
                        "command_type": command_type,
                        "message": f"{doctor_name} 선생님의 예약을 {status}했습니다."
                    }
            
            elif command_type == "title_priority":
                title_order = parameters.get("title_order", [])
                if title_order:
                    self.rules.setdefault("priority", {})["title_order"] = title_order
                    self.save()
                    return {
                        "updated": True, 
                        "command_type": command_type,
                        "title_order": title_order,
                        "message": f"직함 우선순위를 {' → '.join(title_order)}로 설정했습니다."
                    }
            
            elif command_type == "dept_weight":
                weights = parameters.get("weights", {})
                if weights:
                    self.set_weights(weights)
                    return {
                        "updated": True, 
                        "command_type": command_type,
                        "weights": weights,
                        "message": f"가중치를 업데이트했습니다: {weights}"
                    }
            
            elif command_type == "symptom_mapping":
                mapping = parameters.get("mapping", {})
                if mapping:
                    self.set_fallback_depts_map(mapping)
                    return {
                        "updated": True, 
                        "command_type": command_type,
                        "mapping": mapping,
                        "message": f"증상 매핑을 업데이트했습니다: {mapping}"
                    }
            
            return {"updated": False, "message": f"지원하지 않는 명령 타입입니다: {command_type}"}
            
        except Exception as e:
            return {"updated": False, "message": f"명령 실행 중 오류 발생: {str(e)}"}

    def apply_admin_command(self, text: str, _translated: bool = False) -> Dict[str, Any]:
        """관리자 명령을 처리합니다. LLM 기반 처리를 우선 시도하고, 실패시 정규식 기반으로 fallback합니다."""
        
        # 1. LLM 기반 명령 처리 시도
        try:
            from .llm import LLMClient
            llm = LLMClient()
            command_data = llm.process_admin_command(text, self.rules)
            
            if command_data and command_data.get("confidence", 0) > 0.6:  # 신뢰도가 60% 이상인 경우만
                result = self.execute_llm_command(command_data)
                if result.get("updated"):
                    result["method"] = "llm"
                    result["confidence"] = command_data.get("confidence")
                    result["explanation"] = command_data.get("explanation")
                    return result
        except Exception as e:
            # print(f"LLM 명령 처리 중 오류 (fallback to regex): {e}")
            pass
        
        # 2. 정규식 기반 fallback 처리
        # 2-0. 먼저 인식하지 못하는 유니코드 기호들을 하이픈으로 정규화
        text_normalized = normalize_keyboard_symbols_only(text)
        
        # 2-1. 기호가 포함된 경우 LLM을 이용한 자연어 변환 시도
        if not _translated and any(symbol in text_normalized for symbol in ["→", "➜", "⇒", ">>", "≫", "=", ":", "->", "-"]):
            try:
                from .llm import LLMClient
                llm = LLMClient()
                llm_converted = llm.convert_symbols_to_natural_language(text_normalized)
                if llm_converted and llm_converted != text.strip():
                    return self.apply_admin_command(llm_converted, _translated=True)
            except Exception as e:
                print(f"LLM 기호 변환 실패, 정규식 변환으로 fallback: {e}")
        
        # 2-2. LLM 변환 실패 시 정규식 기반 변환 시도
        if not _translated:
            t_translated = translate_symbols_to_text(text_normalized)  # 정규화된 텍스트로 정규식 기반 변환
            if t_translated != text.strip():
                return self.apply_admin_command(t_translated, _translated=True)
        
        # 2-2. 기본 텍스트 정규화
        t = normalize_text(text)  # 기본 정규화 (기호 제거됨)
        t_symbols = normalize_text_preserve_symbols(text)  # 기호 보존 정규화
        
        # 2-1. 직함 우선순위 설정 명령 처리 (기호 보존 필요)
        
        # 직함 우선순위 패턴들 (LLM 변환된 자연어 우선 처리)
        title_priority_patterns = [
            # LLM 변환된 패턴: "센터장 다음 원장 다음 교수 다음 전문의 순으로 우선순위 설정"
            r".*다음.*다음.*순으로\s*우선순위\s*설정",
            # 기본 변환 패턴: "센터장 다음 원장 다음 교수 다음 전문의 순으로"
            r".*다음.*다음.*순으로",
            # 우선순위 설정 명시: "우선순위를 센터장 원장 교수 전문의로 설정"
            r"우선순위를?\s*((?:대표원장|센터장|내과센터장|신경센터장|교수|원장|부원장|전문의|전임의|임상강사)(?:\s+(?:대표원장|센터장|내과센터장|신경센터장|교수|원장|부원장|전문의|전임의|임상강사))*)\s*(?:로\s*설정|순으로)",
            # 기존 자연어 패턴: "대표원장 다음 교수 다음 전문의 순으로"
            r"((?:대표원장|센터장|내과센터장|신경센터장|교수|원장|부원장|전문의|전임의|임상강사)(?:\s*(?:다음|그다음|그리고)\s*(?:대표원장|센터장|내과센터장|신경센터장|교수|원장|부원장|전문의|전임의|임상강사))*)\s*순(?:으로|서로)?",
            # 화살표가 포함된 모든 텍스트를 캐치
            r".*(→.*){2,}.*",  # 화살표가 2번 이상 나오는 패턴 (최소 3개 항목)
            # 하이픈이 포함된 모든 텍스트를 캐치
            r".*(-.*){2,}.*",   # 하이픈이 2번 이상 나오는 패턴 (최소 3개 항목)
            # 매핑 순서: "매핑 순서를 대표원장부터 교수까지"
            r"매핑\s*순서를?\s*((?:대표원장|센터장|내과센터장|신경센터장|교수|원장|부원장|전문의|전임의|임상강사)(?:부터|에서)?\s*(?:대표원장|센터장|내과센터장|신경센터장|교수|원장|부원장|전문의|전임의|임상강사))*",
        ]
        
        order = None
        for pattern in title_priority_patterns:
            m = re.search(pattern, t_symbols)  # 기호 보존된 텍스트 사용
            if m:
                # 매칭된 전체 텍스트 또는 그룹 사용
                matched_text = m.group(1) if m.groups() else m.group(0)
                
                # LLM 변환된 자연어나 기호가 포함된 경우 직함 추출
                titles = ["대표원장", "센터장", "내과센터장", "신경센터장", "교수", "원장", "부원장", "전문의", "전임의", "임상강사"]
                found_titles = []
                
                # 텍스트에서 직함들을 순서대로 찾기
                for title in titles:
                    if title in matched_text:
                        found_titles.append(title)
                
                # 최소 2개 이상의 직함이 있어야 유효한 우선순위 설정
                if len(found_titles) >= 2:
                    order = found_titles
                    break
        
        if order:
            self.rules.setdefault("priority", {})["title_order"] = order
            self.save()
            return {"updated": True, "method": "regex", "title_order": order, "message": f"직함 우선순위를 {' → '.join(order)}로 설정했습니다."}
        
        # 2-2. 의료진 예약 상태 변경 명령 처리 (대폭 확장된 패턴)
        
        # 예약 중단 패턴들 (개선된 이름 추출)
        doctor_unavailable_patterns = [
            # 복잡한 문장 패턴: "홍길동 박사님은 잠시 예약을 받을 수 없게 해줘"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)\s*(?:은|는)\s*(?:당분간|일시적으로|잠시|잠깐)?\s*(?:예약을?\s*)?(?:받을\s*수\s*없게|못\s*받게|안\s*되게|받지\s*않도록|안\s*받도록|중단하도록|정지하도록)",
            # 기본 패턴: "김재훈 선생님은 당분간 예약을 받지 않도록 해줘"  
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:은|는)?\s*(?:당분간|일시적으로|잠시|잠깐)?\s*(?:예약을?\s*)?(?:받지\s*않도록|안\s*받도록|중단하도록|정지하도록)",
            # 간단한 패턴: "김재훈 예약 중단"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:예약\s*)?(?:중단|정지|차단|막기)",
            # 명령형: "김재훈 선생님 예약 받지마"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:예약\s*)?(?:받지\s*마|받지\s*말고|못\s*받게|안\s*받게)",
            # 상태 설정: "김재훈 선생님을 예약 불가로 설정"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:을|를)?\s*(?:예약\s*)?(?:불가|불가능|비활성|오프|off)(?:로\s*설정|상태로|하게)?",
            # 일반적인 자연어: "김재훈 선생님 예약을 받을 수 없게 해줘" (복잡한 문장이 아닌 경우)
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:예약을?\s*)?(?:받을\s*수\s*없게|못\s*받게|안\s*되게)",
        ]
        
        for pattern in doctor_unavailable_patterns:
            m_unavailable = re.search(pattern, t)
            if m_unavailable:
                doctor_name = m_unavailable.group(1)
                self.set_doctor_availability(doctor_name, False)
                return {"updated": True, "method": "regex", "doctor_availability": {doctor_name: False}, "message": f"{doctor_name} 선생님의 예약을 중단했습니다."}
        
        # 예약 재개 패턴들 (개선된 이름 추출)
        doctor_available_patterns = [
            # 복잡한 문장 패턴: 이름이 앞에 명확히 있는 경우 우선 처리
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)\s*(?:은|는)\s*(?:예약을?\s*)?(?:다시\s*받도록|재개하도록|시작하도록|받을\s*수\s*있도록|받을\s*수\s*있게|가능하게)",
            # 기본 패턴: "김재훈 선생님은 예약을 다시 받도록 해줘"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:은|는)?\s*(?:예약을?\s*)?(?:다시\s*받도록|재개하도록|시작하도록|받을\s*수\s*있도록)",
            # 간단한 패턴: "김재훈 예약 재개"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:예약\s*)?(?:재개|시작|활성화|복구)",
            # 명령형: "김재훈 선생님 예약 받아", "김철수 의사님 예약 받게 해줘"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:예약\s*)?(?:받아|받도록|받게)",
            # 상태 설정: "김재훈 선생님을 예약 가능으로 설정"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:을|를)?\s*(?:예약\s*)?(?:가능|활성|온|on)(?:로\s*설정|상태로|하게)?",
            # 일반적인 자연어: "김재훈 선생님 예약을 받을 수 있게 해줘"
            r"([가-힣]{2,4})\s*(?:선생님?|의사님?|교수님?|박사님?)?\s*(?:예약을?\s*)?(?:받을\s*수\s*있게|받을\s*수\s*있도록|가능하게)",
        ]
        
        for pattern in doctor_available_patterns:
            m_available = re.search(pattern, t)
            if m_available:
                doctor_name = m_available.group(1)
                self.set_doctor_availability(doctor_name, True)
                return {"updated": True, "method": "regex", "doctor_availability": {doctor_name: True}, "message": f"{doctor_name} 선생님의 예약을 재개했습니다."}
        
        # 2-3. TOP_K 설정 명령 처리 (구어체 패턴 추가)
        topk_patterns = [
            r"(?:top\s*k|추천\s*개수|추천\s*수|결과\s*개수)를?\s*(\d+)(?:개|명)?(?:로\s*설정|으로\s*변경)?",
            r"(\d+)(?:개|명)(?:\s*추천|까지\s*추천|만\s*추천)",
            r"최대\s*(\d+)(?:개|명)(?:\s*까지)?",
            # 구어체 패턴: "5명까지만 추천해줘"
            r"(\d+)(?:개|명)까지만?\s*(?:추천|보여줘|알려줘|해줘)",
            r"(\d+)(?:개|명)\s*정도만?\s*(?:추천|보여줘|알려줘)",
            # 자연어: "추천은 3개 정도로 해줘"
            r"추천은?\s*(\d+)(?:개|명)\s*정도로?\s*(?:해줘|하자|설정)",
            # 간단한 패턴: "3개로 해줘"
            r"(\d+)(?:개|명)로\s*(?:해줘|하자|설정)",
        ]
        
        for pattern in topk_patterns:
            m = re.search(pattern, t)
            if m:
                try:
                    top_k_value = int(m.group(1))
                    if 1 <= top_k_value <= 10:  # 합리적인 범위 제한
                        self.rules["top_k"] = top_k_value
                        self.save()
                        return {"updated": True, "method": "regex", "top_k": top_k_value, "message": f"TOP_K를 {top_k_value}로 설정했습니다."}
                except ValueError:
                    continue
        
        # 2-4. 가중치 설정 명령 처리 (기호 사용 가능)
        weight_patterns = [
            r"([가-힣]+(?:과|센터|부))\s*가중치를?\s*([0-9.]+)(?:로\s*설정|으로\s*변경)?",
            r"([가-힣]+)\s*(?:의\s*)?중요도를?\s*([0-9.]+)(?:로\s*설정|으로\s*변경)?",
            r"([가-힣]+)\s*점수를?\s*([0-9.]+)(?:로\s*설정|으로\s*변경)?",
            # 기호를 사용한 패턴들
            r"([가-힣]+(?:과|센터|부))\s*[:=]\s*([0-9.]+)",
            r"([가-힣]+)\s*가중치\s*[:=]\s*([0-9.]+)",
            r"([가-힣]+)\s*weight\s*[:=]\s*([0-9.]+)",
        ]
        
        for pattern in weight_patterns:
            # 기호가 포함될 수 있으므로 기호 보존 텍스트도 시도
            m = re.search(pattern, t) or re.search(pattern, t_symbols)
            if m:
                try:
                    dept_name = m.group(1)
                    weight_value = float(m.group(2))
                    if 0.1 <= weight_value <= 5.0:  # 합리적인 범위 제한
                        self.rules.setdefault("weights", {})
                        # 진료과별 가중치를 dept_exact 형태로 저장
                        weight_key = f"{dept_name}_weight"
                        self.rules["weights"][weight_key] = weight_value
                        self.save()
                        return {"updated": True, "method": "regex", "weights": {weight_key: weight_value}, "message": f"{dept_name}의 가중치를 {weight_value}로 설정했습니다."}
                except ValueError:
                    continue
        
        # 2-5. 증상-진료과 매핑 설정 (정확한 증상 추출, 기호 사용 가능)
        symptom_mapping_patterns = [
            # 복합 증상 패턴: "허리통증 환자는 정형외과로 분류" -> "허리통증" 추출
            r"([가-힣]+(?:통증|아픔|질환))\s*환자는?\s*([가-힣]+(?:과|센터|부))(?:로\s*연결|에\s*매핑|으로\s*보내기|로\s*분류)",
            # 기본 패턴: "두통증상은 신경과로 연결"
            r"([가-힣]+(?:통증|아픔|증상|질환))\s*(?:은|는)?\s*([가-힣]+(?:과|센터|부))(?:로\s*연결|에\s*매핑|으로\s*보내기)",
            # 간단한 증상 패턴: "두통은 신경과"
            r"([가-힣]+(?:통|통증|아픔|병|질환))\s*(?:은|는)\s*([가-힣]+(?:과|센터|부))",
            # 자연어 패턴: "관절 증상은 정형외과"
            r"([가-힣]+)\s*증상\s*(?:은|는)?\s*([가-힣]+(?:과|센터|부))",
            # 담당/분류 패턴: "소화불량은 내과에서 담당"
            r"([가-힣]+(?:불량|장애|이상))\s*(?:은|는)\s*([가-힣]+(?:과|센터|부))(?:에서\s*담당|로\s*분류)",
            # 기호를 사용한 매핑 패턴들
            r"([가-힣]+(?:통증|아픔|증상|질환))\s*[-→>:]\s*([가-힣]+(?:과|센터|부))",
            r"([가-힣]+)\s*->\s*([가-힣]+(?:과|센터|부))",
            r"([가-힣]+)\s*:\s*([가-힣]+(?:과|센터|부))",
        ]
        
        for pattern in symptom_mapping_patterns:
            # 기호가 포함될 수 있으므로 기호 보존 텍스트도 시도
            m = re.search(pattern, t) or re.search(pattern, t_symbols)
            if m:
                symptom = m.group(1)
                dept = m.group(2)
                self.rules.setdefault("fallback_depts_map", {})
                self.rules["fallback_depts_map"][symptom] = [dept]
                self.save()
                return {"updated": True, "method": "regex", "symptom_mapping": {symptom: [dept]}, "message": f"{symptom}을(를) {dept}로 매핑했습니다."}
        
        # 2-6. 전체 시스템 설정
        system_patterns = [
            r"(?:전체\s*)?(?:시스템|설정)\s*(?:초기화|리셋|재설정)",
            r"(?:모든\s*)?(?:규칙|설정)\s*(?:삭제|제거|초기화)",
        ]
        
        for pattern in system_patterns:
            if re.search(pattern, t):
                # 기본 설정으로 초기화
                self.rules = {"priority": {"title_order": []}, "weights": {}, "top_k": 3, "fallback_depts_map": {}, "doctor_availability": {}}
                self.save()
                return {"updated": True, "method": "regex", "message": "시스템 설정을 초기화했습니다."}
        
        return {"updated": False, "message": "규칙 문장을 파싱할 수 없습니다. 지원되는 명령어 형식을 확인해주세요."}

    def title_priority_score(self, title: str) -> float:
        order = self.get_title_order()
        title_canon = canonical_title(title or "")
        if title_canon in order:
            idx = order.index(title_canon)
            score = max(0.0, 1.0 - 0.1 * idx)
            return score
        return 0.0
