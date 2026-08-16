"""
에이전트1: 관리자 에이전트 (프롬프트 기반)
- LLM을 사용하여 사용자 입력을 분석하고 적절한 하위 에이전트나 툴로 분기
- 예약 관련 → 에이전트2
- 증상-의료진 매핑 → 에이전트3  
- 병원 정보 → Tavily 검색 툴
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from .prompts import MANAGER_AGENT_PROMPT

class Agent1Manager:
    """관리자 에이전트 - LLM 기반 요청 분기 및 라우팅"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or self._get_default_llm_client()
        
        # 의도별 키워드 정의
        self.intent_keywords = {
            "reservation": [
                # 예약 생성 관련
                "예약", "예약하고", "예약하고싶", "예약하고싶어", "예약하고싶다", "예약하고싶어요", "예약하고싶습니다",
                "예약하고싶어", "예약하고싶다", "예약하고싶어요", "예약하고싶습니다",
                # 예약 확인 관련
                "예약확인", "예약 확인", "예약조회", "예약 조회", "내예약", "내 예약", "예약내역", "예약 내역",
                "예약상태", "예약 상태", "예약정보", "예약 정보", "예약내용", "예약 내용",
                # 예약 취소 관련
                "예약취소", "예약 취소", "예약삭제", "예약 삭제", "취소하고", "취소하고싶", "취소하고싶어",
                # 예약 변경 관련
                "예약변경", "예약 변경", "예약수정", "예약 수정", "시간바꾸", "시간 바꾸", "일정바꾸", "일정 바꾸"
            ],
            "symptom_doctor": ["아프", "통증", "부상", "다치", "불편", "증상", "무릎", "어깨", "목", "허리", "등", "발목", "손목", "두통", "어지럼", "복통", "소화", "내시경"],
            "hospital_info": ["휴무일", "휴진일", "휴일", "휴무", "운영시간", "진료시간", "병원시간", "영업시간", "연락처", "전화번호", "번호", "주소", "위치", "오시는길"]
        }
    
    def _get_default_llm_client(self):
        """기본 LLM 클라이언트 설정"""
        try:
            # OpenAI 클라이언트 설정
            import openai
            return openai.OpenAI()
        except ImportError:
            print("⚠️ OpenAI 클라이언트를 사용할 수 없습니다. 기본 키워드 기반으로 폴백합니다.")
            return None
    
    def analyze_user_intent(self, user_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        LLM을 사용한 사용자 입력 분석 및 의도 분류
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            의도 분석 결과
        """
        print(f"�� 의도 분석 시작: {user_input}")
        try:
            if self.llm_client:
                result = self._llm_based_intent_analysis(user_input, conversation_context)
                print(f"�� LLM 분석 결과: {result}")
                return result
            else:
                result = self._fallback_keyword_analysis(user_input, conversation_context)
                print(f"🔍 폴백 분석 결과: {result}")
                return result

        except Exception as e:
            print(f"❌ 의도 분석 오류: {e}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "message": "요청 분석 중 오류가 발생했습니다."
            }
    
    def _llm_based_intent_analysis(self, user_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """LLM을 사용한 의도 분석"""
        try:
            # 컨텍스트가 있으면 프롬프트에 포함
            if conversation_context:
                context_str = json.dumps(conversation_context, ensure_ascii=False, indent=2)
                
                # 이전 대화에서 예약 확인을 요청했는지 확인
                previous_intent = conversation_context.get('previous_intent', '') if conversation_context else ''
                if previous_intent == 'check':
                    context_info = f"\n\n**중요**: 이전 대화에서 사용자가 '예약 확인'을 요청했습니다. 현재 입력 '{user_input}'은 예약 확인을 위한 환자 정보 제공입니다. intent는 'reservation'이고 action은 'check'로 분류하세요."
                elif previous_intent == 'create':
                    context_info = f"\n\n**중요**: 이전 대화에서 사용자가 '예약 생성'을 요청했습니다. 현재 입력 '{user_input}'은 예약 생성을 위한 환자 정보 제공입니다. intent는 'reservation'이고 action은 'create'로 분류하세요."
                else:
                    context_info = f"\n\n**중요**: 이전 대화에서 예약 관련 정보가 수집되고 있다면, 현재 입력도 예약 관련으로 분류하세요. 예를 들어, '이름은 박영희'나 '전화번호는 010-1234-5678' 같은 입력은 예약 정보 수집의 연속입니다."
                
                prompt = MANAGER_AGENT_PROMPT.format(
                    user_input=user_input, 
                    conversation_context=context_str,
                    current_datetime=self._get_current_datetime_info()
                ) + context_info
            else:
                prompt = MANAGER_AGENT_PROMPT.format(
                    user_input=user_input, 
                    conversation_context="없음",
                    current_datetime=self._get_current_datetime_info()
                )
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 바른마디병원의 지능형 관리자 에이전트입니다. 사용자 요청을 분석하고 적절한 전문 에이전트나 도구로 라우팅하는 것이 당신의 임무입니다. 이전 대화 컨텍스트를 고려하여 사용자의 의도를 정확히 파악하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # JSON 응답 파싱
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # 라우팅 정보 생성
            routing_info = self._get_routing_info_from_llm_result(result)
            
            return {
                "success": True,
                "primary_intent": result.get("intent", "unclear"),
                "confidence": result.get("confidence", 0.8),
                "extracted_info": result.get("extracted_info", {}),
                "routing": routing_info,
                "reasoning": result.get("reasoning", ""),
                "message": f"{result.get('intent', 'unknown')} 관련 요청으로 분석되었습니다."
            }
            
        except json.JSONDecodeError:
            return {
                "success": False,
                "status": "parse_error",
                "message": "LLM 응답을 파싱할 수 없습니다. 다시 시도해주세요."
            }
        except Exception as e:
            return {
                "success": False,
                "status": "llm_error",
                "error": str(e),
                "message": "LLM 분석 중 오류가 발생했습니다."
            }
    
    def _get_routing_info_from_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 결과에서 라우팅 정보 생성"""
        intent = result.get("intent", "unclear")
        next_action = result.get("next_action", "")
        
        routing_map = {
            "reservation": {
                "target_agent": "agent2_reservation",
                "action": "process_reservation_request",
                "description": "예약 처리 에이전트로 라우팅"
            },
            "symptom_doctor": {
                "target_agent": "agent3_rag",
                "action": "recommend_doctors_for_symptoms",
                "description": "RAG 에이전트로 라우팅하여 의료진 추천"
            },
            "hospital_info": {
                "target_tool": "tavily_search",
                "action": "search_hospital_general",
                "description": "Tavily 검색 툴로 라우팅하여 병원 정보 검색"
            }
        }
        
        return routing_map.get(intent, {
            "target_agent": "agent2_reservation",
            "action": "process_reservation_request",
            "description": "기본 예약 처리로 라우팅"
        })
    
    def _fallback_keyword_analysis(self, user_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """LLM 실패 시 키워드 기반 분석 (폴백)"""
        # 기존 키워드 기반 로직을 유지
        normalized_input = self._normalize_input(user_input)
        intent_scores = self._classify_intent(normalized_input)
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        confidence = primary_intent[1]
        
        if confidence < 0.3:
            return {
                "success": False,
                "status": "unclear_intent",
                "message": "요청을 명확히 이해하지 못했습니다. 좀 더 구체적으로 말씀해주세요.",
                "suggestions": [
                    "예약 관련: '예약하고 싶어요'",
                    "의료진 추천: '어떤 의사가 좋을까요?'", 
                    "병원 정보: '휴무일이 언제인가요?'"
                ]
            }
        
        extracted_info = self._extract_intent_specific_info(normalized_input, primary_intent[0])
        
        return {
            "success": True,
            "primary_intent": primary_intent[0],
            "confidence": confidence,
            "extracted_info": extracted_info,
            "routing": self._get_routing_info(primary_intent[0], extracted_info),
            "message": f"{primary_intent[0]} 관련 요청으로 분류되었습니다."
        }
    
    def _normalize_input(self, user_input: str) -> str:
        """입력 정규화"""
        # 공백 정리
        normalized = re.sub(r'\s+', ' ', user_input.strip())
        
        # 특수문자 제거 (한글, 영문, 숫자, 기본 특수문자만 유지)
        normalized = re.sub(r'[^\w\s가-힣.,!?]', '', normalized)
        
        return normalized.lower()
    
    def _classify_intent(self, normalized_input: str) -> Dict[str, float]:
        """의도 분류 (키워드 기반 점수 계산)"""
        intent_scores = {intent: 0.0 for intent in self.intent_keywords.keys()}
        
        # 환자 정보 패턴 감지 (이름, 전화번호) - 예약 의도로 높은 점수 부여
        import re
        phone_pattern = r'01[0-9]-?[0-9]{3,4}-?[0-9]{4}'
        name_pattern = r'[가-힣]{2,4}'
        
        if re.search(phone_pattern, normalized_input) or re.search(name_pattern, normalized_input):
            intent_scores["reservation"] += 0.8  # 환자 정보가 있으면 예약 의도로 높은 점수
        
        # 각 의도별 키워드 매칭 점수 계산
        for intent, keywords in self.intent_keywords.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in normalized_input:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # 키워드 개수에 따른 정규화된 점수
            if keywords:
                intent_scores[intent] += score / len(keywords)
        
        return intent_scores
    
    def _extract_intent_specific_info(self, normalized_input: str, intent: str) -> Dict[str, Any]:
        """의도별 세부 정보 추출"""
        extracted = {"original_input": normalized_input}
        
        if intent == "reservation":
            extracted.update(self._extract_reservation_info(normalized_input))
        elif intent == "symptom_doctor":
            extracted.update(self._extract_symptom_info(normalized_input))
        elif intent == "hospital_info":
            extracted.update(self._extract_hospital_info(normalized_input))
        
        return extracted
    
    def _extract_reservation_info(self, input_text: str) -> Dict[str, Any]:
        """예약 관련 정보 추출"""
        info = {"intent_type": "reservation"}
        
        # 예약 유형 추출 (더 정확한 키워드 매칭)
        if any(word in input_text for word in ["예약확인", "예약 확인", "예약조회", "예약 조회", "내예약", "내 예약", "예약내역", "예약 내역", "예약상태", "예약 상태", "예약정보", "예약 정보"]):
            info["action"] = "check"
        elif any(word in input_text for word in ["재예약", "재 예약", "다시 예약", "또 예약", "같은 의사", "같은 선생님", "이전 의사", "이전 선생님", "전에 봤던", "전에 진료받던"]):
            info["action"] = "rebook"
        elif any(word in input_text for word in ["예약취소", "예약 취소", "예약삭제", "예약 삭제", "취소하고", "취소하고싶", "취소하고싶어"]):
            info["action"] = "cancel"
        elif any(word in input_text for word in ["예약변경", "예약 변경", "예약수정", "예약 수정", "시간바꾸", "시간 바꾸", "일정바꾸", "일정 바꾸"]):
            info["action"] = "modify"
        elif any(word in input_text for word in ["예약", "예약하고", "예약하고싶", "예약하고싶어", "예약하고싶다", "예약하고싶어요", "예약하고싶습니다"]):
            info["action"] = "create"
        else:
            info["action"] = "create"  # 기본값
        
        # 환자 정보 패턴 감지 (이름, 전화번호)
        import re
        phone_pattern = r'01[0-9]-?[0-9]{3,4}-?[0-9]{4}'
        name_pattern = r'[가-힣]{2,4}'
        
        if re.search(phone_pattern, input_text) or re.search(name_pattern, input_text):
            info["has_patient_info"] = True
            info["action"] = "create"  # 환자 정보가 있으면 예약 생성으로 간주
        
        return info
    
    def _extract_symptom_info(self, input_text: str) -> Dict[str, Any]:
        """증상 관련 정보 추출"""
        info = {"intent_type": "symptom_doctor"}
        
        # 증상 추출
        symptoms = []
        symptom_keywords = [
            "아프", "통증", "부상", "다치", "불편", "증상",
            "무릎", "어깨", "목", "허리", "등", "발목", "손목",
            "두통", "어지럼", "복통", "소화", "내시경"
        ]
        
        for keyword in symptom_keywords:
            if keyword in input_text:
                symptoms.append(keyword)
        
        info["symptoms"] = symptoms
        info["has_symptoms"] = len(symptoms) > 0
        
        return info
    
    def _extract_hospital_info(self, input_text: str) -> Dict[str, Any]:
        """병원 정보 관련 정보 추출"""
        info = {"intent_type": "hospital_info"}
        
        # 정보 유형 추출
        if any(word in input_text for word in ["휴무일", "휴진일", "휴일", "휴무"]):
            info["info_type"] = "holidays"
        elif any(word in input_text for word in ["운영시간", "진료시간", "병원시간", "영업시간"]):
            info["info_type"] = "hours"
        elif any(word in input_text for word in ["연락처", "전화번호", "번호"]):
            info["info_type"] = "contact"
        elif any(word in input_text for word in ["주소", "위치", "오시는길"]):
            info["info_type"] = "location"
        else:
            info["info_type"] = "general"
        
        return info
    
    def _get_routing_info(self, intent: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """라우팅 정보 생성"""
        routing_map = {
            "reservation": {
                "target_agent": "agent2_reservation",
                "action": "process_reservation_request",
                "description": "예약 처리 에이전트로 라우팅"
            },
            "symptom_doctor": {
                "target_agent": "agent3_rag", 
                "action": "recommend_doctors_for_symptoms",
                "description": "RAG 에이전트로 라우팅하여 의료진 추천"
            },
            "hospital_info": {
                "target_tool": "tavily_search",
                "action": "search_hospital_general",
                "description": "Tavily 검색 툴로 라우팅하여 병원 정보 검색"
            }
        }
        
        return routing_map.get(intent, {
            "target_agent": "agent2_reservation",
            "action": "process_reservation_request", 
            "description": "기본 예약 처리로 라우팅"
        })
    
    def route_to_agent_or_tool(self, user_input: str, existing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        적절한 에이전트나 툴로 라우팅
        
        Args:
            user_input: 사용자 입력
            existing_context: 기존 컨텍스트 정보
            
        Returns:
            라우팅 결과
        """
        try:
            # 의도 분석 (컨텍스트 포함)
            intent_result = self.analyze_user_intent(user_input, existing_context)
            
            if not intent_result.get("success"):
                return intent_result
            
            primary_intent = intent_result["primary_intent"]
            confidence = intent_result.get("confidence", 0.0)
            routing_info = intent_result["routing"]
            extracted_info = intent_result["extracted_info"]
            
            # 라우팅 실행
            if primary_intent == "reservation":
                result = self._route_to_reservation_agent(user_input, extracted_info, existing_context)
            elif primary_intent == "symptom_doctor":
                result = self._route_to_rag_agent(user_input, extracted_info, existing_context)
            elif primary_intent == "hospital_info":
                result = self._route_to_search_tool(user_input, extracted_info, existing_context)
            elif primary_intent == "greeting" or primary_intent == "general":
                # 기본적인 인사말이나 일반적인 대화는 Agent1이 직접 처리
                result = self._handle_direct_response(user_input, primary_intent)
            else:
                return {
                    "success": False,
                    "message": "처리할 수 없는 요청입니다.",
                    "suggestion": "예약, 의료진 추천, 또는 병원 정보 중 하나를 선택해주세요."
                }
            
            # 의도 정보를 최종 결과에 보존
            result["primary_intent"] = primary_intent
            result["confidence"] = confidence
            result["extracted_info"] = extracted_info
            
            return result
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "라우팅 중 오류가 발생했습니다."
            }
    
    def _route_to_reservation_agent(self, user_input: str, extracted_info: Dict[str, Any], existing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """예약 에이전트로 라우팅"""
        try:
            from .agent2_reservation import Agent2Reservation
            
            # Agent2 인스턴스 생성
            agent2 = Agent2Reservation()
            
            # 기존 컨텍스트와 새 정보 병합
            context_info = existing_context or {}
            
            # Agent2로 예약 요청 처리
            result = agent2.process_reservation_request(user_input, context_info)
            
            # 라우팅 정보 추가
            result["routing_info"] = {
                "target": "agent2_reservation",
                "target_agent": "agent2_reservation",
                "action": "process_reservation_request",
                "intent": "reservation"
            }
            
            return result
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Agent2 임포트 오류: {str(e)}",
                "message": "예약 에이전트를 불러올 수 없습니다."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "예약 에이전트 호출 중 오류가 발생했습니다."
            }
    
    def _route_to_rag_agent(self, user_input: str, extracted_info: Dict[str, Any], existing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """RAG 에이전트로 라우팅"""
        try:
            from .agent3_rag import Agent3RAG
            
            # Agent3 인스턴스 생성
            agent3 = Agent3RAG()
            
            symptoms = extracted_info.get("symptoms", [])
            if not symptoms:
                # 증상이 명시되지 않았으면 입력에서 추출
                symptoms = [user_input]
            
            # Agent3로 의료진 추천 요청
            result = agent3.recommend_doctors(symptoms, user_input)
            
            # 라우팅 정보 추가
            result["routing_info"] = {
                "target": "agent3_rag",
                "target_agent": "agent3_rag",
                "action": "recommend_doctors", 
                "intent": "symptom_doctor"
            }
            
            return result
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Agent3 임포트 오류: {str(e)}",
                "message": "의료진 추천 에이전트를 불러올 수 없습니다."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "의료진 추천 에이전트 호출 중 오류가 발생했습니다."
            }
    
    def _route_to_search_tool(self, user_input: str, extracted_info: Dict[str, Any], existing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """검색 툴로 라우팅"""
        try:
            from ..tools.tavily_search import search_hospital_general
            
            info_type = extracted_info.get("info_type", "general")
            
            # 정보 유형에 따른 특화된 검색
            if info_type == "holidays":
                from ..tools.tavily_search import search_hospital_holidays
                result_message = search_hospital_holidays(user_input)
            elif info_type == "hours":
                from ..tools.tavily_search import search_hospital_hours
                result_message = search_hospital_hours(user_input)
            elif info_type == "contact":
                from ..tools.tavily_search import search_hospital_contact
                result_message = search_hospital_contact(user_input)
            else:
                result_message = search_hospital_general(user_input)
            
            # 검색 결과 검증 및 정확도 향상
            validated_message = self._validate_search_result(result_message, info_type)
            
            return {
                "success": True,
                "message": validated_message,
                "routing_info": {
                    "target": "tavily_search",
                    "target_agent": "tavily_search",
                    "action": f"search_hospital_{info_type}",
                    "intent": "hospital_info"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "병원 정보 검색 중 오류가 발생했습니다."
            }
    
    def _get_current_datetime_info(self) -> str:
        """현재 날짜/시간 정보를 한국어로 반환"""
        try:
            from datetime import datetime, timedelta
            
            # 현재 날짜 정보
            now = datetime.now()
            current_date = now.strftime("%Y년 %m월 %d일")
            current_time = now.strftime("%H시 %M분")
            current_weekday = now.strftime("%A")
            current_weekday_kr = {
                "Monday": "월요일",
                "Tuesday": "화요일", 
                "Wednesday": "수요일",
                "Thursday": "목요일",
                "Friday": "금요일",
                "Saturday": "토요일",
                "Sunday": "일요일"
            }.get(current_weekday, "알 수 없음")
            
            # 상대적 날짜 계산
            tomorrow = (now + timedelta(days=1)).strftime("%Y년 %m월 %d일")
            day_after_tomorrow = (now + timedelta(days=2)).strftime("%Y년 %m월 %d일")
            
            return f"오늘: {current_date} {current_weekday_kr} {current_time}, 내일: {tomorrow}, 모레: {day_after_tomorrow}"
            
        except Exception as e:
            print(f"⚠️ 날짜/시간 정보 조회 오류: {e}")
            return f"현재 날짜: {datetime.now().strftime('%Y년 %m월 %d일')}"
    
    def _validate_search_result(self, result_message: str, info_type: str) -> str:
        """검색 결과 검증 및 정확도 향상"""
        try:
            from datetime import datetime, timedelta
            
            # 현재 날짜 정보
            now = datetime.now()
            current_date = now.strftime("%Y년 %m월 %d일")
            current_weekday = now.strftime("%A")
            current_weekday_kr = {
                "Monday": "월요일",
                "Tuesday": "화요일", 
                "Wednesday": "수요일",
                "Thursday": "목요일",
                "Friday": "금요일",
                "Saturday": "토요일",
                "Sunday": "일요일"
            }.get(current_weekday, "알 수 없음")
            
            # 휴무일 정보 검증
            if info_type == "holidays":
                # 잘못된 날짜 정보 수정
                if "2025년 9월 26일" in result_message:
                    result_message = result_message.replace("2025년 9월 26일", current_date)
                
                # 휴무일 로직 검증
                if "월요일" in result_message and "휴무일" in result_message:
                    # 월요일이 휴무일인지 확인
                    if current_weekday == "Monday":
                        # 오늘이 월요일이면 휴무일
                        result_message = result_message.replace("오늘은 휴무일입니다", "오늘은 휴무일입니다")
                    else:
                        # 오늘이 월요일이 아니면 영업일
                        result_message = result_message.replace("오늘은 휴무일입니다", f"오늘({current_weekday_kr})은 영업일입니다")
                
                # 정확한 휴무일 정보 추가
                result_message += f"\n\n📅 **현재 날짜**: {current_date} ({current_weekday_kr})"
                result_message += f"\n🏥 **바른마디병원 휴무일**: 매주 월요일"
                
                if current_weekday == "Monday":
                    result_message += f"\n⚠️ **오늘은 휴무일입니다**"
                else:
                    result_message += f"\n✅ **오늘은 영업일입니다**"
            
            return result_message
            
        except Exception as e:
            # 검증 실패 시 원본 메시지 반환
            return result_message
    
    def _handle_direct_response(self, user_input: str, intent: str) -> Dict[str, Any]:
        """Agent1이 직접 처리할 수 있는 기본적인 응답"""
        try:
            if intent == "greeting":
                # 인사말에 대한 응답
                responses = [
                    "안녕하세요! 바른마디병원 AI 어시스턴트입니다. 😊",
                    "안녕하세요! 무엇을 도와드릴까요?",
                    "안녕하세요! 예약, 의료진 추천, 또는 병원 정보 중 어떤 것을 도와드릴까요?"
                ]
                import random
                message = random.choice(responses)
                
                return {
                    "success": True,
                    "message": message,
                    "routing_info": {
                        "target": "agent1_direct",
                        "target_agent": "agent1_direct",
                        "action": "handle_greeting",
                        "intent": "greeting"
                    }
                }
            
            elif intent == "general":
                # 일반적인 대화에 대한 응답
                return {
                    "success": True,
                    "message": "안녕하세요! 바른마디병원을 이용해주셔서 감사합니다. 예약, 의료진 추천, 또는 병원 정보 중 어떤 것을 도와드릴까요?",
                    "routing_info": {
                        "target": "agent1_direct",
                        "target_agent": "agent1_direct",
                        "action": "handle_general",
                        "intent": "general"
                    }
                }
            
            else:
                return {
                    "success": False,
                    "message": "죄송합니다. 이해하지 못했습니다.",
                    "suggestion": "예약, 의료진 추천, 또는 병원 정보 중 하나를 선택해주세요."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "응답 생성 중 오류가 발생했습니다."
            }

# LangGraph 노드에서 사용할 수 있는 함수들
def analyze_and_route_user_request(user_input: str, existing_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """사용자 요청 분석 및 라우팅 (LangGraph 노드용)"""
    manager = Agent1Manager()
    return manager.route_to_agent_or_tool(user_input, existing_context)

def format_manager_response(result: Dict[str, Any]) -> str:
    """관리자 에이전트 응답 포맷팅"""
    if not result.get("success"):
        return f"❌ 요청 처리 실패\n{result.get('message', '알 수 없는 오류가 발생했습니다.')}"
    
    # 라우팅 정보가 있으면 해당 에이전트/툴의 응답을 반환
    if "message" in result:
        return result["message"]
    
    # 기본 응답
    return "요청이 처리되었습니다."
