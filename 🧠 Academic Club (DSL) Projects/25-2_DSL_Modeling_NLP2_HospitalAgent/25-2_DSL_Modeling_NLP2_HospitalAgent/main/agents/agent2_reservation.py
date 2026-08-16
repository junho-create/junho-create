"""
에이전트2: 예약 처리 에이전트 (LangChain MCP Adapters 사용)
- LLM을 사용한 사용자 정보 수집
- 증상 수집 후 에이전트3 호출
- LangChain MCP Adapters를 사용한 Supabase 예약 관리
"""
import os
import json
from typing import Dict, List, Any, Optional
from .prompts import RESERVATION_AGENT_PROMPT, RESERVATION_TOOL_SELECTION_PROMPT

class Agent2Reservation:
    """예약 처리 에이전트 - LangChain MCP Adapters 사용"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or self._get_default_llm_client()
        self.required_fields = ["환자명", "전화번호"]  # 실제 스키마 컬럼명 사용
        self.optional_fields = ["성별", "symptoms", "preferred_date", "preferred_time", "preferred_doctor", "notes", "schedule_preference"]
        
        # Tool Calling을 위한 도구 바인딩
        self._setup_tool_calling()
    
    def _get_default_llm_client(self):
        """기본 LLM 클라이언트 설정"""
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                model="gpt-4o-mini",
                temperature=0.1
            )
        except ImportError:
            print("⚠️ ChatOpenAI 클라이언트를 사용할 수 없습니다.")
            return None
    
    def _setup_tool_calling(self):
        """Tool Calling 설정"""
        try:
            print(f"🔍 LLM 클라이언트 확인: {self.llm_client}")
            print(f"🔍 LLM 클라이언트 타입: {type(self.llm_client)}")
            
            if not self.llm_client:
                print("⚠️ LLM 클라이언트가 없어 Tool Calling을 설정할 수 없습니다.")
                self.llm_with_tools = None
                return
            
            # Supabase MCP 도구들 가져오기
            from ..tools.supabase_mcp_tool import get_supabase_tools_for_binding
            self.tools = get_supabase_tools_for_binding()
            print(f"🔍 도구 개수: {len(self.tools)}")
            
            # LLM에 도구 바인딩
            try:
                print(f"🔍 bind_tools 메서드 확인: {hasattr(self.llm_client, 'bind_tools')}")
                self.llm_with_tools = self.llm_client.bind_tools(self.tools)
                print("✅ Tool Calling 바인딩 성공")
            except AttributeError as e:
                print(f"⚠️ bind_tools 메서드가 지원되지 않습니다: {e}")
                print("Fallback 방식 사용")
                self.llm_with_tools = None
            except Exception as e:
                print(f"⚠️ bind_tools 실행 중 오류: {e}")
                self.llm_with_tools = None
            
            if self.llm_with_tools:
                print(f"✅ Tool Calling 설정 완료: {len(self.tools)}개 도구 바인딩")
            else:
                print(f"✅ Fallback 방식 설정 완료: {len(self.tools)}개 도구 준비")
            
        except Exception as e:
            print(f"❌ Tool Calling 설정 실패: {e}")
            self.llm_with_tools = None
            self.tools = []
    
    def process_reservation_request(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """예약 요청 처리 (Tool Calling 사용)"""
        try:
            print(f"🔍 예약 에이전트 시작 - 기존 정보: {existing_info}")
            
            # 관리자 에이전트의 결과가 있는지 먼저 확인
            if existing_info and existing_info.get("status") == "found_reservations":
                print(f"🔍 관리자 에이전트에서 예약 정보를 찾았습니다. 예약 정보 반환")
                return {
                    "success": True,
                    "status": "found_reservations",
                    "message": existing_info.get("message", "예약 정보를 찾았습니다."),
                    "reservations": existing_info.get("reservations", []),
                    "patient_info": existing_info.get("patient_info", {}),
                    "collected_info": existing_info
                }
            
            # 먼저 현재 입력으로 의도 분석
            intent_analysis = self._analyze_reservation_intent(user_input)
            current_intent = intent_analysis.get("action", "create")
            print(f"🔍 현재 입력 의도 분석: {current_intent}")
            
            # 특별한 액션들은 이전 컨텍스트를 무시하고 현재 입력을 우선
            if current_intent in ["rebook", "cancel", "modify"]:
                reservation_type = current_intent
                print(f"🔍 특별 액션으로 의도 설정: {reservation_type}")
            else:
                # 일반적인 경우에만 이전 컨텍스트 확인
                reservation_type = "create"  # 기본값
                
                if existing_info:
                    # 이전 컨텍스트에서 의도 정보 확인
                    if 'previous_intent' in existing_info:
                        reservation_type = existing_info['previous_intent']
                        print(f"🔍 이전 컨텍스트에서 의도 확인: {reservation_type}")
                    elif 'extracted_info' in existing_info and 'action' in existing_info['extracted_info']:
                        reservation_type = existing_info['extracted_info']['action']
                        print(f"🔍 extracted_info에서 의도 확인: {reservation_type}")
                
                # 이전 컨텍스트가 없으면 현재 입력 사용
                if reservation_type == "create":
                    reservation_type = current_intent
                    print(f"🔍 현재 입력으로 의도 설정: {reservation_type}")
            
            print(f"🔍 최종 예약 유형: {reservation_type}")
            
            # 예약 유형에 따른 처리
            if reservation_type == "check":
                return self._handle_reservation_check(user_input, existing_info)
            elif reservation_type == "rebook":
                return self._handle_reservation_rebook(user_input, existing_info)
            elif reservation_type == "cancel":
                return self._handle_reservation_cancel(user_input, existing_info)
            elif reservation_type == "modify":
                return self._handle_reservation_modify(user_input, existing_info)
            else:  # create
                # Tool Calling이 가능한 경우
                if self.llm_with_tools and self.tools:
                    return self._process_with_tool_calling(user_input, existing_info)
                else:
                    # 기존 방식으로 폴백
                    return self._process_without_tool_calling(user_input, existing_info)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "예약 처리 중 오류가 발생했습니다."
            }
    
    def _process_with_tool_calling(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Tool Calling을 사용한 예약 처리"""
        try:
            # 기존 정보와 새 입력 결합
            collected_info = existing_info.copy() if existing_info else {}
            
            # 기존 컨텍스트에서 증상 정보 복원
            if not collected_info.get("symptoms") and existing_info:
                # 컨텍스트에서 증상 정보 찾기
                context_symptoms = existing_info.get("symptoms", [])
                if context_symptoms:
                    collected_info["symptoms"] = context_symptoms
                    print(f"🔍 컨텍스트에서 증상 정보 복원: {context_symptoms}")
                
                # 추천 의료진에서 증상 정보 추출
                if not context_symptoms and existing_info.get("recommended_doctors"):
                    for doctor in existing_info["recommended_doctors"]:
                        if doctor.get("symptoms"):
                            collected_info["symptoms"] = doctor["symptoms"]
                            print(f"🔍 추천 의료진에서 증상 정보 추출: {doctor['symptoms']}")
                            break
            
            # LLM을 사용한 정보 추출
            extraction_result = self._extract_information_with_llm(user_input, collected_info)
            
            if not extraction_result.get("success"):
                return extraction_result
            
            # 추출된 정보 업데이트 (컬럼명 매핑)
            extracted_info = extraction_result.get("extracted_info", {})
            
            # 영어 컬럼명을 실제 스키마 컬럼명으로 매핑
            if "patient_name" in extracted_info:
                extracted_info["환자명"] = extracted_info.pop("patient_name")
            if "patient_phone" in extracted_info:
                extracted_info["전화번호"] = extracted_info.pop("patient_phone")
            if "patient_gender" in extracted_info:
                extracted_info["성별"] = extracted_info.pop("patient_gender")
            
            if collected_info is not None:
                collected_info.update(extracted_info)
            
            # 필수 정보 확인
            missing_info = self._check_missing_information(collected_info)
            
            if missing_info:
                # 더 친근한 메시지 생성
                if len(missing_info) == 1:
                    message = f"예약 정보 수집 중... {missing_info[0]}을(를) 알려주세요."
                elif len(missing_info) == 2:
                    message = f"예약 정보 수집 중... {missing_info[0]}과(와) {missing_info[1]}을(를) 알려주세요."
                else:
                    message = f"예약 정보 수집 중... {', '.join(missing_info[:-1])}과(와) {missing_info[-1]}을(를) 알려주세요."
                
                return {
                    "success": False,
                    "status": "missing_info",
                    "missing_fields": missing_info,
                    "message": message,
                    "collected_info": collected_info
                }
            
            # 증상이 있으면 에이전트3 호출
            print(f"🔍 증상 확인: {collected_info.get('symptoms')}")
            print(f"🔍 collected_info 전체: {collected_info}")
            symptoms = collected_info.get("symptoms", [])
            print(f"🔍 symptoms 타입: {type(symptoms)}")
            print(f"🔍 symptoms 길이: {len(symptoms)}")
            print(f"🔍 symptoms bool 평가: {bool(symptoms)}")
            if symptoms and len(symptoms) > 0:
                rag_result = self._call_agent3_for_symptoms(collected_info["symptoms"])
                if rag_result.get("success"):
                    print(f"✅ 의료진 추천 성공")
                    print(f"📝 추천 진료과: {rag_result.get('department', 'Unknown')}")
                    print(f"👨‍⚕️ 추천 의료진 수: {len(rag_result.get('recommended_doctors', []))}")
                    
                    collected_info["recommended_department"] = rag_result.get("department")
                    collected_info["recommended_doctors"] = rag_result.get("recommended_doctors", [])
                    collected_info["rag_confidence"] = rag_result.get("confidence", 0.0)
                    
                    # 상위 3명 의료진 출력
                    for i, doctor in enumerate(rag_result.get('recommended_doctors', [])[:3], 1):
                        print(f"   {i}. {doctor.get('name', 'Unknown')} - {doctor.get('department', 'Unknown')}")
                        print(f"      추천 근거: {doctor.get('reasoning', 'No reasoning')}")
                    
                    # 첫 번째 추천 의료진의 일정 조회
                    if rag_result.get('recommended_doctors'):
                        first_doctor = rag_result.get('recommended_doctors')[0]
                        doctor_name = first_doctor.get('name', '')
                        if doctor_name:
                            print(f"\n🔍 추천 의료진 '{doctor_name}'의 예약 가능 일정을 조회합니다...")
                            schedule_result = self._get_doctor_schedule(doctor_name)
                            if schedule_result.get("success"):
                                available_schedules = schedule_result.get("data", [])
                                
                                # 사용자 일정 선호도 파싱 및 매칭
                                if collected_info.get("schedule_preference"):
                                    print(f"🔍 사용자 일정 선호도 분석: {collected_info.get('schedule_preference')}")
                                    preference_result = self._parse_schedule_preference(collected_info.get('schedule_preference'))
                                    if preference_result.get("success"):
                                        preference = preference_result.get("parsed_preference", {})
                                        matched_schedules = self._match_schedule_with_preference(available_schedules, preference)
                                        collected_info["available_schedules"] = matched_schedules
                                        collected_info["schedule_preference_parsed"] = preference
                                        print(f"📅 선호도 매칭된 일정: {len(matched_schedules)}건")
                                    else:
                                        collected_info["available_schedules"] = available_schedules
                                        print(f"📅 예약 가능한 일정: {len(available_schedules)}건")
                                else:
                                    collected_info["available_schedules"] = available_schedules
                                    print(f"📅 예약 가능한 일정: {len(available_schedules)}건")
                            else:
                                print(f"⚠️ 일정 조회 실패: {schedule_result.get('message', 'No message')}")
                else:
                    print(f"⚠️ 의료진 추천 실패: {rag_result.get('message', 'No message')}")
            
            # 증상이 있으면 Tool Calling을 사용한 툴 실행
            if symptoms and len(symptoms) > 0:
                # 환자 정보가 없으면 환자 정보를 물어보기
                if not collected_info.get("환자명") or not collected_info.get("전화번호"):
                    print(f"🔍 증상은 있지만 환자 정보가 없어서 need_patient_info 반환")
                    result = {
                        "success": False,
                        "status": "need_patient_info",
                        "collected_info": collected_info,
                        "message": f"✅ 증상이 확인되었습니다.\n\n🩺 **증상:** {', '.join(symptoms)}\n\n👤 **환자 정보를 알려주세요:**\n• 이름과 전화번호를 입력해주세요\n예: 박세현, 010-1234-5678"
                    }
                    print(f"🔍 Tool Calling 경로 반환값: {result}")
                    return result
                
                # 일정 정보가 없으면 일정을 물어보기
                if not collected_info.get("preferred_date") and not collected_info.get("schedule_preference"):
                    print(f"🔍 증상과 환자 정보는 있지만 일정 정보가 없어서 need_schedule 반환")
                    result = {
                        "success": False,
                        "status": "need_schedule",
                        "collected_info": collected_info,
                        "message": f"✅ 환자 정보와 증상이 확인되었습니다.\n\n👤 **환자 정보:**\n• 이름: {collected_info.get('환자명', 'N/A')}\n• 연락처: {collected_info.get('전화번호', 'N/A')}\n\n🩺 **증상:** {', '.join(symptoms)}\n\n📅 **언제 예약하시겠어요?**\n예: 내일, 다음주 월요일, 10월 15일 오후 2시"
                    }
                    print(f"🔍 Tool Calling 경로 반환값: {result}")
                    return result
                
                # 모든 정보가 있으면 일정 조회 및 사용자 선택 처리
                return self._handle_schedule_selection(user_input, collected_info)
            else:
                # 증상이 없으면 증상을 물어보기
                print(f"🔍 Tool Calling 경로에서 증상이 없어서 need_symptoms 반환")
                print(f"🔍 collected_info in tool calling: {collected_info}")
                result = {
                    "success": False,
                    "status": "need_symptoms",
                    "collected_info": collected_info,
                    "message": f"✅ 환자 정보가 확인되었습니다.\n\n👤 **환자 정보:**\n• 이름: {collected_info.get('환자명', 'N/A')}\n• 연락처: {collected_info.get('전화번호', 'N/A')}\n\n🩺 **어떤 증상으로 예약하시나요?**\n예: 무릎이 아파요, 허리가 아프고 디스크가 있어요, 두통과 어지러움이 있어요"
                }
                print(f"🔍 Tool Calling 경로 반환값: {result}")
                return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Tool Calling 처리 중 오류가 발생했습니다."
            }
    
    def _process_without_tool_calling(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Tool Calling 없이 예약 처리 (기존 방식)"""
        try:
            # 기존 정보와 새 입력 결합
            collected_info = existing_info.copy() if existing_info else {}
            
            # LLM을 사용한 정보 추출
            extraction_result = self._extract_information_with_llm(user_input, collected_info)
            
            if not extraction_result.get("success"):
                return extraction_result
            
            # 추출된 정보 업데이트 (컬럼명 매핑)
            extracted_info = extraction_result.get("extracted_info", {})
            
            # 영어 컬럼명을 실제 스키마 컬럼명으로 매핑
            if "patient_name" in extracted_info:
                extracted_info["환자명"] = extracted_info.pop("patient_name")
            if "patient_phone" in extracted_info:
                extracted_info["전화번호"] = extracted_info.pop("patient_phone")
            if "patient_gender" in extracted_info:
                extracted_info["성별"] = extracted_info.pop("patient_gender")
            
            if collected_info is not None:
                collected_info.update(extracted_info)
            
            # 필수 정보 확인
            missing_info = self._check_missing_information(collected_info)
            
            if missing_info:
                # 더 친근한 메시지 생성
                if len(missing_info) == 1:
                    message = f"예약 정보 수집 중... {missing_info[0]}을(를) 알려주세요."
                elif len(missing_info) == 2:
                    message = f"예약 정보 수집 중... {missing_info[0]}과(와) {missing_info[1]}을(를) 알려주세요."
                else:
                    message = f"예약 정보 수집 중... {', '.join(missing_info[:-1])}과(와) {missing_info[-1]}을(를) 알려주세요."
                
                return {
                    "success": False,
                    "status": "missing_info",
                    "missing_fields": missing_info,
                    "message": message,
                    "collected_info": collected_info
                }
            
            # 증상이 있으면 에이전트3 호출
            print(f"🔍 증상 확인: {collected_info.get('symptoms')}")
            print(f"🔍 collected_info 전체: {collected_info}")
            symptoms = collected_info.get("symptoms", [])
            print(f"🔍 symptoms 타입: {type(symptoms)}")
            print(f"🔍 symptoms 길이: {len(symptoms)}")
            print(f"🔍 symptoms bool 평가: {bool(symptoms)}")
            if symptoms and len(symptoms) > 0:
                rag_result = self._call_agent3_for_symptoms(collected_info["symptoms"])
                if rag_result.get("success"):
                    print(f"✅ 의료진 추천 성공")
                    print(f"📝 추천 진료과: {rag_result.get('department', 'Unknown')}")
                    print(f"👨‍⚕️ 추천 의료진 수: {len(rag_result.get('recommended_doctors', []))}")
                    
                    collected_info["recommended_department"] = rag_result.get("department")
                    collected_info["recommended_doctors"] = rag_result.get("recommended_doctors", [])
                    collected_info["rag_confidence"] = rag_result.get("confidence", 0.0)
                    
                    # 상위 3명 의료진 출력
                    for i, doctor in enumerate(rag_result.get('recommended_doctors', [])[:3], 1):
                        print(f"   {i}. {doctor.get('name', 'Unknown')} - {doctor.get('department', 'Unknown')}")
                        print(f"      추천 근거: {doctor.get('reasoning', 'No reasoning')}")
                    
                    # 첫 번째 추천 의료진의 일정 조회
                    if rag_result.get('recommended_doctors'):
                        first_doctor = rag_result.get('recommended_doctors')[0]
                        doctor_name = first_doctor.get('name', '')
                        if doctor_name:
                            print(f"\n🔍 추천 의료진 '{doctor_name}'의 예약 가능 일정을 조회합니다...")
                            schedule_result = self._get_doctor_schedule(doctor_name)
                            if schedule_result.get("success"):
                                available_schedules = schedule_result.get("data", [])
                                
                                # 사용자 일정 선호도 파싱 및 매칭
                                if collected_info.get("schedule_preference"):
                                    print(f"🔍 사용자 일정 선호도 분석: {collected_info.get('schedule_preference')}")
                                    preference_result = self._parse_schedule_preference(collected_info.get('schedule_preference'))
                                    if preference_result.get("success"):
                                        preference = preference_result.get("parsed_preference", {})
                                        matched_schedules = self._match_schedule_with_preference(available_schedules, preference)
                                        collected_info["available_schedules"] = matched_schedules
                                        collected_info["schedule_preference_parsed"] = preference
                                        print(f"📅 선호도 매칭된 일정: {len(matched_schedules)}건")
                                    else:
                                        collected_info["available_schedules"] = available_schedules
                                        print(f"📅 예약 가능한 일정: {len(available_schedules)}건")
                                else:
                                    collected_info["available_schedules"] = available_schedules
                                    print(f"📅 예약 가능한 일정: {len(available_schedules)}건")
                            else:
                                print(f"⚠️ 일정 조회 실패: {schedule_result.get('message', 'No message')}")
                else:
                    print(f"⚠️ 의료진 추천 실패: {rag_result.get('message', 'No message')}")
                
                # 증상이 있을 때는 예약 처리를 계속 진행
                tool_result = self._select_and_execute_supabase_tool(user_input, collected_info)
                
                return {
                    "success": True,
                    "status": "completed",
                    "collected_info": collected_info,
                    "tool_result": tool_result,
                    "message": "예약 처리가 완료되었습니다."
                }
            else:
                # 증상이 없으면 증상을 물어보기
                print(f"🔍 else 블록 실행: 증상이 없어서 need_symptoms 반환")
                print(f"🔍 collected_info in else: {collected_info}")
                result = {
                    "success": False,
                    "status": "need_symptoms",
                    "collected_info": collected_info,
                    "message": f"✅ 환자 정보가 확인되었습니다.\n\n👤 **환자 정보:**\n• 이름: {collected_info.get('환자명', 'N/A')}\n• 연락처: {collected_info.get('전화번호', 'N/A')}\n\n🩺 **어떤 증상으로 예약하시나요?**\n예: 무릎이 아파요, 허리가 아프고 디스크가 있어요, 두통과 어지러움이 있어요"
                }
                print(f"🔍 else 블록 반환값: {result}")
                return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "예약 처리 중 오류가 발생했습니다."
            }
    
    def _extract_information_with_llm(self, user_input: str, existing_info: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용한 정보 추출"""
        try:
            if not self.llm_client:
                return self._fallback_information_extraction(user_input, existing_info)
            
            prompt = RESERVATION_AGENT_PROMPT.format(
                user_input=user_input,
                existing_info=json.dumps(existing_info or {}, ensure_ascii=False, indent=2),
                current_datetime=self._get_current_datetime_info()
            )
            
            response = self.llm_client.invoke([
                {"role": "system", "content": "당신은 바른마디병원의 예약 전문 에이전트입니다. 사용자로부터 예약에 필요한 정보를 수집하는 것이 당신의 임무입니다."},
                {"role": "user", "content": prompt}
            ])
            
            content = response.content
            result = json.loads(content)
            
            return {
                "success": True,
                "extracted_info": result.get("extracted_info", {}),
                "message": "정보 추출이 완료되었습니다."
            }
            
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "LLM 응답을 파싱할 수 없습니다.",
                "message": "다시 시도해주세요."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "정보 추출 중 오류가 발생했습니다."
            }
    
    def _fallback_information_extraction(self, user_input: str, existing_info: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 없이 기본 규칙 기반 정보 추출"""
        extracted_info = {}
        
        # 기본적인 패턴 매칭
        import re
        
        # 이름 추출 (실제 스키마 컬럼명 사용)
        name_patterns = [r"이름[은는]?\s*([가-힣]+)", r"저[는은]\s*([가-힣]+)", r"([가-힣]{2,4})\s*입니다"]
        for pattern in name_patterns:
            match = re.search(pattern, user_input)
            if match:
                extracted_info["환자명"] = match.group(1)
                break
        
        # 전화번호 추출 (실제 스키마 컬럼명 사용)
        phone_pattern = r"(\d{3}[-.]?\d{3,4}[-.]?\d{4})"
        phone_match = re.search(phone_pattern, user_input)
        if phone_match:
            extracted_info["전화번호"] = phone_match.group(1)
        
        # 성별 추출 (실제 스키마 컬럼명 사용)
        if "남자" in user_input or "남성" in user_input:
            extracted_info["성별"] = "남"
        elif "여자" in user_input or "여성" in user_input:
            extracted_info["성별"] = "여"
        
        return {
            "success": True,
            "extracted_info": extracted_info,
            "message": "기본 정보 추출이 완료되었습니다."
        }
    
    def _check_missing_information(self, collected_info: Dict[str, Any]) -> List[str]:
        """누락된 필수 정보 확인"""
        missing_fields = []
        
        for field in self.required_fields:
            if not collected_info.get(field):
                missing_fields.append(field)
        
        return missing_fields
    
    def _call_agent3_for_symptoms(self, symptoms: str) -> Dict[str, Any]:
        """에이전트3 호출 (증상-의료진 매핑)"""
        try:
            from .agent3_rag import Agent3RAG
            
            rag_agent = Agent3RAG()
            
            # symptoms를 리스트로 변환
            if isinstance(symptoms, str):
                symptoms_list = [symptoms]
            else:
                symptoms_list = symptoms
            
            print(f"🔍 Agent3 호출: 증상 {symptoms_list}")
            result = rag_agent.recommend_doctors(symptoms_list, f"증상: {', '.join(symptoms_list)}")
            print(f"📝 Agent3 결과: {result}")
            
            return result
            
        except Exception as e:
            print(f"❌ Agent3 호출 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "의료진 추천 중 오류가 발생했습니다."
            }
    
    def _select_and_execute_supabase_tool(self, user_input: str, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain MCP Adapters를 사용한 Supabase 툴 선택 및 실행"""
        try:
            # LLM을 사용한 툴 선택
            tool_selection = self._llm_based_tool_selection(user_input, collected_info)
            
            if not tool_selection.get("success"):
                return tool_selection
            
            # 선택된 툴 실행
            selected_tool = tool_selection["selected_tool"]
            parameters = tool_selection["parameters"]
            
            return self._execute_supabase_tool(selected_tool, parameters)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Supabase 툴 실행 중 오류가 발생했습니다."
            }
    
    def _llm_based_tool_selection(self, user_input: str, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용한 툴 선택"""
        try:
            if not self.llm_client:
                return self._fallback_tool_selection(user_input, collected_info)
            
            prompt = RESERVATION_TOOL_SELECTION_PROMPT.format(
                user_input=user_input,
                collected_info=json.dumps(collected_info, ensure_ascii=False, indent=2),
                current_datetime=self._get_current_datetime_info()
            )
            
            response = self.llm_client.invoke([
                {"role": "system", "content": "당신은 바른마디병원의 예약 도구 선택 전문가입니다. 사용자의 요청에 따라 적절한 Supabase 도구를 선택하는 것이 당신의 임무입니다."},
                {"role": "user", "content": prompt}
            ])
            
            content = response.content
            result = json.loads(content)
            
            return {
                "success": True,
                "selected_tool": result.get("selected_tool"),
                "parameters": result.get("parameters", {}),
                "reasoning": result.get("reasoning", "")
            }
            
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "LLM 응답을 파싱할 수 없습니다.",
                "message": "다시 시도해주세요."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "툴 선택 중 오류가 발생했습니다."
            }
    
    def _fallback_tool_selection(self, user_input: str, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """기본 규칙 기반 툴 선택"""
        user_input_lower = user_input.lower()
        
        if any(keyword in user_input_lower for keyword in ["예약", "예약하고", "예약하고 싶어"]):
            return {
                "success": True,
                "selected_tool": "supabase_create_tool",
                "parameters": {
                    "table": "예약정보",
                    "data": collected_info
                }
            }
        elif any(keyword in user_input_lower for keyword in ["조회", "확인", "내 예약"]):
            return {
                "success": True,
                "selected_tool": "supabase_read_tool",
                "parameters": {
                    "table": "예약정보",
                    "filters": {"환자명": collected_info.get("환자명")}
                }
            }
        elif any(keyword in user_input_lower for keyword in ["변경", "수정", "바꾸고"]):
            return {
                "success": True,
                "selected_tool": "supabase_update_tool",
                "parameters": {
                    "table": "예약정보",
                    "filters": {"환자명": collected_info.get("환자명")},
                    "data": collected_info
                }
            }
        elif any(keyword in user_input_lower for keyword in ["취소", "삭제", "취소하고"]):
            return {
                "success": True,
                "selected_tool": "supabase_delete_tool",
                "parameters": {
                    "table": "예약정보",
                    "filters": {"환자명": collected_info.get("환자명")}
                }
            }
        else:
            return {
                "success": True,
                "selected_tool": "supabase_read_tool",
                "parameters": {
                    "table": "예약정보",
                    "filters": {}
                }
            }
    
    def _execute_supabase_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain MCP Adapters를 사용한 Supabase 툴 실행"""
        try:
            from ..tools.supabase_mcp_tool import (
                SupabaseReadTool, SupabaseCreateTool, 
                SupabaseUpdateTool, SupabaseDeleteTool
            )
            
            if tool_name == "supabase_read_tool":
                tool = SupabaseReadTool()
                result = tool._run(
                    table=parameters.get("table", "예약정보"),
                    filters=parameters.get("filters", {}),
                    run_manager=None
                )
            elif tool_name == "supabase_create_tool":
                tool = SupabaseCreateTool()
                result = tool._run(
                    table=parameters.get("table", "예약정보"),
                    data=parameters.get("data", {}),
                    run_manager=None
                )
            elif tool_name == "supabase_update_tool":
                tool = SupabaseUpdateTool()
                result = tool._run(
                    table=parameters.get("table", "예약정보"),
                    filters=parameters.get("filters", {}),
                    data=parameters.get("data", {}),
                    run_manager=None
                )
            elif tool_name == "supabase_delete_tool":
                tool = SupabaseDeleteTool()
                result = tool._run(
                    table=parameters.get("table", "예약정보"),
                    filters=parameters.get("filters", {}),
                    run_manager=None
                )
            else:
                return {
                    "success": False,
                    "error": f"지원하지 않는 툴: {tool_name}",
                    "message": "알 수 없는 툴입니다."
                }
            
            result_data = json.loads(result)
            return {
                "success": result_data.get("success", False),
                "data": result_data.get("data"),
                "message": result_data.get("message", "툴 실행이 완료되었습니다."),
                "tool_name": tool_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                 "message": f"Supabase 툴 실행 중 오류가 발생했습니다: {tool_name}"
             }
    
    def _handle_schedule_selection(self, user_input: str, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """일정 조회 및 사용자 선택 처리"""
        try:
            # 이미 일정이 조회되었는지 확인
            available_schedules = collected_info.get('available_schedules', [])
            
            if not available_schedules:
                # 일정 조회가 필요한 경우
                print("🔍 일정 조회가 필요합니다")
                return self._query_available_schedules(collected_info)
            
            # 사용자가 일정을 선택했는지 확인
            selected_schedule = self._extract_schedule_selection(user_input, available_schedules)
            
            if selected_schedule:
                # 선택된 일정으로 예약 처리
                print(f"🔍 선택된 일정: {selected_schedule}")
                return self._process_reservation_with_schedule(collected_info, selected_schedule)
            else:
                # 사용자에게 일정 선택 요청
                return self._present_schedule_options(collected_info, available_schedules)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "일정 선택 처리 중 오류가 발생했습니다."
            }
    
    def _query_available_schedules(self, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """가용 일정 조회"""
        try:
            # 의료진 정보 확인
            preferred_doctor = collected_info.get('preferred_doctor', '')
            if not preferred_doctor:
                # 추천된 의료진 목록을 사용자에게 표시
                recommended_doctors = collected_info.get('recommended_doctors', [])
                if recommended_doctors:
                    message = "👨‍⚕️ **추천 의료진을 선택해주세요:**\n\n"
                    for i, doctor in enumerate(recommended_doctors[:5], 1):  # 최대 5명만 표시
                        doctor_name = doctor.get('name', '')
                        department = doctor.get('department', '')
                        reasoning = doctor.get('reasoning', '')
                        
                        # 클리닉명이 아닌 실제 의사명만 표시
                        if '/' in doctor_name:
                            display_name = doctor_name.split('/')[0]  # "양재혁/D003" -> "양재혁"
                        else:
                            display_name = doctor_name
                        
                        message += f"{i}. **{display_name}** ({department})\n"
                        if reasoning:
                            # 추천 근거를 간단히 표시
                            reason_short = reasoning.split('/')[0] if '/' in reasoning else reasoning
                            message += f"   💡 {reason_short}\n"
                        message += "\n"
                    
                    message += "**원하시는 의료진 번호를 선택해주세요** (예: 1번, 2번 등)"
                else:
                    message = "예약할 의료진을 선택해주세요."
                
                return {
                    "success": False,
                    "status": "need_doctor_selection",
                    "collected_info": collected_info,
                    "message": message
                }
            
            # 일정 조회를 위한 Tool Calling 실행
            tool_result = self._execute_with_tool_calling("일정 조회", collected_info)
            
            if tool_result.get("success"):
                # 조회된 일정을 collected_info에 저장
                collected_info['available_schedules'] = tool_result.get('data', [])
                
                if collected_info['available_schedules']:
                    return self._present_schedule_options(collected_info, collected_info['available_schedules'])
                else:
                    return {
                        "success": False,
                        "status": "no_available_schedules",
                        "collected_info": collected_info,
                        "message": "죄송합니다. 선택하신 의료진의 가용 일정이 없습니다. 다른 의료진을 선택하시거나 다른 날짜를 문의해주세요."
                    }
            else:
                return {
                    "success": False,
                    "status": "schedule_query_failed",
                    "collected_info": collected_info,
                    "message": "일정 조회 중 오류가 발생했습니다."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "일정 조회 중 오류가 발생했습니다."
            }
    
    def _present_schedule_options(self, collected_info: Dict[str, Any], available_schedules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """사용자에게 일정 옵션 제시"""
        try:
            # 일정 옵션 포맷팅
            schedule_options = []
            for i, schedule in enumerate(available_schedules[:5], 1):  # 최대 5개 옵션
                date = schedule.get('날짜', 'N/A')
                time = schedule.get('시간', 'N/A')
                doctor = schedule.get('의료진', 'N/A')
                schedule_options.append(f"{i}. {date} {time} ({doctor})")
            
            options_text = "\n".join(schedule_options)
            
            return {
                "success": False,
                "status": "need_schedule_selection",
                "collected_info": collected_info,
                "message": f"📅 **예약 가능한 일정**\n\n{options_text}\n\n**원하시는 일정 번호를 선택해주세요** (예: 1번, 2번 등)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "일정 옵션 제시 중 오류가 발생했습니다."
            }
    
    def _extract_schedule_selection(self, user_input: str, available_schedules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """사용자 입력에서 일정 선택 추출"""
        try:
            # 숫자 패턴 매칭 (1번, 2번, 1, 2 등)
            import re
            number_match = re.search(r'(\d+)번?', user_input)
            
            if number_match:
                selected_number = int(number_match.group(1))
                if 1 <= selected_number <= len(available_schedules):
                    return available_schedules[selected_number - 1]
            
            # 직접적인 일정 정보 매칭 (날짜, 시간)
            for schedule in available_schedules:
                date = schedule.get('날짜', '')
                time = schedule.get('시간', '')
                if date in user_input or time in user_input:
                    return schedule
            
            return None
            
        except Exception as e:
            print(f"일정 선택 추출 오류: {e}")
            return None
    
    def _process_reservation_with_schedule(self, collected_info: Dict[str, Any], selected_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """선택된 일정으로 예약 처리"""
        try:
            # 예약 정보 구성
            reservation_data = {
                "환자명": collected_info.get('환자명', ''),
                "전화번호": collected_info.get('전화번호', ''),
                "성별": collected_info.get('성별', ''),
                "증상": ', '.join(collected_info.get('symptoms', [])),
                "의료진": selected_schedule.get('의료진', ''),
                "예약날짜": selected_schedule.get('날짜', ''),
                "예약시간": selected_schedule.get('시간', ''),
                "일정ID": selected_schedule.get('일정ID', ''),
                "DocID": selected_schedule.get('DocID', '')
            }
            
            # 예약 생성 Tool Calling 실행
            # 1. 환자 조회
            patient_lookup_result = self._execute_with_tool_calling("환자 조회", reservation_data)
            
            # 2. 환자가 없으면 새 환자 생성
            patient_id = None
            if patient_lookup_result.get("success") and not patient_lookup_result.get("data"):
                print("🔍 환자가 없으므로 새 환자 생성")
                # 환자 생성용 데이터 구성 (필수 필드 포함)
                patient_data = {
                    "환자PWD": "123456",  # 필수 필드
                    "이름": reservation_data["환자명"],
                    "전화번호": reservation_data["전화번호"],
                    "성별": reservation_data.get("성별", ""),
                    "생년월일": None,
                    "이메일": "",
                    "주소": ""
                }
                # 환자 생성 직접 도구 호출
                patient_create_result = self._create_patient_direct(patient_data)
                if not patient_create_result.get("success"):
                    print(f"⚠️ 환자 생성 실패: {patient_create_result.get('message', '알 수 없는 오류')}")
                    # 환자 생성 실패해도 예약은 진행 (기존 환자로 처리)
                else:
                    print("✅ 환자 생성 성공")
                    # 새로 생성된 환자의 ID 추출
                    if patient_create_result.get("data") and len(patient_create_result["data"]) > 0:
                        patient_id = patient_create_result["data"][0].get("환자ID")
                        print(f"🔍 새로 생성된 환자ID: {patient_id}")
                    else:
                        print("⚠️ 환자 생성 성공했지만 환자ID를 찾을 수 없음")
            else:
                # 기존 환자가 있는 경우
                if patient_lookup_result.get("success") and patient_lookup_result.get("data"):
                    patient_id = patient_lookup_result["data"][0].get("환자ID")
                    print(f"🔍 기존 환자ID: {patient_id}")
            
            # 3. 예약 생성
            print("🔍 예약 생성 시작")
            
            # 예약 정보 생성용 데이터 구성 (올바른 컬럼명 사용)
            reservation_create_data = {
                "환자ID": patient_id,  # 환자ID 추가
                "환자명": reservation_data["환자명"],
                "예약시간": reservation_data["예약시간"],
                "진료일자": reservation_data["예약날짜"],
                "진료시간": reservation_data["예약시간"],  # 진료시간 추가
                "종료시간": self._calculate_end_time(reservation_data["예약시간"]),  # 종료시간 계산
                "DocID": reservation_data["DocID"],
                "예약구분": "외래 진료",  # 예약구분 수정
                "예약상태": "확정",
                "예약일자": self._get_current_date(),  # 예약일자 추가
                "수정일자": self._get_current_date()  # 수정일자 추가
            }
            
            print(f"🔍 예약 생성 데이터: {reservation_create_data}")
            # 예약 생성 직접 도구 호출
            reservation_create_result = self._create_reservation_direct(reservation_create_data)
            
            # 4. 가용일정 업데이트 (예약가능여부를 N으로 변경)
            print("🔍 가용일정 업데이트 시작")
            schedule_update_result = self._execute_with_tool_calling("가용일정 업데이트", {
                "일정ID": selected_schedule.get('일정ID', ''),
                "예약가능여부": "N"
            })
            
            # 모든 결과를 통합
            tool_result = {
                "success": patient_lookup_result.get("success") and reservation_create_result.get("success") and schedule_update_result.get("success"),
                "patient_lookup": patient_lookup_result,
                "reservation_create": reservation_create_result,
                "schedule_update": schedule_update_result,
                "message": "예약 처리가 완료되었습니다."
            }
            
            if tool_result.get("success"):
                return {
                    "success": True,
                    "status": "completed",
                    "collected_info": collected_info,
                    "reservation_data": reservation_data,
                    "tool_result": tool_result,
                    "message": f"✅ **예약이 완료되었습니다!**\n\n📋 **예약 정보**\n• 환자: {reservation_data['환자명']}\n• 연락처: {reservation_data['전화번호']}\n• 의료진: {reservation_data['의료진']}\n• 예약일시: {reservation_data['예약날짜']} {reservation_data['예약시간']}\n• 증상: {reservation_data['증상']}\n\n🏥 바른마디병원에서 만나요!"
                }
            else:
                return {
                    "success": False,
                    "status": "reservation_failed",
                    "collected_info": collected_info,
                    "message": f"예약 처리 중 오류가 발생했습니다: {tool_result.get('message', '알 수 없는 오류')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "예약 처리 중 오류가 발생했습니다."
            }

    def _calculate_end_time(self, start_time: str) -> str:
        """진료 시작시간에서 30분 후 종료시간 계산"""
        try:
            from datetime import datetime, timedelta
            
            # 시간 파싱 (HH:MM 형식)
            time_obj = datetime.strptime(start_time, "%H:%M")
            
            # 30분 추가
            end_time_obj = time_obj + timedelta(minutes=30)
            
            # 문자열로 변환
            return end_time_obj.strftime("%H:%M")
            
        except Exception as e:
            print(f"⚠️ 종료시간 계산 오류: {e}")
            # 기본값으로 30분 후 반환
            return "11:30"  # 기본값
    
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
    
    def _get_current_date(self) -> str:
        """현재 날짜를 YYYY-MM-DD 형식으로 반환"""
        try:
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"⚠️ 현재 날짜 조회 오류: {e}")
            return "2025-09-30"  # 기본값
    
    def _check_returning_patient(self, patient_name: str, phone_number: str) -> Dict[str, Any]:
        """재진 환자 확인 및 이전 의사 정보 조회"""
        try:
            print(f"🔍 재진 환자 확인: {patient_name}, {phone_number}")
            
            # 1단계: 환자 정보 조회
            patient_lookup_result = self._lookup_patient_info({
                "환자명": patient_name,
                "전화번호": phone_number
            })
            
            if not patient_lookup_result.get("success") or not patient_lookup_result.get("data"):
                return {
                    "success": False,
                    "is_returning_patient": False,
                    "message": "환자 정보를 찾을 수 없습니다."
                }
            
            patient_data = patient_lookup_result["data"][0]
            patient_id = patient_data.get("환자ID")
            
            # 2단계: 이전 예약 기록 조회
            reservation_result = self._lookup_reservations_by_patient_id(patient_id)
            
            if not reservation_result.get("success"):
                return {
                    "success": False,
                    "is_returning_patient": False,
                    "message": "예약 기록 조회 중 오류가 발생했습니다."
                }
            
            reservations = reservation_result.get("data", [])
            
            if not reservations:
                return {
                    "success": True,
                    "is_returning_patient": False,
                    "message": "이전 예약 기록이 없습니다. 초진 환자입니다.",
                    "patient_id": patient_id
                }
            
            # 3단계: 가장 최근 예약에서 의사 정보 추출
            recent_reservation = reservations[0]  # 가장 최근 예약
            doc_id = recent_reservation.get("DocID")
            doctor_name = recent_reservation.get("의료진명", "")
            
            print(f"🔍 재진 환자 확인 완료: DocID {doc_id}, 의사명 {doctor_name}")
            
            return {
                "success": True,
                "is_returning_patient": True,
                "message": f"{patient_name}님은 재진 환자입니다. 이전에 {doctor_name} 의사님께 진료받으셨습니다.",
                "patient_id": patient_id,
                "previous_doctor": {
                    "DocID": doc_id,
                    "의료진명": doctor_name
                },
                "total_visits": len(reservations)
            }
            
        except Exception as e:
            print(f"❌ 재진 환자 확인 오류: {e}")
            return {
                "success": False,
                "is_returning_patient": False,
                "error": str(e),
                "message": "재진 환자 확인 중 오류가 발생했습니다."
            }
    
    def _extract_patient_info_from_input(self, user_input: str) -> Dict[str, Any]:
        """사용자 입력에서 환자 정보 추출"""
        import re
        
        patient_info = {}
        
        # 전화번호 패턴 매칭 (010-1234-5678, 01012345678 등)
        phone_patterns = [
            r'01[0-9]-?\d{3,4}-?\d{4}',
            r'01[0-9]\s?\d{3,4}\s?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, user_input)
            if phone_match:
                phone = phone_match.group().replace('-', '').replace(' ', '')
                # 전화번호 포맷팅 (010-1234-5678)
                if len(phone) == 11:
                    phone = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
                patient_info["전화번호"] = phone
                break
        
        # 이름 추출 (한글 2-4자)
        name_pattern = r'[가-힣]{2,4}'
        name_matches = re.findall(name_pattern, user_input)
        
        # 전화번호가 있으면 이름도 추출 시도
        if patient_info.get("전화번호") and name_matches:
            # 전화번호 앞뒤로 있는 이름을 우선 선택
            for name in name_matches:
                if len(name) >= 2:  # 2글자 이상
                    patient_info["환자명"] = name
                    break
        
        print(f"🔍 입력에서 추출한 환자 정보: {patient_info}")
        return patient_info
    
    def _handle_reservation_rebook(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """재예약 처리"""
        try:
            print(f"🔍 재예약 처리 시작: {user_input}")
            print(f"🔍 기존 정보: {existing_info}")
            
            # 이전 컨텍스트에서 재예약 정보 확인
            rebooking_context = existing_info.get("rebooking_context") if existing_info else None
            
            # 재예약 컨텍스트가 없으면 현재 입력에서 환자 정보 추출 시도
            if not rebooking_context:
                print(f"🔍 재예약 컨텍스트가 없음. 현재 입력에서 환자 정보 추출 시도")
                
                # 현재 입력에서 환자 정보 추출
                patient_info = self._extract_patient_info_from_input(user_input)
                if patient_info.get("환자명") and patient_info.get("전화번호"):
                    print(f"🔍 현재 입력에서 환자 정보 추출 성공: {patient_info}")
                    
                    # 환자 정보로 예약 조회해서 이전 의사 정보 가져오기
                    patient_lookup = self._lookup_patient_info(patient_info)
                    if patient_lookup.get("success"):
                        patient_data_list = patient_lookup["data"]
                        if patient_data_list and len(patient_data_list) > 0:
                            patient_data = patient_data_list[0]  # 첫 번째 환자 정보 사용
                            reservations_lookup = self._lookup_reservations_by_patient_id(patient_data["환자ID"])
                        else:
                            return {
                                "success": False,
                                "status": "patient_not_found",
                                "message": "환자 정보를 찾을 수 없습니다. 정확한 이름과 전화번호를 확인해주세요.",
                                "collected_info": existing_info or {}
                            }
                        
                            if reservations_lookup.get("success") and reservations_lookup.get("data"):
                                reservations = reservations_lookup["data"]
                                # 가장 최근 예약에서 의사 정보 추출
                                latest_reservation = reservations[0]
                                doc_id = latest_reservation.get("DocID")
                                
                                if doc_id:
                                    # 의사 정보 조회
                                    doctor_tool = SupabaseReadTool()
                                    doctor_result = doctor_tool._run(
                                        table="의사",
                                        filters={"DocID": doc_id}
                                    )
                                    import json
                                    doctor_data = json.loads(doctor_result)
                                    doctor_name = ""
                                    if doctor_data.get("success") and doctor_data.get("data"):
                                        doctor_name = doctor_data["data"][0].get("의료진명", f"DocID {doc_id}")
                                    
                                    # 재예약 컨텍스트 생성
                                    rebooking_context = {
                                        "previous_doctor": {
                                            "name": doctor_name,
                                            "DocID": doc_id
                                        },
                                        "patient_info": patient_data,
                                        "is_rebooking": True
                                    }
                                    print(f"🔍 재예약 컨텍스트 생성: {rebooking_context}")
                                else:
                                    return {
                                        "success": False,
                                        "status": "no_previous_reservation",
                                        "message": "이전 예약 정보를 찾을 수 없습니다. 새로 예약해주세요.",
                                        "collected_info": existing_info or {}
                                    }
                            else:
                                return {
                                    "success": False,
                                    "status": "no_previous_reservation",
                                    "message": "이전 예약 정보를 찾을 수 없습니다. 새로 예약해주세요.",
                                    "collected_info": existing_info or {}
                                }
                    else:
                        return {
                            "success": False,
                            "status": "patient_not_found",
                            "message": "환자 정보를 찾을 수 없습니다. 정확한 이름과 전화번호를 확인해주세요.",
                            "collected_info": existing_info or {}
                        }
                else:
                    # 환자 정보가 충분하지 않으면 요청
                    return {
                        "success": False,
                        "status": "need_patient_info",
                        "message": "재예약을 위해 환자명과 전화번호를 알려주세요.",
                        "collected_info": existing_info or {}
                    }
            
            # 재예약 컨텍스트에서 정보 추출
            previous_doctor = rebooking_context.get("previous_doctor", {})
            patient_info = rebooking_context.get("patient_info", {})
            
            doctor_name = previous_doctor.get("name", "")
            doc_id = previous_doctor.get("DocID", "")
            patient_name = patient_info.get("환자명", "")
            patient_phone = patient_info.get("전화번호", "")
            
            if not doctor_name or not doc_id:
                return {
                    "success": False,
                    "status": "doctor_info_missing",
                    "message": "이전 의사 정보를 찾을 수 없습니다. 새로 예약해주세요.",
                    "collected_info": existing_info or {}
                }
            
            print(f"🔍 재예약 대상: {doctor_name} 의사님 (DocID: {doc_id})")
            
            # 이전 의사로 예약 가능한 일정 조회
            schedule_result = self._get_doctor_schedule_by_doc_id(doc_id)
            
            if not schedule_result.get("success"):
                return {
                    "success": False,
                    "status": "schedule_lookup_failed",
                    "message": f"{doctor_name} 의사님의 예약 가능한 일정을 조회할 수 없습니다.",
                    "collected_info": existing_info or {}
                }
            
            available_schedules = schedule_result.get("data", [])
            
            if not available_schedules:
                return {
                    "success": False,
                    "status": "no_available_schedules",
                    "message": f"{doctor_name} 의사님의 예약 가능한 일정이 없습니다.",
                    "collected_info": existing_info or {}
                }
            
            # 재예약 정보 구성
            rebooking_info = {
                "환자명": patient_name,
                "전화번호": patient_phone,
                "selected_doctor": {
                    "name": doctor_name,
                    "DocID": doc_id,
                    "department": "이전 진료과",
                    "is_rebooking": True
                },
                "available_schedules": available_schedules,
                "is_rebooking": True,
                "rebooking_context": rebooking_context
            }
            
            print(f"📅 재예약 가능한 일정: {len(available_schedules)}건")
            
            # 사용자 입력 분석 - 자연어 요청 처리
            user_request_lower = user_input.lower()
            
            # 가장 빠른 일정 요청 (더 구체적인 키워드 우선)
            if any(keyword in user_request_lower for keyword in ["가장 빠른", "빠른", "빨리", "최대한 빨리", "얼른"]):
                # 날짜와 시간 기준으로 가장 빠른 일정 찾기
                earliest_schedule = None
                earliest_datetime = None
                
                for schedule in available_schedules:
                    schedule_date = schedule.get('날짜', '')
                    schedule_time = schedule.get('시간', '')
                    
                    if schedule_date and schedule_time:
                        try:
                            from datetime import datetime
                            schedule_datetime = datetime.strptime(f"{schedule_date} {schedule_time}", "%Y-%m-%d %H:%M")
                            if earliest_datetime is None or schedule_datetime < earliest_datetime:
                                earliest_datetime = schedule_datetime
                                earliest_schedule = schedule
                        except:
                            continue
                
                if earliest_schedule:
                    # 가장 빠른 일정으로 자동 예약 진행
                    return self._process_automatic_rebooking(rebooking_info, earliest_schedule)
                else:
                    message = "❌ 예약 가능한 일정을 찾을 수 없습니다."
                    
            # 특정 시간 요청 (예: "14:00 시간으로")
            elif ":" in user_input and any(keyword in user_request_lower for keyword in ["시간", "시"]):
                requested_time = None
                for word in user_input.split():
                    if ":" in word and len(word) <= 6:  # 시간 형식 체크
                        try:
                            requested_time = word
                            break
                        except:
                            continue
                
                if requested_time:
                    # 요청한 시간과 일치하는 일정 찾기
                    matching_schedules = []
                    for schedule in available_schedules:
                        if schedule.get('시간', '').startswith(requested_time):
                            matching_schedules.append(schedule)
                    
                    if matching_schedules:
                        # 첫 번째 매칭 일정으로 자동 예약
                        return self._process_automatic_rebooking(rebooking_info, matching_schedules[0])
                    else:
                        message = f"❌ {requested_time} 시간에 예약 가능한 일정이 없습니다.\n\n📅 **예약 가능한 일정:**\n"
                        for i, schedule in enumerate(available_schedules[:5], 1):
                            message += f"{i}. {schedule.get('날짜', 'N/A')} {schedule.get('시간', 'N/A')}\n"
                        message += "\n**원하시는 일정 번호를 선택해주세요**"
                else:
                    message = "❌ 시간 형식을 인식할 수 없습니다. (예: 14:00)"
                    
            # 일정 목록을 보여달라는 요청
            elif any(keyword in user_request_lower for keyword in ["일정", "몇개", "몇 건", "보여", "알려", "목록"]):
                schedule_list = ""
                for i, schedule in enumerate(available_schedules[:5], 1):
                    schedule_list += f"{i}. {schedule.get('날짜', 'N/A')} {schedule.get('시간', 'N/A')}\n"
                
                message = f"📅 **{doctor_name} 의사님의 예약 가능한 일정**\n\n{schedule_list}\n**원하시는 일정 번호를 선택해주세요** (예: 1번, 2번 등)"
                    
            else:
                # 기본 메시지
                message = f"✅ **재예약 준비 완료**\n\n👤 **환자 정보:**\n• 이름: {patient_name}\n• 연락처: {patient_phone}\n\n👨‍⚕️ **선택된 의사:** {doctor_name} 의사님\n📅 **예약 가능한 일정:** {len(available_schedules)}건\n\n원하시는 일정을 선택해주세요:\n예: '가장 빠른 시간으로', '14:00 시간으로', '일정 보여줘'"
            
            return {
                "success": True,
                "status": "rebooking_ready",
                "message": message,
                "collected_info": rebooking_info
            }
            
        except Exception as e:
            print(f"❌ 재예약 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "재예약 처리 중 오류가 발생했습니다."
            }
    
    def _process_automatic_rebooking(self, rebooking_info: Dict[str, Any], selected_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """자동 재예약 처리 (가장 빠른 일정 또는 특정 시간 선택)"""
        try:
            print(f"🔍 자동 재예약 처리 시작: {selected_schedule}")
            
            # 선택된 일정 정보 업데이트
            rebooking_info['selected_schedule'] = selected_schedule
            rebooking_info['preferred_date'] = selected_schedule.get('날짜', '')
            rebooking_info['preferred_time'] = selected_schedule.get('시간', '')
            
            # 예약 생성 데이터 준비
            reservation_data = {
                "환자명": rebooking_info.get('환자명', ''),
                "전화번호": rebooking_info.get('전화번호', ''),
                "DocID": str(selected_schedule.get('DocID', '')),  # 문자열로 변환
                "진료일자": selected_schedule.get('날짜', ''),
                "진료시간": selected_schedule.get('시간', ''),
                "예약구분": "외래 진료",
                "is_rebooking": True
            }
            
            # 예약 생성 실행
            result = self._process_reservation_with_schedule(rebooking_info, selected_schedule)
            
            if result.get("success"):
                return {
                    "success": True,
                    "status": "reservation_completed",
                    "message": f"✅ **재예약 완료!**\n\n👤 **환자:** {rebooking_info.get('환자명', '')}\n👨‍⚕️ **의사:** {rebooking_info['selected_doctor']['name']}\n📅 **진료일시:** {selected_schedule.get('날짜', '')} {selected_schedule.get('시간', '')}\n\n🎉 재예약이 성공적으로 완료되었습니다!",
                    "reservation_info": result.get("reservation_info", {}),
                    "collected_info": rebooking_info
                }
            else:
                return {
                    "success": False,
                    "status": "reservation_failed",
                    "message": f"❌ 재예약 처리 중 오류가 발생했습니다.\n{result.get('message', '알 수 없는 오류')}",
                    "collected_info": rebooking_info
                }
                
        except Exception as e:
            print(f"❌ 자동 재예약 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "자동 재예약 처리 중 오류가 발생했습니다."
            }

    def _create_patient_direct(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """환자 생성 직접 도구 호출"""
        try:
            from main.tools.supabase_mcp_tool import SupabaseCreateTool
            
            # 환자 생성 도구 직접 호출
            create_tool = SupabaseCreateTool()
            result = create_tool._run(
                table="환자정보",
                data=patient_data
            )
            
            # 결과 파싱
            import json
            try:
                parsed_result = json.loads(result)
                return parsed_result
            except:
                return {
                    "success": False,
                    "error": "결과 파싱 실패",
                    "message": result
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"환자 생성 중 오류: {e}"
            }
    
    def _create_reservation_direct(self, reservation_data: Dict[str, Any]) -> Dict[str, Any]:
        """예약 생성 직접 도구 호출"""
        try:
            from main.tools.supabase_mcp_tool import SupabaseCreateTool
            
            # 예약 생성 도구 직접 호출
            create_tool = SupabaseCreateTool()
            result = create_tool._run(
                table="예약정보",
                data=reservation_data
            )
            
            # 결과 파싱
            import json
            try:
                parsed_result = json.loads(result)
                return parsed_result
            except:
                return {
                    "success": False,
                    "error": "결과 파싱 실패",
                    "message": result
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"예약 생성 중 오류: {e}"
            }

    def _execute_with_tool_calling(self, user_input: str, collected_info: Dict[str, Any]) -> Dict[str, Any]:
        """Tool Calling을 사용한 툴 실행"""
        try:
            if not self.llm_with_tools:
                return {
                    "success": False,
                    "error": "Tool Calling이 설정되지 않았습니다.",
                    "message": "Tool Calling을 사용할 수 없습니다."
                }
            
            # 사용자 입력과 수집된 정보를 결합한 프롬프트 생성
            prompt = f"""
사용자 요청: {user_input}

수집된 정보:
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 적절한 Supabase 도구를 사용하여 요청을 처리해주세요.
"""
            
            # LLM이 툴을 자동으로 선택하고 실행
            response = self.llm_with_tools.invoke(prompt)
            
            # 툴 호출 결과 처리
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = []
                for tool_call in response.tool_calls:
                    # tool_call이 딕셔너리인 경우 처리
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                    else:
                        tool_name = getattr(tool_call, 'name', '')
                        tool_args = getattr(tool_call, 'args', {})
                    
                    # 툴 실행
                    tool_result = self._execute_tool_by_name(tool_name, tool_args)
                    tool_results.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "result": tool_result
                    })
                
                return {
                    "success": True,
                    "tool_calls": tool_results,
                    "message": "Tool Calling을 사용한 처리가 완료되었습니다."
                }
            else:
                # 툴 호출이 없는 경우 텍스트 응답
                return {
                    "success": True,
                    "response": response.content,
                    "message": "LLM 응답이 반환되었습니다."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Tool Calling 실행 중 오류가 발생했습니다."
            }
    
    def _execute_tool_by_name(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """툴 이름으로 툴 실행"""
        try:
            # 툴 이름에 따라 적절한 툴 실행
            if tool_name == "supabase_read_direct":
                from ..tools.supabase_mcp_tool import SupabaseReadTool
                tool = SupabaseReadTool()
                
                # 실제 컬럼명으로 매핑
                filters = tool_args.get("filters", {})
                if "patient_name" in filters:
                    filters["환자명"] = filters.pop("patient_name")
                if "patient_phone" in filters:
                    filters["전화번호"] = filters.pop("patient_phone")
                
                # 전화번호가 있으면 환자정보에서 환자ID 조회 후 예약정보 조회
                if "전화번호" in filters:
                    patient_result = self._find_patient_by_phone(filters["전화번호"])
                    if patient_result.get("success") and patient_result.get("data"):
                        patient_id = patient_result["data"][0]["환자ID"]
                        # 전화번호 필터 제거하고 환자ID로 조회
                        filters.pop("전화번호", None)
                        filters["환자ID"] = patient_id
                
                result = tool._run(
                    table=tool_args.get("table", "예약정보"),
                    filters=filters,
                    run_manager=None
                )
            elif tool_name == "supabase_create_direct":
                from ..tools.supabase_mcp_tool import SupabaseCreateTool
                tool = SupabaseCreateTool()
                
                # 실제 컬럼명으로 매핑
                data = tool_args.get("data", {})
                if "patient_name" in data:
                    data["환자명"] = data.pop("patient_name")
                if "patient_phone" in data:
                    data["전화번호"] = data.pop("patient_phone")
                
                # 전화번호가 있으면 환자정보에서 환자ID 조회
                if "전화번호" in data:
                    patient_result = self._find_patient_by_phone(data["전화번호"])
                    if patient_result.get("success") and patient_result.get("data"):
                        existing_patient = patient_result["data"][0]
                        # 환자명 일치 확인
                        print(f"🔍 환자명 비교: DB='{existing_patient.get('이름')}' vs 입력='{data.get('환자명')}'")
                        if existing_patient.get("이름") == data.get("환자명"):
                            print(f"✅ 환자명 일치: 기존 환자 사용")
                            patient_id = existing_patient["환자ID"]
                            data["환자ID"] = patient_id
                            data.pop("전화번호", None)  # 전화번호는 예약정보에 없으므로 제거
                        else:
                            # 환자명이 다르면 새 환자 생성
                            print(f"⚠️ 환자명 불일치: DB={existing_patient.get('이름')}, 입력={data.get('환자명')}")
                            new_patient_result = self._create_new_patient(data)
                            if new_patient_result.get("success"):
                                data["환자ID"] = new_patient_result["data"]["환자ID"]
                                data.pop("전화번호", None)
                    else:
                        # 환자를 찾을 수 없으면 새 환자 생성
                        new_patient_result = self._create_new_patient(data)
                        if new_patient_result.get("success"):
                            data["환자ID"] = new_patient_result["data"]["환자ID"]
                            data.pop("전화번호", None)
                
                result = tool._run(
                    table=tool_args.get("table", "예약정보"),
                    data=data,
                    run_manager=None
                )
            elif tool_name == "supabase_update_direct":
                from ..tools.supabase_mcp_tool import SupabaseUpdateTool
                tool = SupabaseUpdateTool()
                result = tool._run(
                    table=tool_args.get("table", "예약정보"),
                    filters=tool_args.get("filters", {}),
                    data=tool_args.get("data", {}),
                    run_manager=None
                )
            elif tool_name == "supabase_delete_direct":
                from ..tools.supabase_mcp_tool import SupabaseDeleteTool
                tool = SupabaseDeleteTool()
                result = tool._run(
                    table=tool_args.get("table", "예약정보"),
                    filters=tool_args.get("filters", {}),
                    run_manager=None
                )
            elif tool_name == "supabase_patient_lookup":
                from ..tools.supabase_mcp_tool import SupabasePatientLookupTool
                tool = SupabasePatientLookupTool()
                result = tool._run(
                    phone_number=tool_args.get("phone_number", ""),
                    patient_name=tool_args.get("patient_name", ""),
                    run_manager=None
                )
            else:
                return {
                    "success": False,
                    "error": f"지원하지 않는 툴: {tool_name}",
                    "message": "알 수 없는 툴입니다."
                }
            
            result_data = json.loads(result)
            print(f"🔍 도구 실행 결과: {result}")
            print(f"🔍 파싱된 결과: {result_data}")
            return {
                "success": result_data.get("success", False),
                "data": result_data.get("data"),
                "message": result_data.get("message", "툴 실행이 완료되었습니다."),
                "tool_name": tool_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"툴 실행 중 오류가 발생했습니다: {tool_name}"
            }
    
    def _find_patient_by_phone(self, phone_number: str) -> Dict[str, Any]:
        """전화번호로 환자정보에서 환자ID 조회"""
        try:
            from ..tools.supabase_mcp_tool import SupabaseReadTool
            tool = SupabaseReadTool()
            
            result = tool._run(
                table="환자정보",
                filters={"전화번호": phone_number},
                run_manager=None
            )
            
            result_data = json.loads(result)
            return result_data
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"환자 조회 중 오류가 발생했습니다: {phone_number}"
            }
    
    def _create_new_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """새 환자 생성"""
        try:
            from ..tools.supabase_mcp_tool import SupabaseCreateTool
            tool = SupabaseCreateTool()
            
            # 환자정보 테이블에 필요한 데이터만 추출
            patient_info = {
                "이름": patient_data.get("환자명", ""),
                "전화번호": patient_data.get("전화번호", ""),
                "성별": patient_data.get("성별", ""),
                "생년월일": patient_data.get("생년월일", ""),
                "이메일": patient_data.get("이메일", ""),
                "주소": patient_data.get("주소", "")
            }
            
            result = tool._run(
                table="환자정보",
                data=patient_info,
                run_manager=None
            )
            
            result_data = json.loads(result)
            if result_data.get("success"):
                # 생성된 환자의 ID를 반환
                created_patient = result_data.get("data", [{}])[0]
                result_data["data"] = {"환자ID": created_patient.get("환자ID")}
            return result_data
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"새 환자 생성 중 오류가 발생했습니다"
            }
    
    def _get_doctor_schedule(self, doctor_name: str) -> Dict[str, Any]:
        """의료진의 예약 가능 일정 조회"""
        try:
            from ..tools.supabase_mcp_tool import SupabaseDoctorLookupTool, SupabaseScheduleLookupTool
            
            # 의료진명 정리 (예: "양재혁/D003" -> "양재혁")
            clean_doctor_name = doctor_name.split('/')[0].strip()
            print(f"🔍 정리된 의료진명: '{clean_doctor_name}' (원본: '{doctor_name}')")
            
            # 1. 의료진명으로 DocID 조회
            doctor_tool = SupabaseDoctorLookupTool()
            doctor_result = doctor_tool._run(doctor_name=clean_doctor_name, run_manager=None)
            doctor_data = json.loads(doctor_result)
            
            if not doctor_data.get("success") or not doctor_data.get("data"):
                return {
                    "success": False,
                    "error": f"의료진 '{doctor_name}'을 찾을 수 없습니다.",
                    "message": "의료진 조회 실패"
                }
            
            doc_id = doctor_data["data"][0]["DocID"]
            print(f"🔍 의료진 '{doctor_name}'의 DocID: {doc_id}")
            
            # 2. DocID로 가용일정 조회
            schedule_tool = SupabaseScheduleLookupTool()
            schedule_result = schedule_tool._run(doc_id=doc_id, limit=5, run_manager=None)
            schedule_data = json.loads(schedule_result)
            
            if schedule_data.get("success"):
                # 일정을 사용자 친화적인 형태로 변환
                formatted_schedules = []
                for schedule in schedule_data.get("data", []):
                    formatted_schedule = {
                        "일정ID": schedule.get("일정ID"),
                        "날짜": f"{schedule.get('진료년')}-{schedule.get('진료월'):02d}-{schedule.get('진료일'):02d}",
                        "시간": f"{schedule.get('진료시'):02d}:{schedule.get('진료분'):02d}",
                        "의료진": doctor_name,
                        "DocID": doc_id
                    }
                    formatted_schedules.append(formatted_schedule)
                
                return {
                    "success": True,
                    "data": formatted_schedules,
                    "message": f"의료진 '{doctor_name}'의 예약 가능 일정 {len(formatted_schedules)}건 조회 완료"
                }
            else:
                return {
                    "success": False,
                    "error": schedule_data.get("error", "일정 조회 실패"),
                    "message": "가용일정 조회 실패"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"의료진 일정 조회 중 오류가 발생했습니다"
            }
    
    def _get_doctor_schedule_by_doc_id(self, doc_id: int) -> Dict[str, Any]:
        """DocID로 예약 가능한 일정 조회"""
        try:
            print(f"🔍 DocID로 일정 조회: {doc_id}")
            
            from ..tools.supabase_mcp_tool import SupabaseScheduleLookupTool
            
            # DocID로 가용일정 조회
            schedule_tool = SupabaseScheduleLookupTool()
            schedule_result = schedule_tool._run(doc_id=doc_id, limit=5, run_manager=None)
            schedule_data = json.loads(schedule_result)
            
            if schedule_data.get("success"):
                # 일정을 사용자 친화적인 형태로 변환
                formatted_schedules = []
                for schedule in schedule_data.get("data", []):
                    formatted_schedule = {
                        "일정ID": schedule.get("일정ID"),
                        "날짜": f"{schedule.get('진료년')}-{schedule.get('진료월'):02d}-{schedule.get('진료일'):02d}",
                        "시간": f"{schedule.get('진료시'):02d}:{schedule.get('진료분'):02d}",
                        "DocID": doc_id
                    }
                    formatted_schedules.append(formatted_schedule)
                
                return {
                    "success": True,
                    "data": formatted_schedules,
                    "message": f"DocID {doc_id}의 예약 가능 일정 {len(formatted_schedules)}건 조회 완료"
                }
            else:
                return {
                    "success": False,
                    "error": schedule_data.get("error", "일정 조회 실패"),
                    "message": "가용일정 조회 실패"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"DocID 일정 조회 중 오류가 발생했습니다"
            }
    
    def _parse_schedule_preference(self, user_input: str) -> Dict[str, Any]:
        """자연어로 된 예약 희망 일정 파싱"""
        try:
            if not self.llm_client:
                return {"success": False, "parsed_preference": None}
            
            prompt = f"""
사용자의 예약 희망 일정을 분석해주세요.

사용자 입력: "{user_input}"

다음 형식으로 응답해주세요:
{{
    "success": true,
    "parsed_preference": {{
        "urgency": "high|medium|low",
        "preferred_date": "YYYY-MM-DD 또는 null",
        "preferred_time": "HH:MM 또는 null", 
        "time_period": "morning|afternoon|evening|any",
        "days_from_now": 숫자 또는 null,
        "natural_language": "원본 자연어"
    }}
}}

분석 기준:
- "최대한 빨리", "급해요" → urgency: "high", days_from_now: 0-1
- "내일", "다음 주" → days_from_now 계산
- "오전", "오후", "저녁" → time_period 설정
- "월요일", "화요일" 등 → 요일 기반 날짜 계산
- 구체적인 날짜/시간이 있으면 preferred_date, preferred_time 설정
"""
            
            response = self.llm_client.invoke([
                {"role": "system", "content": "당신은 예약 일정을 분석하는 전문가입니다. 사용자의 자연어 입력을 구조화된 데이터로 변환해주세요."},
                {"role": "user", "content": prompt}
            ])
            
            content = response.content
            result = json.loads(content)
            
            return {
                "success": True,
                "parsed_preference": result.get("parsed_preference", {})
            }
            
        except Exception as e:
            print(f"⚠️ 일정 선호도 파싱 오류: {e}")
            return {
                "success": False,
                "parsed_preference": None
            }
    
    def _match_schedule_with_preference(self, available_schedules: List[Dict], preference: Dict) -> List[Dict]:
        """가용일정과 사용자 선호도 매칭"""
        try:
            if not preference or not available_schedules:
                return available_schedules
            
            matched_schedules = []
            urgency = preference.get("urgency", "medium")
            time_period = preference.get("time_period", "any")
            days_from_now = preference.get("days_from_now")
            
            # 긴급도에 따른 우선순위 정렬
            if urgency == "high":
                # 최근 일정 우선
                sorted_schedules = sorted(available_schedules, key=lambda x: x.get("날짜", ""))
            else:
                # 기본 순서 유지
                sorted_schedules = available_schedules
            
            # 시간대 필터링
            for schedule in sorted_schedules:
                schedule_time = schedule.get("시간", "")
                hour = int(schedule_time.split(":")[0]) if ":" in schedule_time else 0
                
                # 시간대 매칭
                if time_period == "morning" and 6 <= hour < 12:
                    matched_schedules.append(schedule)
                elif time_period == "afternoon" and 12 <= hour < 18:
                    matched_schedules.append(schedule)
                elif time_period == "evening" and 18 <= hour < 22:
                    matched_schedules.append(schedule)
                elif time_period == "any":
                    matched_schedules.append(schedule)
            
            # 매칭된 일정이 없으면 모든 일정 반환
            return matched_schedules if matched_schedules else available_schedules
            
        except Exception as e:
            print(f"⚠️ 일정 매칭 오류: {e}")
            return available_schedules
    
    def _analyze_reservation_intent(self, user_input: str) -> Dict[str, Any]:
        """예약 의도 분석"""
        try:
            # 예약 확인 관련 키워드
            check_keywords = ["예약확인", "예약 확인", "예약조회", "예약 조회", "내예약", "내 예약", "예약내역", "예약 내역", "예약상태", "예약 상태", "예약정보", "예약 정보"]
            rebook_keywords = ["재예약", "재 예약", "다시 예약", "또 예약", "같은 의사", "같은 선생님", "이전 의사", "이전 선생님", "전에 봤던", "전에 진료받던"]
            cancel_keywords = ["예약취소", "예약 취소", "예약삭제", "예약 삭제", "취소하고", "취소하고싶", "취소하고싶어"]
            modify_keywords = ["예약변경", "예약 변경", "예약수정", "예약 수정", "시간바꾸", "시간 바꾸", "일정바꾸", "일정 바꾸"]
            
            user_input_lower = user_input.lower()
            
            if any(keyword in user_input_lower for keyword in check_keywords):
                return {"action": "check", "confidence": 0.9}
            elif any(keyword in user_input_lower for keyword in rebook_keywords):
                return {"action": "rebook", "confidence": 0.9}
            elif any(keyword in user_input_lower for keyword in cancel_keywords):
                return {"action": "cancel", "confidence": 0.9}
            elif any(keyword in user_input_lower for keyword in modify_keywords):
                return {"action": "modify", "confidence": 0.9}
            else:
                return {"action": "create", "confidence": 0.7}
                
        except Exception as e:
            print(f"❌ 의도 분석 오류: {e}")
            return {"action": "create", "confidence": 0.5}
    
    def _handle_reservation_check(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """예약 확인 처리"""
        try:
            print("🔍 예약 확인 처리 시작")
            
            # 사용자 정보 추출 (이름, 전화번호)
            extracted_info = self._extract_patient_info_for_check(user_input, existing_info)
            
            if not extracted_info.get("환자명") or not extracted_info.get("전화번호"):
                return {
                    "success": False,
                    "status": "need_patient_info",
                    "message": "예약 확인을 위해 환자명과 전화번호를 알려주세요.",
                    "missing_fields": ["환자명", "전화번호"]
                }
            
            # Supabase에서 예약 정보 조회
            if self.llm_with_tools and self.tools:
                # Tool Calling을 사용한 조회
                return self._check_reservation_with_llm_tools(user_input, extracted_info)
            else:
                # 기본 조회 방식
                return self._check_reservation_basic(extracted_info)
                
        except Exception as e:
            print(f"❌ 예약 확인 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "예약 확인 중 오류가 발생했습니다."
            }
    
    def _extract_patient_info_for_check(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """예약 확인을 위한 환자 정보 추출"""
        try:
            import re
            
            extracted = {}
            
            # 기존 정보가 있으면 우선 사용
            if existing_info:
                extracted.update(existing_info)
            
            # 전화번호 패턴 매칭
            phone_pattern = r'01[0-9]-?[0-9]{3,4}-?[0-9]{4}'
            phone_match = re.search(phone_pattern, user_input)
            if phone_match:
                extracted["전화번호"] = phone_match.group()
            
            # 이름 패턴 매칭 (한글 2-4자)
            name_pattern = r'[가-힣]{2,4}'
            name_matches = re.findall(name_pattern, user_input)
            if name_matches:
                # 가장 긴 이름을 선택
                extracted["환자명"] = max(name_matches, key=len)
            
            return extracted
            
        except Exception as e:
            print(f"❌ 환자 정보 추출 오류: {e}")
            return {}
    
    def _check_reservation_with_llm_tools(self, user_input: str, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """올바른 예약 확인 로직: 환자정보 → 예약정보"""
        try:
            print("🔍 올바른 예약 확인 로직 시작")
            
            # 1단계: 환자정보에서 환자번호 조회
            patient_result = self._lookup_patient_info(patient_info)
            
            if not patient_result.get("success"):
                return {
                    "success": False,
                    "status": "patient_not_found",
                    "message": "등록된 환자 정보를 찾을 수 없습니다.",
                    "patient_info": patient_info
                }
            
            patient_data = patient_result["data"][0]
            patient_id = patient_data["환자ID"]
            patient_name = patient_data["이름"]
            
            print(f"🔍 환자 조회 성공: 환자번호 {patient_id}, 이름 {patient_name}")
            
            # 2단계: 예약정보에서 환자번호로 예약 조회
            reservation_result = self._lookup_reservations_by_patient_id(patient_id)
            
            if not reservation_result.get("success"):
                return {
                    "success": False,
                    "status": "reservation_lookup_failed",
                    "message": "예약 정보 조회 중 오류가 발생했습니다.",
                    "patient_info": patient_info
                }
            
            reservations = reservation_result["data"]
            
            if reservations:
                return {
                    "success": True,
                    "status": "found_reservations",
                    "message": f"{patient_name}님의 예약 정보를 찾았습니다. 총 {len(reservations)}건의 예약이 있습니다.",
                    "reservations": reservations,
                    "patient_info": {
                        "환자ID": patient_id,
                        "환자명": patient_name,
                        "전화번호": patient_data.get("전화번호")
                    }
                }
            else:
                return {
                    "success": True,
                    "status": "no_reservations",
                    "message": f"{patient_name}님의 등록된 예약이 없습니다.",
                    "reservations": [],
                    "patient_info": {
                        "환자ID": patient_id,
                        "환자명": patient_name,
                        "전화번호": patient_data.get("전화번호")
                    }
                }
                
        except Exception as e:
            print(f"❌ 예약 확인 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "예약 확인 중 오류가 발생했습니다."
            }
    
    def _lookup_patient_info(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """환자정보 테이블에서 환자 조회"""
        try:
            # supabase_patient_lookup 도구 찾기
            patient_lookup_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == 'supabase_patient_lookup':
                    patient_lookup_tool = tool
                    break
            
            if not patient_lookup_tool:
                return {"success": False, "error": "환자 조회 도구를 찾을 수 없습니다."}
            
            # 환자 조회 실행
            result = patient_lookup_tool.invoke({
                "phone_number": patient_info.get("전화번호"),
                "patient_name": patient_info.get("환자명")
            })
            
            # 결과 파싱
            if isinstance(result, str):
                import json
                result = json.loads(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 환자 조회 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def _lookup_reservations_by_patient_id(self, patient_id: int) -> Dict[str, Any]:
        """예약정보 테이블에서 환자번호로 예약 조회"""
        try:
            # supabase_read_direct 도구 찾기
            read_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == 'supabase_read_direct':
                    read_tool = tool
                    break
            
            if not read_tool:
                return {"success": False, "error": "예약 조회 도구를 찾을 수 없습니다."}
            
            print(f"🔍 예약 조회 실행: 환자번호 {patient_id}")
            
            # 예약 조회 실행
            result = read_tool.invoke({
                "table": "예약정보",
                "filters": {"환자ID": patient_id}
            })
            
            print(f"🔍 예약 조회 결과: {result}")
            
            # 결과 파싱
            if isinstance(result, str):
                import json
                result = json.loads(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 예약 조회 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_tool_for_check(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """예약 확인을 위한 도구 실행"""
        try:
            # 도구 찾기
            target_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    target_tool = tool
                    break
            
            if not target_tool:
                return {"error": f"도구를 찾을 수 없습니다: {tool_name}"}
            
            # 도구 실행
            result = target_tool.invoke(tool_args)
            return {
                "tool_name": tool_name,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            return {
                "tool_name": tool_name,
                "error": str(e),
                "success": False
            }
    
    def _analyze_check_results(self, tool_results: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """예약 확인 결과 분석"""
        try:
            reservations = []
            
            for tool_result in tool_results:
                if tool_result.get("success") and tool_result.get("result"):
                    result_data = tool_result["result"]
                    if isinstance(result_data, dict) and result_data.get("data"):
                        reservations.extend(result_data["data"])
            
            if reservations:
                return {
                    "success": True,
                    "status": "found_reservations",
                    "message": f"예약 정보를 찾았습니다. 총 {len(reservations)}건의 예약이 있습니다.",
                    "reservations": reservations,
                    "patient_info": patient_info
                }
            else:
                return {
                    "success": True,
                    "status": "no_reservations",
                    "message": "등록된 예약이 없습니다.",
                    "reservations": [],
                    "patient_info": patient_info
                }
                
        except Exception as e:
            print(f"❌ 결과 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "예약 확인 결과 분석 중 오류가 발생했습니다."
            }
    
    def _check_reservation_with_tools(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Tool Calling을 사용한 예약 확인"""
        try:
            # Supabase 조회 도구 사용
            from ..tools.supabase_mcp_tool import get_supabase_tools_for_binding
            tools = get_supabase_tools_for_binding()
            
            # supabase_read_direct 도구 찾기
            read_tool = None
            for tool in tools:
                if hasattr(tool, 'name') and 'read' in tool.name.lower():
                    read_tool = tool
                    break
            
            if not read_tool:
                return self._check_reservation_basic(patient_info)
            
            # 예약 정보 조회
            filters = {
                "환자명": patient_info.get("환자명"),
                "전화번호": patient_info.get("전화번호")
            }
            
            result = read_tool.invoke({
                "table": "예약정보",
                "filters": filters
            })
            
            if result and result.get("data"):
                reservations = result["data"]
                return {
                    "success": True,
                    "status": "found_reservations",
                    "message": f"예약 정보를 찾았습니다. 총 {len(reservations)}건의 예약이 있습니다.",
                    "reservations": reservations,
                    "patient_info": patient_info
                }
            else:
                return {
                    "success": True,
                    "status": "no_reservations",
                    "message": "등록된 예약이 없습니다.",
                    "reservations": [],
                    "patient_info": patient_info
                }
                
        except Exception as e:
            print(f"❌ Tool Calling 예약 확인 오류: {e}")
            return self._check_reservation_basic(patient_info)
    
    def _check_reservation_basic(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """기본 예약 확인 (폴백)"""
        try:
            # 기본 응답 (실제 구현에서는 Supabase 연결 필요)
            return {
                "success": True,
                "status": "basic_check",
                "message": f"환자 {patient_info.get('환자명')}님의 예약 정보를 확인했습니다. (기본 모드)",
                "reservations": [],
                "patient_info": patient_info
            }
            
        except Exception as e:
            print(f"❌ 기본 예약 확인 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "예약 확인 중 오류가 발생했습니다."
            }
    
    def _handle_reservation_cancel(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """예약 취소 처리"""
        try:
            print("🔍 예약 취소 처리 시작")
            return {
                "success": True,
                "status": "cancel_request",
                "message": "예약 취소 기능은 준비 중입니다. 전화 상담(1599-0015)을 이용해주세요.",
                "action": "cancel"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "예약 취소 처리 중 오류가 발생했습니다."
            }
    
    def _handle_reservation_modify(self, user_input: str, existing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """예약 변경 처리"""
        try:
            print("🔍 예약 변경 처리 시작")
            return {
                "success": True,
                "status": "modify_request",
                "message": "예약 변경 기능은 준비 중입니다. 전화 상담(1599-0015)을 이용해주세요.",
                "action": "modify"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "예약 변경 처리 중 오류가 발생했습니다."
            }
