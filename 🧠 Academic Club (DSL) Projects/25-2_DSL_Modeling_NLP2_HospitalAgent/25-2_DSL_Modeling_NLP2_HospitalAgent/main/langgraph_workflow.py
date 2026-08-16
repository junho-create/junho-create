"""
새로운 LangGraph 워크플로우
3개 에이전트 시스템: 관리자 → 예약/RAG/검색
LangSmith 연동 포함
"""
import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 환경 변수 로딩 (LangGraph Studio에서도 .env 파일을 로드하도록)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 환경 변수 로딩 완료 (.env 파일)")
except ImportError:
    print("⚠️ dotenv 패키지가 없습니다. 환경 변수를 수동으로 설정해주세요.")
except Exception as e:
    print(f"⚠️ 환경 변수 로딩 중 오류: {e}")

from main.langgraph_state import HospitalReservationState, create_initial_state

# LangSmith 연동 설정
if os.getenv("LANGSMITH_TRACING") == "true":
    from langsmith import Client
    from langchain_core.tracers import LangChainTracer
    
    # LangSmith 클라이언트 초기화
    langsmith_client = Client()
    tracer = LangChainTracer(project_name=os.getenv("LANGSMITH_PROJECT", "hospital_agent_v3"))
    print("✅ LangSmith 연동 활성화")
else:
    tracer = None
    print("ℹ️ LangSmith 연동 비활성화 (LANGSMITH_TRACING=false 또는 설정 없음)")

# 에이전트 import
from main.agents.agent1_manager import analyze_and_route_user_request
from main.agents.agent2_reservation import Agent2Reservation
from main.agents.agent3_rag import recommend_doctors_for_symptoms

def format_reservation_response(result: Dict[str, Any]) -> str:
    """예약 응답 포맷팅"""
    if not result.get("success"):
        # success=False이지만 정상적인 상태들 처리
        status = result.get("status", "unknown")
        if status in ["need_symptoms", "missing_info", "need_more_info", "rebooking_ready", "need_patient_info", "need_schedule_selection", "need_doctor_selection", "reservation_completed", "reservation_failed"]:
            return result.get('message', '추가 정보가 필요합니다.')
        else:
            return f"❌ 예약 처리 실패\n{result.get('message', '알 수 없는 오류가 발생했습니다.')}"
    
    collected_info = result.get("collected_info", {})
    status = result.get("status", "unknown")
    
    if status in ["need_more_info", "missing_info"]:
        # missing_info 상태도 처리
        missing_fields = result.get("missing_fields", [])
        if missing_fields:
            response = result.get('message', f"예약 정보 수집 중... {', '.join(missing_fields)}을(를) 알려주세요.")
        else:
            response = result.get('message', "예약 정보 수집 중... 추가 정보가 필요합니다.")
        
    elif status == "found_reservations":
        # 예약 확인 후 재예약 플로우
        reservations = result.get("reservations", [])
        patient_info = result.get("patient_info", {})
        
        response = f"📋 **{patient_info.get('환자명', '환자')}님의 예약 정보**\n\n"
        response += f"📞 **연락처:** {patient_info.get('전화번호', 'N/A')}\n"
        response += f"🏥 **총 예약 건수:** {len(reservations)}건\n\n"
        
        # 최근 예약 정보 표시 (최대 3건)
        recent_reservations = reservations[:3]
        for i, reservation in enumerate(recent_reservations, 1):
            # 시간 포맷팅 (HH:MM 형식)
            reservation_time = reservation.get('예약시간', 'N/A')
            if reservation_time and reservation_time != 'N/A':
                try:
                    from datetime import datetime
                    time_obj = datetime.strptime(reservation_time, "%H:%M:%S")
                    reservation_time = time_obj.strftime("%H:%M")
                except:
                    pass
            
            response += f"**{i}. {reservation.get('진료일자', 'N/A')} {reservation_time}**\n"
            response += f"   • 의료진: {reservation.get('의료진명', 'DocID ' + str(reservation.get('DocID', 'Unknown')))}\n"
            response += f"   • 진료과: {reservation.get('진료과', 'N/A')}\n"
            response += f"   • 상태: {reservation.get('예약상태', 'N/A')}\n\n"
        
        if len(reservations) > 3:
            response += f"... 외 {len(reservations) - 3}건의 예약이 더 있습니다.\n\n"
        
        # 재예약 안내 메시지
        if recent_reservations:
            latest_doc_id = recent_reservations[0].get('DocID', '')
            latest_doctor = recent_reservations[0].get('의료진명', '')
            
            # 의료진명이 없으면 DocID로 조회
            if not latest_doctor and latest_doc_id:
                try:
                    from main.tools.supabase_mcp_tool import SupabaseReadTool
                    doctor_tool = SupabaseReadTool()
                    doctor_result = doctor_tool._run(
                        table="의사",
                        filters={"DocID": latest_doc_id}
                    )
                    import json
                    doctor_data = json.loads(doctor_result)
                    if doctor_data.get("success") and doctor_data.get("data"):
                        latest_doctor = doctor_data["data"][0].get("의료진명", "")
                except Exception as e:
                    print(f"⚠️ 의료진명 조회 오류: {e}")
                    latest_doctor = f"DocID {latest_doc_id}"
            
            if not latest_doctor:
                latest_doctor = f"DocID {latest_doc_id}"
            
            response += f"🔄 **재예약 안내**\n"
            response += f"이전에 {latest_doctor} 의사님께 진료받으셨습니다.\n"
            response += f"같은 의사님으로 재예약하시려면 **'같은 의사로 재예약하고 싶어요'**라고 말씀해주세요.\n\n"
            response += f"📞 **추가 문의:** 1599.0015"
            
            # 재예약을 위한 컨텍스트 정보 저장
            rebooking_context = {
                "previous_doctor": {
                    "name": latest_doctor,
                    "DocID": latest_doc_id
                },
                "patient_info": patient_info,
                "is_rebooking": True
            }
            result["rebooking_context"] = rebooking_context
            result["collected_info"] = {"rebooking_context": rebooking_context}
        else:
            response += f"📞 **추가 문의:** 1599.0015"
        
    elif status == "no_reservations":
        # 예약이 없는 경우
        patient_info = result.get("patient_info", {})
        response = f"📋 **{patient_info.get('환자명', '환자')}님의 예약 정보**\n\n"
        response += f"📞 **연락처:** {patient_info.get('전화번호', 'N/A')}\n"
        response += f"📝 **예약 내역:** 등록된 예약이 없습니다.\n\n"
        response += f"🆕 **새 예약을 원하시면** '예약하고 싶어요'라고 말씀해주세요.\n"
        response += f"📞 **추가 문의:** 1599.0015"
        
    elif status == "info_complete":
        response = f"✅ **예약 정보 수집 완료**\n\n"
        response += f"👤 **환자 정보:**\n"
        response += f"• 이름: {collected_info.get('환자명', 'N/A')}\n"
        response += f"• 연락처: {collected_info.get('전화번호', 'N/A')}\n\n"
        
        if collected_info.get("symptoms"):
            response += f"🩺 **증상:** {', '.join(collected_info['symptoms'])}\n\n"
        
        if collected_info.get("available_schedules"):
            schedules = collected_info["available_schedules"]
            response += f"📅 **예약 가능 일정** ({len(schedules)}건)\n\n"
            for i, schedule in enumerate(schedules[:3], 1):
                response += f"{i}. **{schedule.get('날짜', 'N/A')} {schedule.get('시간', 'N/A')}**\n"
                response += f"   • 의료진: {schedule.get('의료진', 'N/A')}\n\n"
            
            response += "💡 **예약 희망 일정을 자연어로 말씀해주세요!**\n"
            response += "예: '최대한 빨리', '내일 오후', '다음 주 월요일 오전'"
        else:
            response += "예약을 진행하시겠나요? 사용 가능한 시간을 확인해드릴까요?"
        
    else:
        response = result.get('message', '예약 처리 중입니다.')
    
    return response

def format_doctor_recommendation(result: Dict[str, Any]) -> str:
    """의료진 추천 응답 포맷팅"""
    if not result.get("success"):
        return f"❌ 의료진 추천 실패\n{result.get('message', '알 수 없는 오류가 발생했습니다.')}"
    
    recommended_doctors = result.get("recommended_doctors", [])
    department = result.get("department", "")
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "")
    
    response = f"👨‍⚕️ **의료진 추천 결과**\n\n"
    response += f"📋 **추천 진료과**: {department}\n"
    response += f"🎯 **추천 근거**: {reasoning}\n"
    response += f"📊 **신뢰도**: {confidence:.1%}\n\n"
    
    if recommended_doctors:
        response += "**추천 의료진:**\n"
        for i, doctor in enumerate(recommended_doctors[:3], 1):
            response += f"{i}. **{doctor.get('name', 'Unknown')} 의사**\n"
            response += f"   • 진료과: {doctor.get('department', 'Unknown')}\n"
            response += f"   • 추천 근거: {doctor.get('reasoning', 'N/A')}\n\n"
    
    response += "📅 **다음 단계**: 예약을 원하시면 의료진을 선택하고 예약 정보를 알려주세요!"
    
    return response

def create_hospital_reservation_workflow():
    """병원 예약 시스템 워크플로우 생성 (새로운 에이전트 구조)"""
    
    print("🔧 LangGraph Studio 워크플로우 생성 시작")
    print(f"🔑 OPENAI_API_KEY 설정 여부: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    print(f"🔑 SUPABASE_URL 설정 여부: {'✅' if os.getenv('SUPABASE_URL') else '❌'}")
    print(f"🔑 SUPABASE_ANON_KEY 설정 여부: {'✅' if os.getenv('SUPABASE_ANON_KEY') else '❌'}")
    
    # StateGraph 생성
    workflow = StateGraph(HospitalReservationState)
    
    # 노드 추가
    workflow.add_node("manager_agent", manager_agent_node)
    workflow.add_node("reservation_agent", reservation_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("response_generation", response_generation_node)
    
    # 엔트리 포인트 설정
    workflow.set_entry_point("manager_agent")
    
    # 라우팅 함수
    def route_after_manager(state: HospitalReservationState) -> str:
        """관리자 에이전트 후 라우팅"""
        routing_info = state.get('routing_info')
        
        # routing_info가 None이거나 빈 딕셔너리인 경우 처리
        if not routing_info:
            print(f"🎯 관리자 라우팅: None (routing_info 없음)")
            return "response_generation"
        
        # routing_info에서 target_agent 찾기 (중첩된 구조 고려)
        target = None
        if 'routing' in routing_info and routing_info['routing']:
            target = routing_info['routing'].get('target_agent')
        elif 'target_agent' in routing_info:
            target = routing_info['target_agent']
        elif 'target' in routing_info:
            target = routing_info['target']
        
        print(f"🎯 관리자 라우팅: {target}")
        
        if target == "agent2_reservation":
            return "reservation_agent"
        elif target == "agent3_rag":
            return "rag_agent"
        elif target == "agent1_direct":
            return "response_generation"
        else:
            return "response_generation"
    
    def route_after_reservation(state: HospitalReservationState) -> str:
        """예약 에이전트 후 라우팅"""
        status = state.get('reservation_status', 'completed')
        
        if status == "need_more_info":
            # 추가 정보가 필요하면 다시 관리자로
            return "manager_agent"
        else:
            return "response_generation"
    
    # 워크플로우 연결
    workflow.add_conditional_edges(
        "manager_agent",
        route_after_manager,
        {
            "reservation_agent": "reservation_agent",
            "rag_agent": "rag_agent", 
            "response_generation": "response_generation"
        }
    )
    
    workflow.add_conditional_edges(
        "reservation_agent",
        route_after_reservation,
        {
            "manager_agent": "manager_agent",
            "response_generation": "response_generation"
        }
    )
    
    workflow.add_edge("rag_agent", "response_generation")
    workflow.add_edge("response_generation", END)
    
    # 워크플로우 컴파일
    compiled_workflow = workflow.compile()
    
    print("✅ LangGraph Studio 워크플로우 생성 완료")
    return compiled_workflow

def test_langgraph_studio_connection():
    """LangGraph Studio에서 연결 테스트"""
    try:
        print("🧪 LangGraph Studio 연결 테스트 시작")
        
        # 환경 변수 확인
        required_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_ANON_KEY']
        for var in required_vars:
            if not os.getenv(var):
                print(f"❌ {var} 환경 변수가 설정되지 않았습니다")
                return False
            else:
                print(f"✅ {var} 환경 변수 설정됨")
        
        # 워크플로우 생성 테스트
        workflow = create_hospital_reservation_workflow()
        print("✅ 워크플로우 생성 성공")
        
        # 간단한 테스트 실행
        test_state = create_initial_state("테스트 메시지", "test_session")
        print("✅ 초기 상태 생성 성공")
        
        print("🎉 LangGraph Studio 연결 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ LangGraph Studio 연결 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def manager_agent_node(state: HospitalReservationState) -> HospitalReservationState:
    """관리자 에이전트 노드"""
    try:
        print("🎯 관리자 에이전트 시작")
        
        # LangGraph Studio UI에서는 messages에서 사용자 입력을 추출
        if 'messages' in state and state['messages']:
            # 마지막 메시지가 HumanMessage인지 확인
            last_message = state['messages'][-1]
            if hasattr(last_message, 'content'):
                user_input = last_message.content
            else:
                user_input = str(last_message)
        else:
            # 기존 방식 (chat_interface.py에서 사용)
            user_input = state.get('user_query', '')
        
        print(f"🔍 사용자 입력: {user_input}")
        
        existing_context = state.get('context', {})
        print(f"🔍 기존 컨텍스트: {existing_context}")
        
        # 이전 대화 의도 정보를 컨텍스트에서 확인
        if existing_context and existing_context.get('previous_intent'):
            print(f"🔍 이전 대화 의도: {existing_context['previous_intent']}")
        
        # 현재 상태에서도 이전 대화 의도 정보 확인 (기존 로직 유지)
        if 'manager_result' in state and state['manager_result']:
            previous_result = state['manager_result']
            if 'extracted_info' in previous_result and 'action' in previous_result['extracted_info']:
                existing_context['previous_intent'] = previous_result['extracted_info']['action']
                print(f"🔍 현재 상태에서 이전 대화 의도: {existing_context['previous_intent']}")
        
        # 관리자 에이전트로 라우팅
        print("🔍 analyze_and_route_user_request 호출 전")
        result = analyze_and_route_user_request(user_input, existing_context)
        print(f"🔍 analyze_and_route_user_request 결과: {result}")
        
        # 결과를 상태에 저장
        state['manager_result'] = result
        routing_info = result.get('routing_info', {})
        state['routing_info'] = routing_info if routing_info else {}
        state['manager_complete'] = True
        
        # 라우팅 정보 출력
        target = 'unknown'
        if routing_info:
            if 'target' in routing_info:
                target = routing_info['target']
            elif 'target_agent' in routing_info:
                target = routing_info['target_agent']
        
        print(f"✅ 관리자 에이전트 완료: {target}")
        
    except Exception as e:
        print(f"❌ 관리자 에이전트 오류: {e}")
        state['error'] = str(e)
        state['error_step'] = 'manager_agent'
        state['manager_complete'] = False
    
    return state

def reservation_agent_node(state: HospitalReservationState) -> HospitalReservationState:
    """예약 에이전트 노드"""
    try:
        print("📅 예약 에이전트 시작")
        
        # LangGraph Studio UI에서는 messages에서 사용자 입력을 추출
        if 'messages' in state and state['messages']:
            # 마지막 메시지가 HumanMessage인지 확인
            last_message = state['messages'][-1]
            if hasattr(last_message, 'content'):
                user_input = last_message.content
            else:
                user_input = str(last_message)
        else:
            # 기존 방식 (chat_interface.py에서 사용)
            user_input = state.get('user_query', '')
        
        existing_info = state.get('reservation_info', {})
        
        # 기존 컨텍스트 정보도 활용
        context_info = state.get('context', {})
        if context_info:
            existing_info.update(context_info)
        
        # 관리자 에이전트의 결과가 있으면 우선 사용
        manager_result = state.get('manager_result', {})
        if manager_result:
            print(f"🔍 관리자 에이전트 결과 발견: {manager_result.get('status', 'N/A')}")
            existing_info.update(manager_result)
        
        print(f"🔍 기존 정보: {existing_info}")
        
        # 예약 에이전트 인스턴스 생성 및 처리
        agent2 = Agent2Reservation()
        result = agent2.process_reservation_request(user_input, existing_info)
        
        # 결과를 상태에 저장
        state['reservation_result'] = result
        state['reservation_info'] = result.get('collected_info', {})
        state['reservation_status'] = result.get('success', False)
        state['reservation_complete'] = True
        
        # 컨텍스트 업데이트 (다음 단계를 위해)
        if result.get('collected_info'):
            state['context'] = result.get('collected_info', {})
        
        print(f"✅ 예약 에이전트 완료: {result.get('success', False)}")
        
    except Exception as e:
        print(f"❌ 예약 에이전트 오류: {e}")
        state['error'] = str(e)
        state['error_step'] = 'reservation_agent'
        state['reservation_complete'] = False
    
    return state

def rag_agent_node(state: HospitalReservationState) -> HospitalReservationState:
    """RAG 에이전트 노드"""
    try:
        print("🧠 RAG 에이전트 시작")
        
        # LangGraph Studio UI에서는 messages에서 사용자 입력을 추출
        if 'messages' in state and state['messages']:
            # 마지막 메시지가 HumanMessage인지 확인
            last_message = state['messages'][-1]
            if hasattr(last_message, 'content'):
                user_input = last_message.content
            else:
                user_input = str(last_message)
        else:
            # 기존 방식 (chat_interface.py에서 사용)
            user_input = state.get('user_query', '')
        
        # 증상 정보 추출
        manager_result = state.get('manager_result', {})
        extracted_info = manager_result.get('extracted_info', {})
        symptoms = extracted_info.get('symptoms', [])
        
        if not symptoms:
            symptoms = [user_input]
        
        # RAG 에이전트로 의료진 추천
        result = recommend_doctors_for_symptoms(symptoms, user_input)
        
        # 결과를 상태에 저장
        state['rag_result'] = result
        state['rag_complete'] = True
        
        # RAG 결과를 컨텍스트에도 저장 (다음 단계에서 활용)
        if result.get('success') and result.get('recommended_doctors'):
            if 'context' not in state or state['context'] is None:
                state['context'] = {}
            
            # RAG 결과를 컨텍스트에 저장
            state['context']['recommended_doctors'] = result['recommended_doctors']
            state['context']['recommended_department'] = result.get('department')
            state['context']['rag_confidence'] = result.get('confidence', 0.0)
            state['context']['symptoms'] = symptoms  # 증상 정보도 저장
            
            print(f"🔍 RAG 결과를 컨텍스트에 저장: {len(result['recommended_doctors'])}명 의료진")
        
        print(f"✅ RAG 에이전트 완료: {result.get('success', False)}")
        
    except Exception as e:
        print(f"❌ RAG 에이전트 오류: {e}")
        state['error'] = str(e)
        state['error_step'] = 'rag_agent'
        state['rag_complete'] = False
    
    return state

def response_generation_node(state: HospitalReservationState) -> HospitalReservationState:
    """응답 생성 노드"""
    try:
        print("💬 응답 생성 시작")
        
        # 오류가 있으면 오류 응답
        if state.get('error'):
            response = f"❌ {state.get('error_step', 'unknown')} 단계에서 오류가 발생했습니다: {state['error']}"
        else:
            # 각 에이전트의 결과에 따라 응답 생성
            routing_info = state.get('routing_info', {})
            target = ''
            
            if routing_info:
                if 'target' in routing_info:
                    target = routing_info['target']
                elif 'target_agent' in routing_info:
                    target = routing_info['target_agent']
            
            if target == "agent2_reservation":
                # 예약 에이전트 응답
                reservation_result = state.get('reservation_result', {})
                response = format_reservation_response(reservation_result)
                
            elif target == "agent3_rag":
                # RAG 에이전트 응답
                rag_result = state.get('rag_result', {})
                response = format_doctor_recommendation(rag_result)
                
            elif target == "tavily_search":
                # 검색 툴 응답
                manager_result = state.get('manager_result', {})
                response = manager_result.get('message', '병원 정보를 찾을 수 없습니다.')
                
            elif target == "agent1_direct":
                # Agent1 직접 응답
                manager_result = state.get('manager_result', {})
                response = manager_result.get('message', '안녕하세요! 무엇을 도와드릴까요?')
                
            else:
                # 기본 응답
                response = "요청이 처리되었습니다. 추가로 도움이 필요하시면 말씀해주세요!"
        
        # 상태 업데이트
        state['bot_response'] = response
        state['response_generated'] = True
        
        # LangGraph Studio UI를 위한 messages 업데이트
        from langchain_core.messages import AIMessage
        if 'messages' not in state:
            state['messages'] = []
        state['messages'].append(AIMessage(content=response))
        
        # 컨텍스트 업데이트 (다음 대화를 위해)
        if state.get('reservation_info'):
            state['context'] = state['reservation_info']
        
        print(f"✅ 응답 생성 완료")
        
    except Exception as e:
        print(f"❌ 응답 생성 오류: {e}")
        state['error'] = str(e)
        state['error_step'] = 'response_generation'
        state['response_generated'] = False
    
    return state

def run_hospital_reservation(user_query: str, session_id: str = None) -> Dict[str, Any]:
    """병원 예약 시스템 실행"""
    
    # 워크플로우 생성
    workflow = create_hospital_reservation_workflow()
    
    # 초기 상태 생성
    initial_state = create_initial_state(user_query, session_id)
    
    # 워크플로우 실행
    final_state = workflow.invoke(initial_state)
    
    return {
        "success": not bool(final_state.get('error')),
        "response": final_state.get('bot_response', ''),
        "routing_info": final_state.get('routing_info', {}),
        "session_id": final_state.get('session_id'),
        "conversation_history": final_state.get('conversation_history', []),
        "error": final_state.get('error'),
        "error_step": final_state.get('error_step'),
        "context": final_state.get('context', {}),
        "reservation_info": final_state.get('reservation_info', {}),
        "manager_result": final_state.get('manager_result', {}),
        "reservation_result": final_state.get('reservation_result', {}),
        "rag_result": final_state.get('rag_result', {})
    }

def run_hospital_reservation_with_session_data(user_query: str, session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
    """세션 데이터를 포함한 병원 예약 시스템 실행 (2차 대화용)"""
    
    # 워크플로우 생성
    workflow = create_hospital_reservation_workflow()
    
    # 초기 상태 생성
    initial_state = create_initial_state(user_query, session_id)
    
    # 세션 데이터를 상태에 추가
    if session_data:
        initial_state['context'] = session_data.get('context', {})
        initial_state['reservation_info'] = session_data.get('reservation_info', {})
    
    # 워크플로우 실행
    final_state = workflow.invoke(initial_state)
    
    return {
        "success": not bool(final_state.get('error')),
        "response": final_state.get('bot_response', ''),
        "routing_info": final_state.get('routing_info', {}),
        "session_id": final_state.get('session_id'),
        "conversation_history": final_state.get('conversation_history', []),
        "error": final_state.get('error'),
        "error_step": final_state.get('error_step'),
        "context": final_state.get('context', {}),
        "reservation_info": final_state.get('reservation_info', {}),
        "manager_result": final_state.get('manager_result', {}),
        "reservation_result": final_state.get('reservation_result', {}),
        "rag_result": final_state.get('rag_result', {})
    }

def run_continuous_reservation_flow(user_queries: List[str], session_id: str = None) -> Dict[str, Any]:
    """연속적인 예약 플로우 실행 (세션 상태 유지)"""
    
    # 워크플로우 생성
    workflow = create_hospital_reservation_workflow()
    
    # 초기 상태 생성
    current_state = create_initial_state(user_queries[0], session_id)
    
    # 각 쿼리별로 워크플로우 실행 (상태 유지)
    for i, query in enumerate(user_queries):
        print(f"\n🔄 단계 {i+1}: {query}")
        
        # 현재 쿼리로 상태 업데이트
        current_state['user_query'] = query
        
        # 워크플로우 실행
        current_state = workflow.invoke(current_state)
        
        # 오류 발생 시 중단
        if current_state.get('error'):
            break
    
    return {
        "success": not bool(current_state.get('error')),
        "response": current_state.get('bot_response', ''),
        "routing_info": current_state.get('routing_info', {}),
        "session_id": current_state.get('session_id'),
        "conversation_history": current_state.get('conversation_history', []),
        "error": current_state.get('error'),
        "error_step": current_state.get('error_step'),
        "context": current_state.get('context', {}),
        "reservation_info": current_state.get('reservation_info', {}),
        "manager_result": current_state.get('manager_result', {}),
        "reservation_result": current_state.get('reservation_result', {}),
        "rag_result": current_state.get('rag_result', {})
    }

def run_workflow_for_studio(user_query: str, session_id: str = None) -> Dict[str, Any]:
    """LangGraph Studio에서 사용할 워크플로우 실행 함수"""
    return run_hospital_reservation(user_query, session_id)

# LangGraph Studio에서 직접 테스트할 수 있는 함수들
# (노트북에서 더 상세한 테스트가 가능하므로 기본 함수들만 유지)

def test_basic_workflow():
    """기본 워크플로우 테스트"""
    return run_workflow_for_studio("안녕하세요, 도움이 필요해요.")

# LangGraph Studio UI에서 사용할 수 있는 그래프 생성 함수
def create_graph_for_studio():
    """LangGraph Studio UI에서 사용할 그래프 생성"""
    return create_hospital_reservation_workflow()

# LangGraph Studio UI에서 사용할 수 있는 간단한 실행 함수
def run_chat_workflow(state: HospitalReservationState) -> HospitalReservationState:
    """LangGraph Studio UI chat 기능을 위한 워크플로우 실행"""
    try:
        # 워크플로우 생성
        workflow = create_hospital_reservation_workflow()
        
        # 워크플로우 실행
        final_state = workflow.invoke(state)
        
        return final_state
        
    except Exception as e:
        print(f"❌ 워크플로우 실행 오류: {e}")
        state['error'] = str(e)
        state['error_step'] = 'workflow_execution'
        return state