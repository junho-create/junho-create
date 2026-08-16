# langgraph_state.py
"""
LangGraph 상태 모델 정의
병원 예약 시스템의 전체 워크플로우 상태를 관리
"""
from typing import Any, Dict, List, Optional, Union
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
import uuid
from datetime import datetime
from langchain_core.messages import BaseMessage

class HospitalReservationState(TypedDict):
    """병원 예약 시스템의 전체 상태"""
    
    # 기본 정보
    session_id: str
    user_query: str
    timestamp: str
    
    # LangGraph Studio UI chat 기능을 위한 messages 필드
    messages: List[BaseMessage]
    
    # 대화 히스토리
    conversation_history: List[str]  # 단순화: ["user: 메시지", "assistant: 응답"]
    
    # Intent 분류
    user_intent: str  # "create", "check", "cancel"
    intent_classified: bool
    
    # 사용자 정보
    personal_info: str  # JSON 문자열로 저장
    
    # 예약 정보
    reservation_meta: str  # JSON 문자열로 저장
    
    # 파싱 결과
    parsed_data: str  # JSON 문자열로 저장
    parsing_complete: bool
    
    # 증상 매핑 결과
    mapped_symptoms: str  # JSON 문자열로 저장
    symptom_mapping_complete: bool
    
    # 의사 매핑 결과
    mapped_doctors: str  # JSON 문자열로 저장
    doctor_mapping_complete: bool
    
    # 스케줄 확인 결과
    available_slots: str  # JSON 문자열로 저장
    selected_slot: str  # JSON 문자열로 저장
    schedule_check_complete: bool
    
    # 최종 예약 정보
    final_reservation: str  # JSON 문자열로 저장
    reservation_created: bool
    
    # Supabase 저장 관련
    supabase_saved: bool
    supabase_reservation_id: int
    
    # 응답 정보
    bot_response: str
    response_generated: bool
    
    # 세션 상태 관리
    session_step: str  # "initial", "waiting_confirmation", "completed"
    conversation_round: int
    
    # 2차 대화 관련
    second_intent: str  # "confirm", "modify", "other"
    second_intent_classified: bool
    
    # 대기 중인 데이터
    pending_slots: str  # JSON 문자열로 저장
    pending_reservation_info: str  # JSON 문자열로 저장
    
    # 예약 확인 관련
    found_reservations: str  # JSON 문자열로 저장
    reservation_check_complete: bool
    
    # 에러 정보
    error: str
    error_step: str
    
    # 새로운 에이전트 시스템 필드들
    manager_result: Optional[Dict[str, Any]]
    routing_info: Optional[Dict[str, Any]]
    manager_complete: bool
    
    reservation_result: Optional[Dict[str, Any]]
    reservation_info: Optional[Dict[str, Any]]
    reservation_status: str
    reservation_complete: bool
    
    rag_result: Optional[Dict[str, Any]]
    rag_complete: bool
    
    response_complete: bool
    
    # 컨텍스트 정보
    context: Optional[Dict[str, Any]]

def create_initial_state(user_query: str = "", session_id: Optional[str] = None) -> HospitalReservationState:
    """초기 상태 생성"""
    from langchain_core.messages import HumanMessage
    
    return HospitalReservationState(
        session_id=session_id or str(uuid.uuid4()),
        user_query=user_query,
        timestamp=datetime.now().isoformat(),
        messages=[HumanMessage(content=user_query)] if user_query else [],
        conversation_history=[],
        user_intent="",
        intent_classified=False,
        personal_info="{}",
        reservation_meta="{}",
        parsed_data="{}",
        parsing_complete=False,
        mapped_symptoms="[]",
        symptom_mapping_complete=False,
        mapped_doctors="[]",
        doctor_mapping_complete=False,
        available_slots="[]",
        selected_slot="{}",
        schedule_check_complete=False,
        final_reservation="{}",
        reservation_created=False,
        supabase_saved=False,
        supabase_reservation_id=0,
        bot_response="",
        response_generated=False,
        # 세션 상태 관리
        session_step="initial",
        conversation_round=0,
        # 2차 대화 관련
        second_intent="",
        second_intent_classified=False,
        # 대기 중인 데이터
        pending_slots="[]",
        pending_reservation_info="{}",
        # 예약 확인 관련
        found_reservations="[]",
        reservation_check_complete=False,
        # 에러 정보
        error="",
        error_step="",
        
        # 새로운 에이전트 시스템 필드들
        manager_result=None,
        routing_info=None,
        manager_complete=False,
        
        reservation_result=None,
        reservation_info=None,
        reservation_status="",
        reservation_complete=False,
        
        rag_result=None,
        rag_complete=False,
        
        response_complete=False,
        
        # 컨텍스트 정보
        context=None
    )