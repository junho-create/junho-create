# main.py - Medical Reservation Chat API
"""
병원 예약 시스템 Chat API 서버
LangGraph 워크플로우를 사용한 예약 처리 API
"""
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 상위 디렉토리의 모듈들을 import하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangGraph 워크플로우 import
from langgraph_workflow import (
    run_hospital_reservation, 
    run_hospital_reservation_with_session_data
)

# FastAPI 앱 생성
app = FastAPI(
    title="Medical Reservation Chat API",
    description="병원 예약 시스템을 위한 LangGraph 기반 Chat API",
    version="250908-v1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델 정의
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
    session_id: str
    reservation_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_step: Optional[str] = None
    timestamp: str

class ServerStateResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# 한국 시간대 설정
KST = timezone(timedelta(hours=9))

# 세션 관리 (간단한 메모리 기반)
active_sessions: Dict[str, Dict[str, Any]] = {}

def get_korean_time():
    """한국 시간 반환"""
    return datetime.now(KST).isoformat()

def get_session(session_id: str) -> Dict[str, Any]:
    """세션 정보 가져오기"""
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "created_at": get_korean_time(),
            "message_count": 0,
            "last_activity": get_korean_time(),
            "current_step": "initial",  # 세션 상태 추가
            "conversation_round": 0,    # 대화 라운드 추가
            "pending_data": {}          # 대기 중인 데이터 추가
        }
    return active_sessions[session_id]

def update_session(session_id: str):
    """세션 정보 업데이트"""
    session = get_session(session_id)
    session["message_count"] += 1
    session["last_activity"] = get_korean_time()

def update_session_step(session_id: str, new_step: str, data: Dict = None):
    """세션 상태 업데이트"""
    session = get_session(session_id)
    session["current_step"] = new_step
    session["conversation_round"] += 1
    if data:
        session["pending_data"].update(data)
    session["last_activity"] = get_korean_time()
    print(f"🔄 세션 상태 변경: {session_id} -> {new_step} (라운드: {session['conversation_round']})")

def should_wait_for_confirmation(result: Dict) -> bool:
    """확인 대기가 필요한지 판단"""
    return (result.get("success") and 
            result.get("user_intent") == "create" and 
            result.get("available_slots") and 
            not result.get("reservation_info"))


@app.get("/server_state_check", response_model=ServerStateResponse, tags=["default - 서버 실행 확인"])
async def server_state_check():
    """서버 상태 확인 엔드포인트"""
    return ServerStateResponse(
        status="healthy",
        timestamp=get_korean_time(),
        version="250908-v1.0.0"
    )

@app.post("/chat", response_model=ChatResponse, tags=["chat - chat api 호출"])
async def chat(request: ChatRequest):
    """
    채팅 메시지 처리 엔드포인트
    
    Args:
        request: 채팅 요청 (메시지, 세션 ID, 사용자 정보)
    
    Returns:
        ChatResponse: 처리 결과 및 응답
    """
    try:
        # 세션 ID 생성 또는 사용
        session_id = request.session_id or str(uuid.uuid4())
        session = get_session(session_id)
        
        print(f"\n{'='*50}")
        print(f"새로운 채팅 요청 - 세션: {session_id}")
        print(f"현재 세션 상태: {session['current_step']} (라운드: {session['conversation_round']})")
        print(f"메시지: {request.message}")
        print(f"{'='*50}")
        
        # 세션 상태에 따른 워크플로우 실행
        if session["current_step"] == "waiting_confirmation":
            # 2차 대화: 세션 데이터를 상태에 추가
            result = run_hospital_reservation_with_session_data(
                user_query=request.message,
                session_id=session_id,
                session_data=session["pending_data"]
            )
        else:
            # 1차 대화: 워크플로우 실행
            result = run_hospital_reservation(
                user_query=request.message,
                session_id=session_id
            )
        
        # 워크플로우 결과에 따른 세션 상태 업데이트
        if result.get("session_step"):
            update_session_step(session_id, result["session_step"], {
                "available_slots": result.get("available_slots", []),
                "pending_slots": result.get("pending_slots", []),
                "pending_reservation_info": result.get("pending_reservation_info", {})
            })
        
        # 세션 업데이트 (워크플로우 실행 후)
        update_session(session_id)
        
        # 응답 생성
        response = ChatResponse(
            success=result["success"],
            response=result["response"],
            session_id=session_id,
            reservation_info=result.get("reservation_info"),
            error=result.get("error"),
            error_step=result.get("error_step"),
            timestamp=get_korean_time()
        )
        
        print(f"응답 생성 완료 - 성공: {result['success']}")
        if result.get("error"):
            print(f"오류: {result['error']} (단계: {result['error_step']})")
        
        return response
        
    except Exception as e:
        print(f"❌ API 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )

@app.get("/sessions/info/{session_id}", tags=["session - 세션 확인 (활성 세션 리스트 조회, 세션 정보 확인, 세션 삭제)"])
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return {
        "session_id": session_id,
        "session_info": active_sessions[session_id]
    }

@app.delete("/sessions/delete/{session_id}", tags=["session - 세션 확인 (활성 세션 리스트 조회, 세션 정보 확인, 세션 삭제)"])
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    del active_sessions[session_id]
    return {"message": "세션이 삭제되었습니다"}

@app.get("/sessions/list", tags=["session - 세션 확인 (활성 세션 리스트 조회, 세션 정보 확인, 세션 삭제)"])
async def list_sessions():
    """활성 세션 목록 조회"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    }

if __name__ == "__main__":
    print("🏥 Medical Reservation Chat API 서버 시작...")
    print("📡 API 문서: http://localhost:8000/docs")
    print("🔗 채팅 엔드포인트: http://localhost:8000/chat")
    print("💚 헬스 체크: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )