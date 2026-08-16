#!/usr/bin/env python3
"""
RAG Doctor Agent A2A Protocol 래퍼
내부 Agent를 A2A Protocol 호환 형태로 래핑합니다.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 현재 디렉토리를 Python 경로에 추가
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 프로젝트 루트를 Python 경로에 추가 (패키지 임포트를 위해 상위 디렉토리 필요)
PACKAGE_ROOT = os.path.dirname(ROOT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# 폴더 구조 변경 호환: 우선 main/agent 경로 시도, 실패 시 기존 경로 시도
build_and_run_agent = None
OutputSchema = None
try:
    from rag_doctor_agent.main.agent.graph import build_and_run_agent as _run
    from rag_doctor_agent.main.agent.output_enforcer import OutputSchema as _schema
    build_and_run_agent = _run
    OutputSchema = _schema
except Exception:
    try:
        from rag_doctor_agent.agent.graph import build_and_run_agent as _run
        from rag_doctor_agent.agent.output_enforcer import OutputSchema as _schema
        build_and_run_agent = _run
        OutputSchema = _schema
    except Exception:
        pass

# RAG_DATA_DIR 기본값 자동 설정(환경변수 미설정 시)
if not os.getenv("RAG_DATA_DIR"):
    main_data = os.path.join(ROOT_DIR, "main", "data")
    legacy_data = os.path.join(ROOT_DIR, "rag_doctor_agent", "data")
    if os.path.isdir(main_data):
        os.environ["RAG_DATA_DIR"] = main_data
    elif os.path.isdir(legacy_data):
        os.environ["RAG_DATA_DIR"] = legacy_data

app = Flask(__name__)

class RAGDoctorA2AWrapper:
    """RAG Doctor Agent A2A 래퍼"""
    
    def __init__(self):
        self.name = "RAG Doctor Agent"
        self.version = "1.0.0"
        self.description = "의료진 추천 및 증상 기반 진료과 매칭 Agent"
        self.capabilities = [
            "doctor_recommendation",
            "symptom_analysis", 
            "department_matching",
            "medical_staff_selection"
        ]
    
    def process_recommendation(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """의료진 추천 처리"""
        try:
            # RAG Doctor Agent 실행
            result: OutputSchema = build_and_run_agent(patient_data)
            
            # A2A Protocol 형식으로 변환
            a2a_response = {
                "success": True,
                "action": "recommend_doctor",
                "timestamp": datetime.now().isoformat(),
                "input_data": patient_data,
                "output_data": result.model_dump(by_alias=True),
                "agent_info": {
                    "name": self.name,
                    "version": self.version,
                    "capabilities": self.capabilities
                }
            }
            
            return a2a_response
            
        except Exception as e:
            return {
                "success": False,
                "action": "recommend_doctor",
                "error": f"RAG Doctor Agent processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "agent_info": {
                    "name": self.name,
                    "version": self.version
                }
            }

# 글로벌 래퍼 인스턴스
rag_wrapper = RAGDoctorA2AWrapper()

@app.route("/.well-known/agent.json", methods=["GET"])
def agent_card():
    """A2A Protocol Agent Card"""
    card = {
        "schema_version": "1.0.0",
        "name": rag_wrapper.name,
        "version": rag_wrapper.version,
        "description": rag_wrapper.description,
        "capabilities": rag_wrapper.capabilities,
        "endpoints": {
            "process": "/a2a/process",
            "status": "/a2a/status"
        },
        "auth": {
            "type": "none",
            "description": "No authentication required"
        },
        "supported_actions": [
            {
                "action": "recommend_doctor",
                "description": "환자 증상 기반 의료진 추천",
                "input_schema": {
                    "type": "object",
                    "required": ["patient_name", "symptoms"],
                    "properties": {
                        "patient_name": {"type": "string"},
                        "patient_gender": {"type": "string"},
                        "phone_num": {"type": "string"},
                        "chat_start_date": {"type": "string"},
                        "symptoms": {"type": "array", "items": {"type": "string"}},
                        "visit_type": {"type": "string"},
                        "preference_datetime": {"type": "array", "items": {"type": "string"}},
                        "dept": {"type": "string"},
                        "doctor_name": {"type": "string"},
                        "other_info": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "patient_name": {"type": "string"},
                        "patient_gender": {"type": "string"},
                        "phone_num": {"type": "string"},
                        "chat_start_date": {"type": "string"},
                        "symptoms": {"type": "array", "items": {"type": "string"}},
                        "visit_type": {"type": "string"},
                        "preference_datetime": {"type": "array", "items": {"type": "string"}},
                        "dept": {"type": "string"},
                        "doctor_name": {"type": "string"},
                        "top_k_suggestions": {"type": "array"},
                        "retrieval_evidence": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        ],
        "contact": {
            "support_url": "https://github.com/example/rag-doctor-agent"
        },
        "created_at": datetime.now().isoformat()
    }
    return jsonify(card)

@app.route("/a2a/status", methods=["GET"])
def status():
    """A2A Protocol 상태 체크"""
    try:
        ok = bool(build_and_run_agent and OutputSchema)
        return jsonify({
            "status": "ok" if ok else "warn",
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "agent": {"name": rag_wrapper.name, "version": rag_wrapper.version}
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 200

@app.route("/a2a/process", methods=["POST"])
def process_request():
    """A2A Protocol 요청 처리"""
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
        
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                "success": False,
                "error": "Empty request data"
            }), 400
        
        action = request_data.get("action", "")
        data = request_data.get("data", {})
        
        # A2A Protocol 헤더 확인
        a2a_version = request.headers.get("A2A-Version", "1.0.0")
        a2a_request_id = request.headers.get("A2A-Request-ID", "unknown")
        
        if action == "recommend_doctor":
            response = rag_wrapper.process_recommendation(data)
        else:
            response = {
                "success": False,
                "action": action,
                "error": f"Unsupported action: {action}",
                "supported_actions": ["recommend_doctor"]
            }
        
        # A2A 메타데이터 추가
        response["a2a_metadata"] = {
            "request_id": a2a_request_id,
            "version": a2a_version,
            "processed_at": datetime.now().isoformat(),
            "processed_by": rag_wrapper.name
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "a2a_metadata": {
                "request_id": request.headers.get("A2A-Request-ID", "unknown"),
                "version": request.headers.get("A2A-Version", "1.0.0"),
                "processed_at": datetime.now().isoformat(),
                "processed_by": rag_wrapper.name
            }
        }), 500

# 중복된 상태 엔드포인트 제거(위 status 사용)

@app.route("/", methods=["GET"])
def home():
    """기본 정보 페이지"""
    return jsonify({
        "message": "RAG Doctor Agent (A2A Protocol Wrapper)",
        "agent_card": "/.well-known/agent.json",
        "process_endpoint": "/a2a/process",
        "status_endpoint": "/a2a/status",
        "description": rag_wrapper.description
    })

if __name__ == "__main__":
    print("🩺 RAG Doctor Agent A2A Wrapper 시작")
    print("📋 Agent Card: http://localhost:5001/.well-known/agent.json")
    print("🔄 Process Endpoint: http://localhost:5001/a2a/process")
    print("📊 Status: http://localhost:5001/a2a/status")
    print("="*60)
    
    app.run(host="0.0.0.0", port=5001, debug=True)

