# 🚀 LangGraph Studio + LangSmith 연동 가이드

## 📋 사전 준비사항

### 1. 환경 변수 설정
```bash
# .env 파일 생성 (env_example_langgraph.txt 참고)
cp env_example_langgraph.txt .env

# 필수 환경 변수 설정
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=hospital_agent_v3
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. 의존성 설치
```bash
# 모든 의존성 설치
pip install -r requirements.txt

# LangGraph CLI 설치 확인
pip install langgraph-cli
```

## 🎯 LangGraph Studio 실행

### 1. 기본 실행
```bash
# LangGraph Studio 시작
python run_langgraph_studio.py

# 또는 직접 실행
langgraph dev
```

### 2. 브라우저 접속
- **URL**: http://localhost:8123
- **Graph**: medical_reservation
- **Entry Point**: ./main/langgraph_workflow.py:create_hospital_reservation_workflow

## 🔍 LangSmith 연동

### 1. LangSmith 계정 설정
1. [LangSmith](https://smith.langchain.com) 계정 생성
2. API 키 발급
3. 프로젝트 생성 (hospital_agent_v3)

### 2. 환경 변수 설정
```bash
# .env 파일에 추가
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=hospital_agent_v3
```

### 3. 연동 확인
```bash
# LangSmith 연동 테스트
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('LangSmith 연동:', os.getenv('LANGSMITH_TRACING'))
print('프로젝트:', os.getenv('LANGSMITH_PROJECT'))
"
```

## 🧪 테스트 명령어

### 1. 개별 컴포넌트 테스트
```bash
# Agent1 분기 테스트
python test_agent1_routing.py

# RAG Agent 테스트
python test_rag_agent.py

# Tool Calling 테스트
python test_tool_calling.py
```

### 2. 통합 테스트
```bash
# 일정 조회 및 예약 확정 테스트
python test_schedule_booking.py

# 다중 턴 대화 테스트
python test_multi_turn_conversation.py

# 전체 시스템 테스트
python test_agent_system.py
```

### 3. LangGraph Studio UI 테스트
```bash
# LangGraph Studio 실행
python run_langgraph_studio.py

# 브라우저에서 테스트:
# - "안녕하세요, 예약하고 싶어요"
# - "무릎이 아파서 최대한 빨리 예약하고 싶어요"
# - "허리가 아프고 디스크가 있어서 예약하고 싶어요"
# - "병원 휴무일이 언제인가요?"
```

## 🔧 문제 해결

### 1. LangGraph Studio 실행 오류
```bash
# LangGraph CLI 설치 확인
pip install langgraph-cli

# 의존성 재설치
pip install -r requirements.txt
```

### 2. LangSmith 연동 오류
```bash
# 환경 변수 확인
python -c "import os; print(os.getenv('LANGSMITH_TRACING'))"

# LangSmith 패키지 설치 확인
pip install langsmith
```

### 3. Supabase 연결 오류
```bash
# Supabase 환경 변수 확인
python -c "import os; print(os.getenv('SUPABASE_URL'))"

# Supabase 패키지 설치 확인
pip install supabase
```

## 📊 모니터링

### 1. LangSmith 대시보드
- **URL**: https://smith.langchain.com
- **프로젝트**: hospital_agent_v3
- **추적**: 모든 LLM 호출, 에이전트 실행, 도구 사용

### 2. LangGraph Studio UI
- **워크플로우 시각화**: 노드별 실행 상태
- **디버깅**: 각 단계별 결과 확인
- **테스트**: 다양한 시나리오 테스트

## 🎉 성공 확인

### 1. LangGraph Studio UI 접속 성공
- 브라우저에서 http://localhost:8123 접속
- medical_reservation 그래프 로드
- 테스트 입력으로 응답 확인

### 2. LangSmith 추적 활성화
- LangSmith 대시보드에서 실행 로그 확인
- 프로젝트별 추적 데이터 수집
- 성능 분석 및 최적화

### 3. 전체 시스템 동작
- Agent1 → Agent2 → Agent3 플로우
- 자연어 일정 처리
- 의료진 추천 → 일정 조회 → 예약 확정
