# 🧪 테스트 가이드 (업데이트됨)

## 개별 테스트 스크립트들

### 1. Agent1 분기 테스트
```bash
python test_agent1_routing.py
```
**테스트 내용:**
- Agent1이 사용자 요청을 분석하고 적절한 에이전트/도구로 라우팅
- 예약, 증상-의료진, 병원 정보, 모호한 요청 처리
- Tool Calling 및 Agent2 통합 테스트

### 2. RAG Agent 테스트
```bash
python test_rag_agent.py
```
**테스트 내용:**
- Agent3 (RAG)가 증상을 분석하고 의료진 추천
- 관절, 척추, 소화기, 신경, 응급 관련 증상 테스트
- RAG 파이프라인 통합 및 폴백 테스트
- Agent2 → Agent3 통합 테스트

### 3. Tool Calling 테스트
```bash
python test_tool_calling.py
```
**테스트 내용:**
- Agent2의 Tool Calling 기능
- Supabase 도구 바인딩 및 실행
- 전화번호 기반 환자 조회 로직
- 자연어 일정 처리 테스트

### 4. 일정 조회 및 예약 확정 테스트
```bash
python test_schedule_booking.py
```
**테스트 내용:**
- 의료진 추천 → 일정 조회 → 예약 확정 플로우
- 자연어 일정 처리 ("최대한 빨리", "내일 오후" 등)
- SupabaseDoctorLookupTool, SupabaseScheduleLookupTool 테스트
- 완전한 예약 플로우 시뮬레이션

### 5. 다중 턴 대화 테스트
```bash
python test_multi_turn_conversation.py
```
**테스트 내용:**
- 복합 시나리오: 증상 → 의료진 매핑 → 예약
- 단계별 정보 수집 (이름, 전화번호, 증상, 일정 선호도)
- 컨텍스트 유지 및 정보 누적
- 자연어 일정 처리 시나리오

### 6. 전체 시스템 통합 테스트
```bash
python test_agent_system.py
```
**테스트 내용:**
- 3-Agent 시스템 워크플로우 시각화
- 자연어 일정 처리 시뮬레이션
- 의료진 추천 → 일정 조회 → 예약 확정 플로우
- 성능 분석 및 통계

## 테스트 실행 순서

### 1단계: 개별 컴포넌트 테스트
```bash
# Agent1 분기 테스트
python test_agent1_routing.py

# RAG Agent 테스트  
python test_rag_agent.py

# Tool Calling 테스트
python test_tool_calling.py
```

### 2단계: 기능별 통합 테스트
```bash
# 일정 조회 및 예약 확정 테스트
python test_schedule_booking.py

# 다중 턴 대화 테스트
python test_multi_turn_conversation.py
```

### 3단계: 전체 시스템 테스트
```bash
# 워크플로우 시각화 및 시뮬레이션
python test_agent_system.py
```

## 새로운 기능 테스트

### 자연어 일정 처리
- **"최대한 빨리"** → 긴급도 높음, 가장 빠른 일정 추천
- **"내일 오후"** → 내일 오후 시간대 일정만 필터링
- **"다음 주 월요일 오전"** → 특정 요일 오전 시간대 일정 추천
- **"급하게"** → 긴급도 높음, 즉시 가능한 일정 추천

### 의료진 추천 → 일정 조회 → 예약 확정
1. **증상 분석** → RAG 에이전트로 의료진 추천
2. **의료진 조회** → SupabaseDoctorLookupTool로 DocID 조회
3. **일정 조회** → SupabaseScheduleLookupTool로 가용일정 조회
4. **자연어 매칭** → 사용자 선호도에 맞는 일정 필터링
5. **예약 확정** → SupabaseCreateTool로 예약 생성

## 예상 결과

### ✅ 성공 시나리오
- **Agent1**: 요청 분석 → 적절한 라우팅
- **Agent2**: Tool Calling → Supabase 작업 성공
- **Agent3**: 증상 분석 → 의료진 추천 성공
- **통합**: 전체 플로우 원활한 작동

### ⚠️ 주의사항
- 환경 변수 설정 확인 (`.env` 파일)
- Supabase 연결 상태 확인
- OpenAI API 키 설정 확인
- RAG 파이프라인 초기화 확인

## 문제 해결

### 1. Agent1 라우팅 오류
- 프롬프트 확인 및 수정
- 의도 분석 로직 점검

### 2. Agent2 Tool Calling 실패
- Supabase 연결 상태 확인
- 도구 바인딩 상태 확인
- 컬럼명 매핑 확인

### 3. Agent3 RAG 실패
- RAG 파이프라인 초기화 확인
- 증상-의료진 매핑 데이터 확인

### 4. 통합 테스트 실패
- 개별 컴포넌트 테스트 먼저 실행
- 에러 로그 확인
- 환경 설정 재확인

## 성능 지표

### 목표 성능
- **Agent1 정확도**: >90%
- **Agent2 Tool Calling 성공률**: >95%
- **Agent3 추천 정확도**: >85%
- **전체 플로우 성공률**: >90%

### 모니터링 포인트
- 응답 시간
- 에러 발생률
- 사용자 만족도
- 시스템 안정성
