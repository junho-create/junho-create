# MCP 서버 설정 가이드

## **개요**
LangChain MCP Adapters를 사용하여 Supabase와 연동하기 위한 MCP 서버 설정 가이드입니다.

## **1. MCP 서버 설치 및 설정**

### **1.1 Node.js MCP 서버 설치**
```bash
# Node.js MCP 서버 설치
npm install -g @modelcontextprotocol/server

# 또는 로컬 설치
npm install @modelcontextprotocol/server
```

### **1.2 Supabase MCP 서버 설정**
```bash
# Supabase MCP 서버 설치
npm install @supabase/mcp-server
```

## **2. 환경 변수 설정**

### **2.1 .env 파일 설정**
```bash
# Supabase 설정
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here

# MCP 서버 설정
MCP_SERVER_URL=http://localhost:3001
MCP_API_KEY=your_secret_mcp_key_here
```

### **2.2 MCP 서버 실행**
```bash
# MCP 서버 실행 (포트 3001)
npx @supabase/mcp-server --port 3001 --api-key your_secret_mcp_key_here
```

## **3. 테스트**

### **3.1 MCP 서버 연결 테스트**
```bash
# MCP 서버 연결 테스트
curl -X POST http://localhost:3001/health
```

### **3.2 LangChain MCP Adapters 테스트**
```bash
# LangChain MCP Adapters 테스트
python test_langchain_mcp.py
```

## **4. 문제 해결**

### **4.1 MCP 서버 연결 실패**
- MCP 서버가 실행 중인지 확인
- 포트 3001이 사용 가능한지 확인
- 방화벽 설정 확인

### **4.2 Supabase 연결 실패**
- Supabase URL과 API 키 확인
- Supabase 프로젝트 상태 확인
- 네트워크 연결 확인

### **4.3 도구 실행 실패**
- MCP 서버 로그 확인
- 환경 변수 설정 확인
- 의존성 설치 확인

## **5. 고급 설정**

### **5.1 커스텀 MCP 서버**
```javascript
// custom-mcp-server.js
const { Server } = require('@modelcontextprotocol/server');
const { createClient } = require('@supabase/supabase-js');

const server = new Server({
  name: "supabase-mcp-server",
  version: "1.0.0"
});

// Supabase 클라이언트 초기화
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// 도구 등록
server.addTool({
  name: "supabase_query",
  description: "Supabase 데이터베이스 쿼리 실행",
  parameters: {
    type: "object",
    properties: {
      table: { type: "string" },
      operation: { type: "string" },
      data: { type: "object" }
    }
  },
  handler: async (params) => {
    // Supabase 쿼리 실행 로직
    return await executeSupabaseQuery(params);
  }
});

server.start();
```

### **5.2 Docker를 사용한 MCP 서버 실행**
```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
EXPOSE 3001

CMD ["npm", "start"]
```

```bash
# Docker 실행
docker build -t supabase-mcp-server .
docker run -p 3001:3001 supabase-mcp-server
```

## **6. 모니터링**

### **6.1 MCP 서버 상태 확인**
```bash
# 서버 상태 확인
curl http://localhost:3001/status

# 로그 확인
tail -f mcp-server.log
```

### **6.2 성능 모니터링**
- MCP 서버 응답 시간 측정
- Supabase 쿼리 성능 모니터링
- 에러 로그 추적

## **7. 보안 고려사항**

### **7.1 API 키 보안**
- 환경 변수로 API 키 관리
- 프로덕션에서는 암호화된 키 사용
- 정기적인 키 로테이션

### **7.2 네트워크 보안**
- MCP 서버는 내부 네트워크에서만 접근 가능하도록 설정
- HTTPS 사용 권장
- 방화벽 규칙 설정

## **8. 트러블슈팅**

### **8.1 일반적인 오류**
- `MCP_SERVER_URL` 설정 오류
- Supabase 연결 타임아웃
- 도구 실행 권한 오류

### **8.2 로그 확인**
```bash
# MCP 서버 로그
tail -f /var/log/mcp-server.log

# Supabase 연결 로그
tail -f /var/log/supabase-mcp.log
```

## **9. 참고 자료**

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Supabase MCP 서버](https://github.com/supabase/mcp-server)
