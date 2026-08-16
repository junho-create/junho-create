# 침착맨 아바타 데모 실행 순서

팀 데모용 최소 단계. `.env.local`이 채워져 있다고 가정합니다.

**TTS는 기본 ElevenLabs**를 사용합니다. `.env.local`에 `ELEVEN_API_KEY`가 필요합니다. OpenAI TTS로 바꾸려면 `TTS_PROVIDER=openai`를 설정하세요.

---

## 체크리스트

| 순서 | 할 일 | 명령/동작 |
|------|--------|-----------|
| ① | **avatar 폴더로 이동** | 저장소 clone 후 `cd avatar` (이 저장소에는 avatar와 agent-starter-react가 함께 있음) |
| ② | **모델 다운로드** (최초 1회만 해도 됨) | `uv run python src/agent.py download-files` |
| ③ | **음성만 빠르게 테스트** (아바타 없이) | `uv run python src/agent.py console` → 마이크로 말해 보기. 종료: **Ctrl+C** |
| ④ | **로컬 dev + 웹** | 터미널에서 `uv run python src/agent.py dev` 켜 둔 뒤, agent-starter-react에서 `pnpm dev` 실행하고 브라우저에서 토큰 발급해 프론트 접속 |

- **③**까지 성공하면 STT → LLM → TTS 파이프라인은 동작하는 상태입니다.

---

## DEV로 아바타 데모 실행하기 (단계별)

에이전트를 **로컬(DEV)**에서 돌리고, 브라우저에서 아바타를 보며 음성으로 대화하는 방법입니다. **프로젝트에 Token server가 켜져 있고 Sandbox ID가 `myavatar-1j0vgl`** 라고 가정합니다.

---

### 1단계: 사전 준비 (avatar 쪽)

1. **avatar 폴더로 이동**
   ```powershell
   cd c:\dsl\modeling\26_1_modeling_project_multimodal\avatar
   ```
2. **`.env.local` 확인**  
   아래 값이 채워져 있어야 합니다.
   - `LIVEKIT_URL` (예: `wss://my-avatar-vjswic1n.livekit.cloud`)
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `OPENAI_API_KEY`
   - `DEEPGRAM_API_KEY`
   - `ELEVEN_API_KEY` (기본 TTS는 ElevenLabs)
   - `SIMLI_API_KEY`, `SIMLI_FACE_ID` (아바타 영상용)
   - (선택) 아바타 룸에서만 다른 TTS를 쓰려면 `TTS_PROVIDER_AVATAR=openai` 등 설정
3. **모델 다운로드 (최초 1회만)**
   ```powershell
   uv run python src/agent.py download-files
   ```

---

### 2단계: 에이전트(로컬) 켜기

1. **같은 avatar 폴더**에서 터미널을 하나 열고:
   ```powershell
   uv run python src/agent.py dev
   ```
2. 로그에 에이전트가 등록되었다는 메시지가 보일 때까지 대기.
3. **이 터미널은 그대로 두고** 다음 단계로 이동합니다.

---

### 3단계: 프론트엔드 준비 (최초 1회)

이 저장소를 push할 때 **avatar**와 **agent-starter-react** 폴더가 함께 포함되어 있으므로, 팀원은 저장소를 clone하면 두 폴더를 모두 받습니다. 별도로 agent-starter-react를 clone할 필요 없습니다.

1. **프로젝트 루트**에서 agent-starter-react 폴더로 이동:
   ```powershell
   cd c:\dsl\modeling\26_1_modeling_project_multimodal\agent-starter-react
   ```
2. **의존성 설치**
   ```powershell
   pnpm install
   ```
3. **환경 변수 설정**
   - `.env.example`을 복사해 `.env.local` 만듦.
   - `.env.local`에 다음을 넣습니다 (avatar의 `.env.local`에 있는 값과 맞춤).
     ```env
     NEXT_PUBLIC_LIVEKIT_URL=wss://my-avatar-vjswic1n.livekit.cloud
     ```
     (실제 프로젝트의 LiveKit URL로 바꾸세요. avatar의 `LIVEKIT_URL`과 동일하게.)
   - **Sandbox Token server**를 쓰려면, 앱이 요구하는 변수에 **Sandbox ID**를 넣습니다. 예:
     ```env
     NEXT_PUBLIC_LIVEKIT_SANDBOX_ID=myavatar-1j0vgl
     ```
     (agent-starter-react의 README나 `.env.example`에 변수 이름이 다르면 그에 맞춰 수정.)

---

### 4단계: 프론트엔드 실행

1. **새 터미널**을 열고 agent-starter-react 폴더로 이동:
   ```powershell
   cd c:\dsl\modeling\26_1_modeling_project_multimodal\agent-starter-react
   pnpm dev
   ```
2. 브라우저에서 **http://localhost:3000** 접속.

---

### 5단계: 브라우저에서 연결

1. 앱 화면에서 **시작** / **Connect** 버튼 클릭.  
   (Sandbox ID를 넣어 두었으면 토큰이 자동으로 발급되어 연결됩니다.)
2. 마이크 권한 허용 후 말해 보기.
3. 아바타가 보이고, 음성으로 응답하면 성공.


### 토큰 발급 (LiveKit Cloud)

DEV에서 브라우저 프론트가 룸에 들어가려면 **접속 토큰**이 필요합니다. 아래 두 가지 중 편한 방법을 쓰면 됩니다.

---

#### Sandbox 토큰 서버 (토큰 자동 발급, 추천)

프론트가 **토큰을 직접 입력하지 않고** LiveKit Cloud가 발급해 주는 방식입니다. 한 번 만들면 agent-starter-react에서 Sandbox ID만 넣어 두면 됩니다.

1. [LiveKit Cloud](https://cloud.livekit.io/) 로그인.
2. 왼쪽에서 **프로젝트** 선택 (에이전트를 배포한/등록한 그 프로젝트).
3. **Sandboxes** 메뉴로 이동.
4. **Create sandbox** 또는 **샌드박스 만들기** 클릭.
5. 템플릿 목록에서 **"Token server"** (또는 **"token-server"**) 템플릿 선택 후 생성.
   - 직접 링크: [Sandbox token server 템플릿](https://cloud.livekit.io/projects/p_/sandbox/templates/token-server) (로그인 후 본인 프로젝트로 들어가면 `p_` 가 프로젝트 ID로 바뀐 URL이 됨).
6. 샌드박스 이름 입력(예: `dev-token-server`) 후 **Done** / **생성**.
7. 생성된 샌드박스 상세에서 **Sandbox ID** 확인 (예: `dev-token-server-xxxx` 형태). 이 값을 복사합니다.
8. **agent-starter-react**의 `.env.local`에 이 Sandbox ID를 넣도록 설정합니다 (해당 앱 README의 Sandbox Token Source 설정 참고). 그러면 앱이 토큰을 자동으로 요청해 사용합니다.
9. **에이전트 디스패치**: 같은 프로젝트에서 `uv run python src/agent.py dev` 로 에이전트를 켜 두면, Sandbox로 접속할 때 해당 에이전트가 룸에 디스패치됩니다. agent-starter-react에서 에이전트 이름(예: `my-agent`)을 지정하는 필드가 있으면, `livekit.toml`의 에이전트 id/이름과 맞춥니다.

이렇게 하면 **브라우저에서는 "시작"만 눌러도** 토큰이 자동 발급되고, 로컬 dev 에이전트가 붙습니다.


## 데모 시 보여줄 수 있는 것

| 단계 | 보여주는 것 |
|------|-------------|
| `console` | 음성 인식 → LLM 답변 → TTS 재생 (침펄토론 톤) |
| `dev` + 프론트 | 로컬 에이전트 + React 등 프론트로 아바타 + 음성 대화 |

---
