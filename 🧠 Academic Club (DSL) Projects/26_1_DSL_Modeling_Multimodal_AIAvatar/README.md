# 🎙️ 26-1 멀티모달 모델링 프로젝트
**멀티모달 기반의 AI 아바타 침멀토론 서비스**

Author: 여준호(14기), 이건일(14기), 조재우(14기), 백가은(15기), 이은민(15기)

본 프로젝트는 단순한 대화형 AI를 넘어, 특정 인물의 외형, 말투, 목소리, 사고방식을 학습한 아바타와 실시간으로 토론할 수 있는 멀티모달 서비스를 구현하는 것을 목표로 합니다. 팬들이 그리워하는 '침펄토론' 콘텐츠의 감성을 AI 기술로 재현하였습니다.

## 1. Project Overview
- **배경**: 누구나 한 번쯤 꿈꾸는 '원하는 인물과의 시공간을 초월한 직접적인 소통'을 실시간 AI 아바타 기술로 구현하고자 했습니다. 그 첫 시도로, 누적 조회수 약 5,000만 회를 기록하며 종료된 이후에도 팬들의 지속적인 복귀 요구가 이어지고 있는 '침펄토론' 콘텐츠를 타겟으로 삼았습니다.
- **핵심 가치**: LLM을 활용해 침착맨 특유의 억지 논리, 치졸함, 뻔뻔한 반박 등 독특한 페르소나를 구현.
- **주요 기능**: 사용자의 음성을 인식하고, 침착맨의 페르소나로 생성된 답변을 아바타의 음성과 영상(Lip Sync)으로 실시간 출력.

## 2. Pipeline
본 서비스는 실시간 멀티모달 파이프라인으로 구성되며 다음 주요 단계를 거칩니다.

- **STT (Speech-to-Text)**: OpenAI Whisper API를 이용해 사용자의 음성을 실시간으로 텍스트로 변환합니다.
- **LLM & RAG**: 
  - GPT-4o-mini 및 Qwen3 기반의 모델 운영. 
  - **RAG**: 5,000개 이상의 QA 쌍 및 밈 사전 데이터 기반으로 답변 논리 구조 및 근거 생성.
  - **Persona**: Gemini API 및 Kiwi 형태소 분석기를 활용해 발화 스타일, 억지 논리 등 침착맨 페르소나를 반영하여 텍스트를 생성.
- **TTS (Text-to-Speech)**: ElevenLabs Voice API를 통해 침착맨 음성 클로닝 기술을 적용하여 자연스러운 실시간 음성 생성.
- **Avatar**: ANAM (YARRR)을 통해 생성된 음성과 동기화된 얼굴 클로닝 및 Lip Sync 렌더링.

## 3. Output(Evaluation)
이 프로젝트에서는 정량적 지표(BLEURT)와 정성적 지표(LLM-as-a-Judge)를 결합하여 모델 성능을 검증했습니다.

- **평가 지표**: 내용 정합성, 캐릭터성(Persona Alignment), 수사학적 전략 다양성 등.
- **실험 결과**: RAG가 결합된 Qwen 4B 모델이 평가 총점 57.394점으로 가장 우수한 성능을 도출했습니다.
- **비즈니스 모델**: B2C(이용 건당 결제/정기 구독) 및 B2B(SaaS 고객 지원/교육 아바타) 수익 구조 설계 지원.

## 4. Repository Structure
현재 폴더 구조에 기반한 리포지토리 구성은 다음과 같습니다.

```text
├── 26_1_modeling_project_multimodal/
│   ├── agent-starter-react/   # Next.js 기반의 에이전트 서비스 프론트엔드 UI
│   ├── avatar/                # LiveKit 기반 아바타 백엔드 및 WebRTC 실시간 통신 (STT/TTS 파이프라인 연동)
│   ├── data/                  
│   │   └── rag_datasets/      # RAG 및 모델 학습용 QA 쌍, 밈 사전 등의 데이터셋
│   ├── livekit-app/           # React/Vite 기반 LiveKit 실시간 아바타 렌더링 프론트엔드 앱
│   └── rag/                   # RAG 파이프라인 및 LLM 통합 모듈 (vector_db, avatar_llm 등)
└── README.md                  # 프로젝트 개요 및 안내 문서 (현재 문서)
```

## 5. Additional Information
**⚠️ Demo Availability & Legal Notice**
본 프로젝트는 특정 인물의 지식재산권(초상권 및 퍼블리시티권) 보호와 향후 상업화 가능성 등의 이슈를 고려하여 현재 퍼블릭 환경에서의 완전한 재현이 가능한 라이브 데모 구동 및 배포를 제한하였습니다. 
모델의 전체 실시간 파이프라인 실행이나 추가적인 시연(Demonstration)이 필요하신 경우, **26-1 DSL 멀티모달 팀**으로 개별 문의해 주시기 바랍니다.

## 6. How to Run (실행 방법)
> 본 프로젝트는 **백엔드(Avatar & RAG)**와 **프론트엔드(React)**를 각각 실행하여 연동하는 구조입니다. 필요한 API Key들이 포함된 `.env.local` 파일이 구성되어 있다는 가정하에 진행됩니다.

### 1) 백엔드 / 라이브킷 에이전트 실행
1. 터미널을 열고 `avatar` 디렉토리로 이동합니다.
   ```bash
   cd 26_1_modeling_project_multimodal/avatar
   ```
2. (최초 1회) 필요한 모델 및 파일을 다운로드합니다.
   ```bash
   uv run python src/agent.py download-files
   ```
3. 에이전트를 Dev 모드로 실행합니다.
   ```bash
   uv run python src/agent.py dev
   ```
   > 💡 **Tip:** 프론트엔드 없이 콘솔로 음성 파이프라인만 빠르게 테스트하려면 `uv run python src/agent.py console`을 실행하세요.

### 2) 프론트엔드 서버 실행
1. 새로운 터미널을 열고 `agent-starter-react` 디렉토리로 이동합니다.
   ```bash
   cd 26_1_modeling_project_multimodal/agent-starter-react
   ```
2. 패키지를 설치합니다.
   ```bash
   pnpm install
   ```
3. 개발 서버를 실행합니다.
   ```bash
   pnpm dev
   ```
4. 브라우저에서 `http://localhost:3000`에 접속하여 **Connect(시작)** 버튼을 클릭해 대화를 시작합니다.

## 7. References
- Dettmers, T., et al. (2023). QLORA: Efficient Finetuning of Quantized LLMs.
- Sellam, T., et al. (2020). BLEURT: Learning Robust Metrics for Text Generation.
- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior.
