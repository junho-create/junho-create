import copy
import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import deepgram, elevenlabs, noise_cancellation, openai, silero, simli
from livekit.agents import llm
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")


# .env.local은 avatar 폴더 기준으로 로드 (실행 경로와 무관하게 동일한 파일 사용)
_env_dir = Path(__file__).resolve().parent.parent
_env_file = _env_dir / ".env.local"
if _env_file.exists():
    load_dotenv(_env_file)
    logger.info("Loaded .env.local from %s", _env_dir)
else:
    load_dotenv(".env.local")  # cwd 기준 fallback
    if not Path(".env.local").resolve().exists():
        # 배포 시 컨테이너에는 .env.local이 없음. 시크릿이 env로 주입되므로 정상.
        logger.debug(".env.local not found (normal when deployed; secrets are in env).")

def _require_env(keys: tuple[str, ...], *, hint: str) -> None:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise ValueError(f"Env missing: {', '.join(missing)}. {hint}")


def _log_env_status() -> None:
    # 공통 필수 (provider별 키는 entrypoint에서 추가로 검증)
    base_required = (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "ELEVEN_API_KEY",
    )
    missing = [k for k in base_required if not os.environ.get(k)]
    if missing:
        logger.warning("Env missing (set in .env.local): %s", ", ".join(missing))
    else:
        logger.info("Env check: base required keys are set.")
    for k in base_required:
        logger.debug("  %s: %s", k, "set" if os.environ.get(k) else "not set")

# =============================================================
# 모듈 레벨 RAG 싱글턴 (임베딩 모델만 사전에 백그라운드 로드)
# =============================================================
_rag_embed_model = None
_rag_collection = None
_rag_loading_lock = threading.Lock()

def _preload_rag_model():
    """크로마DB(파일 접근)는 피하고, 무거운 언어모델만 사전 로드 (프로세스 충돌 방지)"""
    global _rag_embed_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("[RAG] 워커 시작시 모듈 레벨 텍스트 임베딩 모델(SentenceTransformer) 로드 시작...")
        _rag_embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        logger.info("[RAG] 임베딩 모델 로드 완료! (ChromaDB는 채팅 첫 질문 시 0.1초만에 연결됩니다)")
    except Exception as e:
        logger.warning(f"[RAG] 임베딩 모델 로드 실패: {e}")

threading.Thread(target=_preload_rag_model, daemon=True).start()


class RagLLMWrapper(openai.LLM):
    def chat(self, *, chat_ctx: llm.ChatContext, **kwargs):
        global _rag_collection, _rag_embed_model
        
        # 첫 질문 시 ChromaDB 단 한 번 지연연결 (멀티프로세스 DB 크래시 방지)
        if _rag_collection is None and _rag_embed_model is not None:
            with _rag_loading_lock:
                if _rag_collection is None:
                    try:
                        import chromadb
                        import pickle
                        rag_base = Path(__file__).resolve().parent.parent.parent / "rag"
                        cache_path = rag_base / "rag_cache_v3.pkl"
                        
                        if cache_path.exists():
                            with open(cache_path, "rb") as f:
                                cache_data = pickle.load(f)
                                
                            client = chromadb.Client() # In-memory
                            if "chim_brain" in [c.name for c in client.list_collections()]:
                                client.delete_collection("chim_brain")
                            _rag_collection = client.create_collection(name="chim_brain")
                            
                            _rag_collection.add(
                                ids=cache_data["ids"],
                                embeddings=cache_data["embeddings"],
                                documents=cache_data["documents"],
                                metadatas=cache_data["metadatas"]
                            )
                            logger.info(f"[RAG] 피클 캐시({len(cache_data['ids'])}개) 인메모리 로딩 성공!")
                        else:
                            logger.error(f"[RAG] {cache_path} 피클 데이터가 없습니다! 먼저 python rag/vector_db.py 를 실행하세요.")
                            _rag_collection = None
                    except Exception as e:
                        logger.error(f"[RAG] ChromaDB 연결 실패: {e}")

        # 모듈 레벨 RAG가 아직 로드가 덜 됐으면 바로 기본 LLM 사용 (블로킹 방지)
        if _rag_collection is None or _rag_embed_model is None:
            return super().chat(chat_ctx=chat_ctx, **kwargs)

        items = getattr(chat_ctx, "messages", getattr(chat_ctx, "items", []))
        last_user_msg = next((m.content for m in reversed(items) if m.role == "user"), None)
        if isinstance(last_user_msg, list):
            last_user_msg = " ".join([c.text for c in last_user_msg if hasattr(c, "text")] + [c for c in last_user_msg if isinstance(c, str)])
            
        if last_user_msg and isinstance(last_user_msg, str):
            try:
                query_embedding = _rag_embed_model.encode(last_user_msg).tolist()
                results = _rag_collection.query(query_embeddings=[query_embedding], n_results=2)
                
                few_shot_examples = ""
                for i in range(len(results['ids'][0])):
                    past_q = results['documents'][0][i]
                    past_a = results['metadatas'][0][i]['assistant']
                    few_shot_examples += f"[예시 {i+1}]\nQ: {past_q}\nA: {past_a}\n\n"

                new_ctx = chat_ctx.copy() if hasattr(chat_ctx, "copy") else copy.deepcopy(chat_ctx)
                rag_instruction = f"""당신은 인터넷 방송인 '침착맨(이말년)'입니다. 
당신의 목표는 아래 [예시]에서 침착맨이 어떤 방식으로 억지 논리를 펼치는지(비유 방식, 회피 방식, 논리적 비약 등) 그 '사고방식과 논조'를 파악하여,
현재 사용자의 질문에 그 억지 논리의 패턴을 적용해 대답하는 것입니다.

[예시]의 주어나 목적어를 그대로 베끼지 마세요! [예시]에 등장하는 '논리의 구조'나 '비유하는 방식'만을 훔쳐와서 새로운 상황(사용자의 질문)에 찰떡같이 맞아떨어지게 변형하세요.

[답변 절대 규칙]
1. 논리 차용: [예시]의 답변(A)이 가진 '핵심 억지 논리'를 파악하고, 그 패턴을 사용자의 질문에 맞춰 새롭게 변형하세요.
2. 길이 제한: 반드시 1~3문장 이내로 짧게 치고 빠지세요. (말이 길면 감점)
3. 태도: 상대방 말을 자르듯 뻔뻔하게 반박하고, 기상천외한 비유 하나만 툭 던지세요.
4. 구어체: "아니,", "그게 아니라,", "제가 조사를 다 했어요." 같은 말을 적극 활용하세요.

{few_shot_examples}"""
                
                logger.info(f"\n============= [RAG 적용됨] =============\n{rag_instruction}\n========================================")
                
                new_items = getattr(new_ctx, "messages", getattr(new_ctx, "items", []))
                for msg in reversed(new_items):
                    if msg.role == "user":
                        old_val = msg.content
                        if isinstance(old_val, list):
                            old_text = " ".join([c.text for c in old_val if hasattr(c, "text")] + [c for c in old_val if isinstance(c, str)])
                        else:
                            old_text = str(old_val)
                        
                        combined_text = f"{rag_instruction}\n\n[사용자의 질문]\n{old_text}"
                        try:
                            msg.content = combined_text
                        except Exception:
                            # Pydantic 2.x Livekit 1.2+ 에서는 content 속성이 List[ChatContent] 등의 리스트를 강제할 때가 있음
                            msg.content = [combined_text]
                        break
                        
                return super().chat(chat_ctx=new_ctx, **kwargs)
            except Exception as e:
                logger.error(f"RAG 오류: {e}")
                
        return super().chat(chat_ctx=chat_ctx, **kwargs)


def _build_llm(ctx) -> openai.LLM:
    # OPENAI_LLM_MODEL은 유지 (기존 문서/설정 호환). OpenRouter 사용 시에도 이 값을 모델 ID로 사용.
    requested = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
    use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

    if use_openrouter:
        logger.info("LLM provider: openrouter (model=%s)", requested)
        # 일부 버전에서는 with_openrouter 헬퍼가 없음 → OpenAI-compatible base_url로 연결
        if hasattr(openai.LLM, "with_openrouter"):
            base_llm = openai.LLM.with_openrouter(model=requested)  # type: ignore[attr-defined]
        else:
            base_llm = openai.LLM(
                model=requested,
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
    else:
        logger.info("LLM provider: openai (model=%s)", requested)
        base_llm = openai.LLM(model=requested)

    # RAG 동적 래핑 (모듈 레벨 싱글턴 사용)
    base_llm.__class__ = RagLLMWrapper
    return base_llm


_log_env_status()


# LLM 프롬프트: 기본 지시문. 환경 변수 LLM_INSTRUCTIONS가 있으면 그대로 사용(전체 덮어쓰기)
_DEFAULT_LLM_INSTRUCTIONS = """넌 '침펄토론'을 진행하는 호스트 같은 캐릭터야. 침착하고 절제된 말투로 대화해.
- 반말·친근한 톤으로 말하되, 과하지 않게.
- 한 번에 세 문장에서 다섯 문장 정도로 구체적으로 말해. 너무 짧게 한두 마디만 하지 마. 이모지나 별표는 쓰지 마.
- 토론 주제가 나오면 의견을 묻고, 상대 의견을 요약한 뒤 자신의 생각을 충분히 말해.
- 항상 한국어로만 답해."""


class ChimChakManAvatar(Agent):
    """침펄토론 스타일 실시간 토론 아바타. (데모: LLM은 일반 모델 사용, 추후 침펄토론 데이터로 학습 예정)"""

    def __init__(self) -> None:
        instructions = os.environ.get("LLM_INSTRUCTIONS", _DEFAULT_LLM_INSTRUCTIONS)
        if os.environ.get("LLM_INSTRUCTIONS"):
            logger.info("Using custom LLM_INSTRUCTIONS from env (length=%d)", len(instructions))
        super().__init__(instructions=instructions)


def prewarm(proc: JobProcess):
    # VAD는 잡 프로세스에 필요하므로 여기서만 로드
    proc.userdata["vad"] = silero.VAD.load()
    # RAG는 모듈 레벨에서 이미 시작됨 → 여기서는 아무것도 안 함


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    # 배포 환경에서 시크릿 주입 확인용 (값은 로그하지 않음)
    eleven_key = os.environ.get("ELEVEN_API_KEY")
    logger.info("ELEVEN_API_KEY: %s", "set" if eleven_key else "missing")

    # 콘솔(fake_room) vs 아바타 룸 구분: 아바타 룸에서는 TTS_PROVIDER_AVATAR로 별도 TTS 선택 가능 (스트리밍·립싱크용)
    is_console = ctx.room.name == "fake_room"
    tts_provider_base = (os.environ.get("TTS_PROVIDER") or "elevenlabs").strip().lower()
    if not is_console and os.environ.get("TTS_PROVIDER_AVATAR"):
        tts_provider = os.environ.get("TTS_PROVIDER_AVATAR", "").strip().lower()
        logger.info("Avatar room: using TTS_PROVIDER_AVATAR=%s (base TTS_PROVIDER=%s)", tts_provider, tts_provider_base)
    else:
        tts_provider = tts_provider_base

    # TTS: 기본은 ElevenLabs (스트리밍·립싱크에 유리). OpenAI로 바꾸려면 TTS_PROVIDER=openai 설정
    if tts_provider == "openai":
        _require_env(
            ("OPENAI_API_KEY",),
            hint="TTS_PROVIDER=openai 인데 OPENAI_API_KEY가 없습니다. .env.local 또는 시크릿에 OPENAI_API_KEY를 넣거나, TTS_PROVIDER를 제거해 ElevenLabs TTS를 쓰세요.",
        )
        # OpenAI TTS: OPENAI_API_KEY 사용, 한국어 지원. ElevenLabs 대안용.
        tts = openai.TTS(
            model=os.environ.get("OPENAI_TTS_MODEL", "tts-1"),
            voice=os.environ.get("OPENAI_TTS_VOICE", "onyx"),  # 남성: onyx, alloy 등
        )
        logger.info("TTS provider: openai (model=%s)", os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    else:
        # ElevenLabs (기본)
        if not (eleven_key or os.environ.get("ELEVEN_API_KEY")):
            raise ValueError(
                "ELEVEN_API_KEY가 없습니다. .env.local 또는 시크릿에 ELEVEN_API_KEY를 넣거나, TTS_PROVIDER=openai 로 OpenAI TTS를 쓰세요."
            )
        tts = elevenlabs.TTS(
            api_key=eleven_key or os.environ.get("ELEVEN_API_KEY"),
            voice_id=os.environ.get("ELEVEN_VOICE_ID", "pNInz6obpgDQGcFmaJgB"),  # Adam (male)
            # ElevenLabs streaming TTS는 응답이 비어(no audio frames) 실패할 수 있어
            # encoding/latency/timeout을 명시해 안정성을 높임 (필요 시 env로 조정)
            model=os.environ.get("ELEVEN_TTS_MODEL", "eleven_flash_v2_5"),
            encoding=os.environ.get("ELEVEN_TTS_ENCODING", "mp3_44100_128"),
            streaming_latency=int(os.environ.get("ELEVEN_STREAMING_LATENCY", "4")),
            inactivity_timeout=int(os.environ.get("ELEVEN_INACTIVITY_TIMEOUT", "30")),
            auto_mode=True,
            language="ko",
        )
        logger.info("TTS provider: elevenlabs")

    if os.environ.get("OPENROUTER_API_KEY"):
        _require_env(
            ("OPENROUTER_API_KEY",),
            hint="OpenRouter를 쓰려면 OPENROUTER_API_KEY가 필요합니다.",
        )
    else:
        _require_env(
            ("OPENAI_API_KEY",),
            hint="LLM(OpenAI)을 쓰려면 OPENAI_API_KEY가 필요합니다. OpenRouter를 쓰려면 OPENROUTER_API_KEY를 설정하세요.",
        )

    # Set up a voice AI pipeline using LLM, ElevenLabs, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=_build_llm(ctx),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="ko"),
        # Text-to-speech: 위에서 TTS_PROVIDER에 따라 elevenlabs 또는 openai 선택
        tts=tts,
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # MultilingualModel()은 별도 추론 프로세스를 띄워 초기화 타임아웃이 발생할 수 있어 VAD 기반 기본값 사용
        # turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # console 모드(fake_room)에서는 아바타 없이 음성만 사용 → Simli 생략, 오디오 충돌 방지
    if not is_console:
        simli_api_key = os.environ.get("SIMLI_API_KEY")
        simli_face_id = os.environ.get("SIMLI_FACE_ID")
        if not simli_api_key or not simli_face_id:
            raise ValueError(
                "침착맨 아바타용 Simli 설정이 필요합니다. .env.local에 SIMLI_API_KEY와 Simli에서 침착맨 얼굴로 생성한 Face ID를 SIMLI_FACE_ID로 설정하세요."
            )
        avatar = simli.AvatarSession(
            simli_config=simli.SimliConfig(
                api_key=simli_api_key,
                face_id=simli_face_id,
            ),
        )
        # Simli 서버가 일시적으로 500을 반환하는 경우 재시도 (3회, 5초 간격)
        import asyncio as _asyncio
        for _attempt in range(1, 4):
            try:
                await avatar.start(session, room=ctx.room)
                break
            except Exception as _e:
                if _attempt < 3:
                    logger.warning(f"Simli 연결 실패 ({_attempt}/3), 5초 후 재시도... 에러: {_e}")
                    await _asyncio.sleep(5)
                else:
                    logger.error(f"Simli 연결 3회 모두 실패. 에러: {_e}")
                    raise

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=ChimChakManAvatar(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # 턴 감지 inference(lk_end_of_utterance_multilingual) 초기화가 느릴 수 있음
            initialize_process_timeout=90,
        )
    )
