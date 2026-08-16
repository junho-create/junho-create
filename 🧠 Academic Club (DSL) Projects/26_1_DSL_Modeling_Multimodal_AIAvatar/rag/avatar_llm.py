import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ================= 설정 =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 현재 파일(avatar_llm.py)이 있는 rag 폴더 안의 DB 경로를 자동으로 잡습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_STORE_PATH = os.path.join(BASE_DIR, "chroma_db_store")
# =======================================

class ChimchakAvatar:
    def __init__(self, model_name="google/gemini-2.5-flash"):
        print("⚙️ [시스템] 침착맨 아바타 엔진 초기화 중...")
        self.model_name = model_name
        
        # 1. 한국어 임베딩 모델 로드 (질문의 의도를 파악하는 용도)
        self.embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 2. 로컬 ChromaDB 연결 (과거 억지 논리 저장소)
        try:
            self.chroma_client = chromadb.PersistentClient(path=DB_STORE_PATH)
            self.collection = self.chroma_client.get_collection(name="chim_brain")
            print("✅ [시스템] Vector DB (chim_brain) 연결 성공!")
        except ValueError:
            print(f"❌ [오류] Vector DB를 찾을 수 없습니다. 먼저 'vector_db.py'를 실행해주세요!\n경로: {DB_STORE_PATH}")
            exit(1)
            
        # 3. OpenRouter (LLM) 세팅
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        print("✅ [시스템] LLM API 연결 완료! 토론 준비 끝.\n")
    def generate_response_stream(self, user_message: str):
        """Simli 연동 및 실시간 출력을 위한 고속 스트리밍 함수"""
        
        # 1. 과거 억지 논리 검색
        query_embedding = self.embed_model.encode(user_message).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        few_shot_examples = ""
        for i in range(len(results['ids'][0])):
            past_q = results['documents'][0][i]
            past_a = results['metadatas'][0][i]['assistant']
            few_shot_examples += f"[예시 {i+1}]\nQ: {past_q}\nA: {past_a}\n\n"

        # 2. ⚡ '실시간 대화용' 숏폼 프롬프트 적용
        system_prompt = f"""당신은 인터넷 방송인 '침착맨(이말년)'입니다. 
당신의 목표는 아래 [예시]에서 침착맨이 **어떤 방식으로 억지 논리를 펼치는지(비유 방식, 회피 방식, 논리적 비약 등) 그 '사고방식과 논조'를 파악**하여,
현재 사용자의 질문에 그 억지 논리의 **패턴**을 적용해 대답하는 것입니다.

[예시]의 주어나 목적어를 그대로 베끼지 마세요! [예시]에 등장하는 '논리의 구조'나 '비유하는 방식'만을 훔쳐와서 새로운 상황(사용자의 질문)에 찰떡같이 맞아떨어지게 변형하세요.

[답변 절대 규칙]
1. 논리 차용: [예시]의 답변(A)이 가진 '핵심 억지 논리'를 파악하고, 그 패턴을 사용자의 질문에 맞춰 새롭게 변형하세요.
2. 길이 제한: 반드시 **1~3문장 이내**로 짧게 치고 빠지세요. (말이 길면 감점)
3. 태도: 상대방 말을 자르듯 뻔뻔하게 반박하고, 기상천외한 비유 하나만 툭 던지세요.
4. 구어체: "아니,", "그게 아니라,", "제가 조사를 다 했어요." 같은 말을 적극 활용하세요.

{few_shot_examples}"""

        # 3. ⚡ 스트리밍(stream=True)으로 API 호출
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                # 💡 논리 패턴을 파악하고 새로운 맥락에 적용할 수 있도록 flexibility(temperature) 부여
                temperature=0.7,
                max_tokens=150,
                stream=True
            )
            
            # 💡 여기를 수정했습니다! (빈 상자 방어 로직)
            for chunk in response:
                # choices 리스트가 존재하고, 비어있지 않은지 먼저 확인!
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                        
        except Exception as e:
            yield f" 오류 발생: {e}"


# =====================================================================
# 터미널 테스트 실행부
# =====================================================================
if __name__ == "__main__":
    # 빠른 반응속도를 원하시면 flash 모델이나 gpt-4o-mini 같은 가벼운 모델이 좋습니다.
    avatar = ChimchakAvatar(model_name="google/gemini-2.5-flash")
    
    print("==================================================")
    print("⚡ 침착맨 RAG 고속 스트리밍 모드 가동")
    print("==================================================\n")
    
    while True:
        user_input = input("👤 당신의 공격: ")
        
        if user_input.strip().lower() in ['q', 'quit', 'exit']:
            break
        if not user_input.strip():
            continue
            
        print("😎 침착맨: ", end="", flush=True)
        
        # ⚡ yield로 반환되는 단어들을 실시간으로 화면에 찍어줍니다.
        full_answer = ""
        for word_chunk in avatar.generate_response_stream(user_input):
            print(word_chunk, end="", flush=True)
            full_answer += word_chunk
            
        print("\n" + "-" * 50)