import json
import os
import glob
import pickle
from sentence_transformers import SentenceTransformer

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DATA_DIR = os.path.join(BASE_DIR, "data", "rag_datasets")
CACHE_FILE_PATH = os.path.join(BASE_DIR, "rag", "rag_cache_v3.pkl")

def setup_vector_db():
    print("🧠 침착맨 벡터 DB(기억 저장소) 구축 시작... (In-Memory 고속 로딩 방식)")
    
    embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    json_files = glob.glob(os.path.join(JSON_DATA_DIR, "*.json"))
    if not json_files:
        print(f"❌ 데이터를 찾을 수 없습니다: 폴더 {JSON_DATA_DIR}")
        return

    all_qa_list = []
    for json_file in json_files:
        print(f"📄 파일 로드 중: {os.path.basename(json_file)}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                rag_data = json.load(f)
                qa_list = rag_data.get("qa_list", rag_data) if isinstance(rag_data, dict) else rag_data
                all_qa_list.extend(qa_list)
        except Exception as e:
            print(f"오류 발생 ({os.path.basename(json_file)}): {e}")

    print(f"⏳ 총 {len(all_qa_list)}개의 문장을 AI 벡터로 변환하는 중입니다... (잠시 대기)")
    texts = [item["user"] for item in all_qa_list]
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
    
    metadatas = [{
        "assistant": item["assistant"],
        "source": item.get("source_video") or "Unknown"
    } for item in all_qa_list]
    
    ids = [f'{item["id"]}_{idx}' for idx, item in enumerate(all_qa_list)]
    
    # 윈도우 한글 경로 깨짐, 락 등 온갖 오류의 주범인 C++ 폴더 디스크 방식 대신, 
    # 파이썬 고유 피클링으로 완벽하게 저장합니다.
    print(f"💾 데이터를 피클(Pickle) 캐시 파일로 저장하는 중... ({CACHE_FILE_PATH})")
    cache_data = {
        "ids": ids,
        "embeddings": embeddings,
        "documents": texts,
        "metadatas": metadatas
    }
    with open(CACHE_FILE_PATH, "wb") as f:
        pickle.dump(cache_data, f)
        
    print(f"✅ 총 {len(json_files)}개 파일, {len(all_qa_list)}개 데이터가 성공적으로 캐싱되었습니다!")
    print("이제 agent.py를 실행하시면 디스크 오류 없이 곧바로 불러옵니다.")

if __name__ == "__main__":
    setup_vector_db()