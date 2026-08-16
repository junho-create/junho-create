import json, os
from rag_doctor_agent.data.pipeline import build as pipeline_build
from rag_doctor_agent.agent.graph import build_and_run_agent

def load_sample(path="tests/sample_back_pain.json"): #이 경로 수정하면 다른 파일 테스트 가능
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Build (init + prepare + index). Requires OPENAI_API_KEY for embeddings.
    # If you already prepared/indexed once, you can comment this out to speed up.
    # pipeline_build()
    input_json = load_sample()
    result = build_and_run_agent(input_json)
    os.makedirs("out", exist_ok=True)
    with open("out/result.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2))
    print(json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
