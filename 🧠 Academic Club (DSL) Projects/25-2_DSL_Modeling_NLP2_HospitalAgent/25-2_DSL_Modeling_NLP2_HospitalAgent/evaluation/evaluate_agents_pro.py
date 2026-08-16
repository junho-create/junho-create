
import os, json, re, warnings
from datetime import datetime
from statistics import mean
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# =========================
# 환경 설정
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    warnings.warn("OPENAI_API_KEY not set; LLM metrics may fail.")

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False
TS_FMT = "%Y%m%d_%H%M%S_%f"

# =========================
# DeepEval Import
# =========================
try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_OK = True
except Exception as e:
    DEEPEVAL_OK = False
    warnings.warn(f"DeepEval import failed: {e}")

# =========================
# 유틸리티
# =========================
def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_timestamp(ts: str):
    try:
        return datetime.strptime(ts, TS_FMT)
    except Exception:
        return None

def llm_score(prompt: str, model="gpt-4.1"):
    if not OPENAI_API_KEY:
        return 3.0
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only a number between 0 and 5."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"([0-5](?:\.\d+)?)", text)
        return float(m.group(1)) if m else 3.0
    except Exception:
        return 3.0

# =========================
# Metrics
# =========================
def routing_accuracy(agent1_logs):
    expected = {
        "reservation": "agent2_reservation",
        "symptom_doctor": "agent3_rag",
        "hospital_info": "tavily_search",
        "greeting": "agent1_direct",
    }
    tot, ok = 0, 0
    for rec in agent1_logs:
        intent = safe_get(rec, "output", "primary_intent")
        target = safe_get(rec, "output", "routing_info", "target")
        if intent:
            tot += 1
            if expected.get(intent) == target:
                ok += 1
    return (ok / tot if tot else 0), ok, tot

def supervisor_coherence(agent1_logs, sample=10):
    picks = agent1_logs[:sample]
    scores = []
    for rec in picks:
        prompt = f"Rate routing coherence (0~5).\nUser: {rec.get('user_input')}\nOutput: {rec.get('output')}"
        scores.append(llm_score(prompt))
    return mean(scores) if scores else 0.0, len(picks)

REQ_FIELDS = ["환자명", "전화번호", "symptoms"]

def form_fill_accuracy(agent2_logs):
    scores = []
    for rec in agent2_logs:
        info = safe_get(rec, "output", "collected_info", default={})
        if isinstance(info, dict):
            filled = sum(1 for k in REQ_FIELDS if info.get(k))
            scores.append(filled / len(REQ_FIELDS))
    return mean(scores) if scores else 0.0, len(scores)

def parsing_accuracy(agent2_logs):
    tot, ok = 0, 0
    for rec in agent2_logs:
        out = rec.get("output", {})
        tot += 1
        if isinstance(out.get("success"), bool) and isinstance(out.get("routing_info"), dict):
            ok += 1
    return (ok / tot if tot else 0), tot

# =========================
# RAG DeepEval
# =========================
def build_rag_samples(agent3_logs, limit=20):
    """RAG 로그에서 DeepEval용 샘플 생성 (모든 context를 문자열로 정제)"""
    data = []
    for log in agent3_logs[:limit]:
        q = str(log.get("user_input", "")).strip()
        out = log.get("output", {})
        recommended_doctors = out.get("recommended_doctors", [])
        doctor_names = [
            str(doc.get("name", ""))
            for doc in recommended_doctors
            if isinstance(doc, dict) and doc.get("name")
        ]
        a = f"추천 드리는 의료진: {', '.join(doctor_names)}" if doctor_names else str(out.get("answer", ""))[:300]
        ctx = []
        for doc in recommended_doctors:
            if isinstance(doc, dict) and doc.get("reasoning"):
                ctx.append(str(doc["reasoning"])[:500])
        if not ctx and out.get("reasoning"):
            ctx = [str(out.get("reasoning"))[:500]]
        ctx = [str(c) for c in ctx if isinstance(c, (str, int, float, bool)) and str(c).strip()]
        if q and a and ctx:
            data.append({"question": q, "answer": a, "contexts": ctx})
    return data

import io
import contextlib

def eval_with_deepeval(rag_samples):
    """DeepEval 점수 계산 (summary + stdout pass rate 모두 대응)"""
    if not DEEPEVAL_OK or not rag_samples:
        return {"FaithfulnessMetric": 0.0, "AnswerRelevancyMetric": 0.0, "samples": len(rag_samples)}

    try:
        cases = [
            LLMTestCase(
                input=s["question"],
                actual_output=s["answer"],
                retrieval_context=s["contexts"]
            )
            for s in rag_samples
        ]

        metrics = [
            FaithfulnessMetric(model="gpt-3.5-turbo", async_mode=False),
            AnswerRelevancyMetric(model="gpt-4.1", async_mode=False),
        ]

        # ⚙️ stdout 캡처
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            output = deepeval_evaluate(test_cases=cases, metrics=metrics)
        printed_output = buffer.getvalue()

        # ⚙️ DeepEval 리턴 결과 처리
        results, summary = (output if isinstance(output, tuple) and len(output) == 2 else (output, None))
        faith, rel = None, None

        # 1️⃣ summary dict에서 점수 찾기
        if isinstance(summary, dict):
            metrics_summary = summary.get("metrics_summary", {}) or summary.get("metrics", {})
            for k, v in metrics_summary.items():
                if "faith" in k.lower():
                    faith = float(v)
                elif "relev" in k.lower():
                    rel = float(v)

        # 2️⃣ summary가 비었으면 stdout에서 pass rate 추출
        if (faith is None or rel is None) and printed_output:
            match_f = re.search(r"Faithfulness:\s*([\d.]+)%\s*pass rate", printed_output)
            match_r = re.search(r"Answer Relevancy:\s*([\d.]+)%\s*pass rate", printed_output)
            if match_f:
                faith = float(match_f.group(1)) / 100
            if match_r:
                rel = float(match_r.group(1)) / 100

        # 3️⃣ 그래도 없으면 results에서 평균 계산
        if faith is None or rel is None:
            scores = {"FaithfulnessMetric": [], "AnswerRelevancyMetric": []}
            for r in results or []:
                metrics_list = getattr(r, "metrics", None)
                if metrics_list is None and isinstance(r, dict):
                    metrics_list = r.get("metrics", [])
                for m in metrics_list or []:
                    name = getattr(m, "name", None)
                    score = getattr(m, "score", None)
                    if name and isinstance(score, (float, int)):
                        scores[name].append(score)
            if faith is None:
                faith = mean(scores["FaithfulnessMetric"]) if scores["FaithfulnessMetric"] else 0.0
            if rel is None:
                rel = mean(scores["AnswerRelevancyMetric"]) if scores["AnswerRelevancyMetric"] else 0.0

        return {
            "FaithfulnessMetric": round(faith or 0.0, 4),
            "AnswerRelevancyMetric": round(rel or 0.0, 4),
            "samples": len(rag_samples),
        }

    except Exception as e:
        warnings.warn(f"DeepEval evaluation failed: {e}")
        return {"FaithfulnessMetric": 0.0, "AnswerRelevancyMetric": 0.0, "samples": len(rag_samples)}


def comm_latency(all_logs):
    ts = [parse_timestamp(l.get("timestamp", "")) for l in all_logs]
    ts = [t for t in ts if t]
    if len(ts) < 2:
        return 0, 0
    diffs = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    return mean(diffs), len(diffs)

# =========================
# 메인
# =========================
def run_eval(input_path, out_dir="evaluation/reports"):
    os.makedirs(out_dir, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    agents = {"supervisor": [], "reservation": [], "rag": []}
    for rec in logs:
        nm = rec.get("agent", "").lower()
        if "agent1" in nm:
            agents["supervisor"].append(rec)
        elif "agent2" in nm:
            agents["reservation"].append(rec)
        elif "agent3" in nm:
            agents["rag"].append(rec)

    # ---- Metrics ----
    sup_acc, sup_ok, sup_tot = routing_accuracy(agents["supervisor"])
    sup_coh, coh_n = supervisor_coherence(agents["supervisor"])
    form_acc, form_n = form_fill_accuracy(agents["reservation"])
    parse_acc, parse_n = parsing_accuracy(agents["reservation"])
    rag_samples = build_rag_samples(agents["rag"], 20)
    deepeval_res = eval_with_deepeval(rag_samples)
    lat_mean, lat_count = comm_latency(logs)

    # ---- Summary ----
    overall_coh = (sup_coh / 5 + ((deepeval_res.get("FaithfulnessMetric", 0) + deepeval_res.get("AnswerRelevancyMetric", 0)) / 2)) / 2

    summary = {
        "meta": {
            "input_file": input_path,
            "total_samples": len(logs),
            "samples_per_agent": {
                "supervisor": len(agents["supervisor"]),
                "reservation": len(agents["reservation"]),
                "rag": len(agents["rag"]),
            },
        },
        "supervisor": {
            "routing_accuracy": sup_acc,
            "routing_ok": sup_ok,
            "total": sup_tot,
            "coherence_llm": sup_coh,
            "coherence_samples": coh_n,
        },
        "reservation": {
            "form_fill_accuracy": form_acc,
            "form_fill_count": form_n,
            "parsing_accuracy": parse_acc,
            "parsing_total": parse_n,
        },
        "deepeval": deepeval_res,
        "overall": {
            "A1 RoutingAcc": sup_acc,
            "A1 Coherence": sup_coh / 5,
            "A2 FormFill": form_acc,
            "A2 Parsing": parse_acc,
            "A3 Coherence": sup_coh / 5,
            "Overall Coh.": overall_coh,
            "samples": {
                "A1 RoutingAcc": sup_tot,
                "A1 Coherence": coh_n,
                "A2 FormFill": form_n,
                "A2 Parsing": parse_n,
                "A3 Coherence": deepeval_res.get("samples", 0),
                "A3 Faith/Rel.": deepeval_res.get("samples", 0),
                "Overall Coh.": coh_n + deepeval_res.get("samples", 0),
            },
        },
        "collaboration": {
            "communication_latency_mean_sec": lat_mean,
            "communication_counts": lat_count,
        },
    }

    # ---- 저장 ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"summary_final_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON → {out_json}")

    # ---- 시각화 ----
    metrics = {
        "A1 RoutingAcc": summary["overall"]["A1 RoutingAcc"],
        "A1 Coherence": summary["overall"]["A1 Coherence"],
        "A2 FormFill": summary["overall"]["A2 FormFill"],
        "A2 Parsing": summary["overall"]["A2 Parsing"],
        "A3 Faith.": deepeval_res.get("FaithfulnessMetric", 0),
        "A3 Rel.": deepeval_res.get("AnswerRelevancyMetric", 0),
        "Overall Coh.": summary["overall"]["Overall Coh."],
    }

    samples = {
        "A1 RoutingAcc": summary["overall"]["samples"]["A1 RoutingAcc"],
        "A1 Coherence": summary["overall"]["samples"]["A1 Coherence"],
        "A2 FormFill": summary["overall"]["samples"]["A2 FormFill"],
        "A2 Parsing": summary["overall"]["samples"]["A2 Parsing"],
        "A3 Faith.": deepeval_res.get("samples", 0),
        "A3 Rel.": deepeval_res.get("samples", 0),
        "Overall Coh.": summary["overall"]["samples"]["Overall Coh."],
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color="#4A90E2", edgecolor="black")

    plt.title("Agent Evaluation Summary", fontsize=15, pad=35)
    plt.ylabel("Score (0~1)", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for i, (k, v) in enumerate(metrics.items()):
        n = samples.get(k, 0)
        plt.text(
            i,
            v + 0.000003,
            f"{v:.2f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            fontweight="semibold"
        )

    plot_path = os.path.join(out_dir, f"summary_plot_{ts}.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"📊 Saved Plot → {plot_path}")


if __name__ == "__main__":
    INPUT = os.getenv(
        "COMBINED_LOGS_PATH",
        r"C:\Users\asap0\OneDrive\바탕 화면\연세대학교\25-2 DSL\25-2_DSL_Modeling_NLP2_HospitalAgent\evaluation\combined_logs.json"
    )
    run_eval(INPUT)
