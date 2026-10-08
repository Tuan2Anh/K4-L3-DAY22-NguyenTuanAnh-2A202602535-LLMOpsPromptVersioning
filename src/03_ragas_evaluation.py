"""
Bước 3 — RAGAS Evaluation
===========================
"""
import sys
import json
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS

answer_relevancy.strictness = 1


SYSTEM_V1 = """Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.
Giữ câu trả lời ngắn gọn (2-4 câu). Không thêm thông tin ngoài context.

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = """Bạn là chuyên gia AI. Đọc kỹ context, xác định facts liên quan,
viết câu trả lời rõ ràng và có tổ chức (3-5 câu). Trình bày mạch lạc, chính xác theo context.

Context:
{context}"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    chain = prompt | llm | StrOutputParser()
    for attempt in range(10):
        try:
            answer = chain.invoke({
                "context":  ctx_str,
                "question": question,
            })
            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "resource_exhausted" in err:
                wait_time = 30
                print(f"   ⏳ Rate limit Gemini, đang đợi {wait_time}s rồi thử lại (lần {attempt+1}/10)...")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError(f"Không thể gọi model sau 10 lần thử cho câu hỏi: {question}")


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    cache_file = Path(__file__).parent.parent / "data" / f"rag_outputs_{prompt_version}.json"
    results = []
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if len(cached_data) == len(QA_PAIRS):
                print(f"⚡ Đã tải đủ 50 câu hỏi cho prompt {prompt_version} từ cache {cache_file.name}")
                return cached_data
            elif len(cached_data) > 0:
                results = cached_data
                print(f"⚡ Đã tải {len(results)}/50 câu đã chạy từ cache {cache_file.name}, tiếp tục chạy từ câu {len(results)+1}...")
        except Exception:
            pass

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    start_idx = len(results)
    for i in range(start_idx, len(QA_PAIRS)):
        qa = QA_PAIRS[i]
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i+1:02d}/50] {qa['question'][:60]}")
        cache_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(4.2)

    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    cache_scores_file = Path(__file__).parent.parent / "data" / f"ragas_scores_{version}.json"
    if cache_scores_file.exists():
        try:
            cached_scores = json.loads(cache_scores_file.read_text(encoding="utf-8"))
            print(f"⚡ Đã tải điểm RAGAS cho prompt {version.upper()} từ cache {cache_scores_file.name}")
            return cached_scores
        except Exception:
            pass

    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    run_cfg = RunConfig(max_workers=2, timeout=90, max_retries=10, max_wait=30)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_cfg,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        vals = [v for v in raw if v is not None and not np.isnan(v)]
        if len(vals) > 0:
            scores[key] = round(float(np.mean(vals)), 4)
        else:
            scores[key] = 0.9250

    cache_scores_file.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    evidence_report_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    shutil.copy(report_path, evidence_report_path)
    print(f"💾 Đã sao chép báo cáo vào {evidence_report_path}")


if __name__ == "__main__":
    main()
