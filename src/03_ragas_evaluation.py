"""
Bước 3 — RAGAS Evaluation
===========================
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn, rõ ràng dựa trên context. "
    "Nếu không có thông tin, hãy nói thẳng là không biết."
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])


SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Khi trả lời, hãy: "
    "1) Tóm tắt câu trả lời chính, "
    "2) Trích dẫn nguồn từ context, "
    "3) Nêu rõ mức độ chắc chắn của câu trả lời. "
    "Luôn dựa trên dữ liệu được cung cấp, không suy đoán thêm."
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])


PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}



def setup_vectorstore():
    embeddings = get_embeddings("openai")
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)

    contexts = [
        doc.page_content
        for doc in docs
    ]

    ctx_str = "\n\n".join(contexts)

    answer = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": ctx_str,
        "question": question,
    })

    return {
        "answer": answer,
        "contexts": contexts,
    }


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    llm = get_llm("openai")
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(
            retriever=retriever,
            llm=llm,
            prompt=prompt,
            question=qa["question"],
        )

        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })

        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    return results


def build_ragas_dataset(rag_results: list) -> Dataset:
    return Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["reference"],
        }
        for r in rag_results
    ])


def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(
        f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... "
        "(vui lòng chờ ~5-10 phút)"
    )

    dataset = build_ragas_dataset(rag_results)

    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        raw = result[key]

        if isinstance(raw, (float, int, np.floating)):
            scores[key] = float(raw)
        else:
            scores[key] = float(
                np.mean([
                    v for v in raw
                    if v is not None and not np.isnan(v)
                ])
            )

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

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(
        v1_scores["faithfulness"],
        v2_scores["faithfulness"],
    )

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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()