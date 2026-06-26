"""
Bước 2 — Prompt Hub & A/B Routing
===================================
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


PROMPT_V1_NAME = "chi-nguyen-rag-prompt-v1"
PROMPT_V2_NAME = "chi-nguyen-rag-prompt-v2"


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


def push_prompts_to_hub(client: Client):
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 – ngắn gọn",
        )
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 – có cấu trúc",
        )
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


def pull_prompts_from_hub(client: Client) -> dict:
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}'")

    return prompts


def get_prompt_version(request_id: str) -> str:
    hash_int = int(
        hashlib.md5(request_id.encode()).hexdigest(),
        16,
    )

    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(
    name="ab-rag-query",
    tags=["ab-test", "step2"],
)
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    answer = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "question": question,
        "answer": answer,
        "version": version,
    }


def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(
        api_key=config.LANGSMITH_API_KEY
    )

    push_prompts_to_hub(client)

    prompts = pull_prompts_from_hub(client)

    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    llm = get_llm()

    v1_count, v2_count = 0, 0

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        result = ask_ab(
            retriever=retriever,
            llm=llm,
            prompt=prompt,
            question=question,
            version=version_tag,
        )

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()