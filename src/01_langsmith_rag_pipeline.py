import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")
    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    for attempt in range(5):
        try:
            return chain.invoke(question)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "resource_exhausted" in err:
                print("   ⏳ Rate limit Gemini, đang đợi 20s...")
                time.sleep(20)
            else:
                raise e
    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")
        time.sleep(4.2)

    print(f"\n✅ {len(SAMPLE_QUESTIONS)} traces đã gửi lên LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
