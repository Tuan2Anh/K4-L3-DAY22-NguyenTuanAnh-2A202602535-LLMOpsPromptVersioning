"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
import time
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    from langchain_community.vectorstores import FAISS

    index_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"
    if index_dir.exists() and (index_dir / "index.faiss").exists():
        print("⚡ Tải FAISS vectorstore từ cache local...")
        try:
            return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Không thể tải cache, tạo lại: {e}")

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    batch_size = 50
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   → Đang embed chunk {i + 1} đến {min(i + batch_size, len(chunks))}...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch, embeddings)
                else:
                    vectorstore.add_texts(batch)
                break
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'resource_exhausted' in err_str:
                    wait_sec = 35
                    print(f"   ⏳ Chạm giới hạn quota Gemini (100 req/phút). Đang đợi {wait_sec}s rồi thử lại...")
                    time.sleep(wait_sec)
                else:
                    raise e

        if i + batch_size < len(chunks):
            # Giữ an toàn để không chạm quota 100 req/min
            print("   ⏳ Nghỉ 10s trước batch tiếp theo...")
            time.sleep(10)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print("✅ FAISS vectorstore đã sẵn sàng và lưu cache tại data/faiss_index.")
    return vectorstore
