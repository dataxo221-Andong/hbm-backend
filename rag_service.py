# RAG: PDF 문서 수집 → 청킹 → 임베딩 → 검색
# 기존 챗봇 기능과 분리되어, "engineer_doc" 의도일 때만 사용됩니다.

import os
import re
from pathlib import Path

_CHROMA = None
_EMBEDDING_MODEL = None
_COLLECTION_NAME = "engineer_doc"
_CHUNK_SIZE = 700
_CHUNK_OVERLAP = 150
_TOP_K = 5


def _get_pdf_path():
    """Engineer PDF 경로. 환경변수 또는 프로젝트 data 폴더."""
    path = os.environ.get("ENGINEER_PDF_PATH", "").strip()
    if path and Path(path).is_file():
        return path
    base = Path(__file__).resolve().parent
    for name in ("Engineer.pdf", "data/Engineer.pdf"):
        p = base / name
        if p.is_file():
            return str(p)
    return ""


def _extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출 (pypdf)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf가 필요합니다. pip install pypdf")
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t and t.strip():
                parts.append(t.strip())
        except Exception:
            continue
    return "\n\n".join(parts) if parts else ""


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list:
    """텍스트를 겹치는 청크로 분할."""
    if not text or not text.strip():
        return []
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk.strip():
            start = end - overlap
            continue
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def _get_embedding_model():
    """sentence-transformers 멀티링구얼 모델 (한국어 지원)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return _EMBEDDING_MODEL
    except Exception as e:
        print(f"[RAG] 임베딩 모델 로드 실패: {e}")
        return None


def _get_chroma_client():
    """ChromaDB 클라이언트 (영구 저장)."""
    global _CHROMA
    if _CHROMA is not None:
        return _CHROMA
    try:
        import chromadb
        from chromadb.config import Settings
        persist_dir = Path(__file__).resolve().parent / "data" / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        _CHROMA = chromadb.PersistentClient(
            path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        return _CHROMA
    except Exception as e:
        print(f"[RAG] ChromaDB 초기화 실패: {e}")
        return None


def _embed(texts: list) -> list:
    """텍스트 리스트를 벡터로 변환."""
    model = _get_embedding_model()
    if model is None:
        return []
    try:
        return model.encode(texts, show_progress_bar=False).tolist()
    except Exception as e:
        print(f"[RAG] 임베딩 오류: {e}")
        return []


def ingest_pdf(pdf_path: str) -> bool:
    """PDF를 청킹 후 ChromaDB에 저장. 성공 시 True."""
    if not pdf_path or not Path(pdf_path).is_file():
        return False
    try:
        text = _extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 50:
            print("[RAG] PDF에서 추출된 텍스트가 없거나 너무 짧습니다.")
            return False
        chunks = _chunk_text(text)
        if not chunks:
            return False
        client = _get_chroma_client()
        if client is None:
            return False
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            _COLLECTION_NAME, metadata={"description": "Engineer PDF RAG"}
        )
        embeddings = _embed(chunks)
        if len(embeddings) != len(chunks):
            return False
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks)
        print(f"[RAG] PDF 수집 완료: {len(chunks)}개 청크 (경로: {pdf_path})")
        return True
    except Exception as e:
        print(f"[RAG] PDF 수집 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def _ensure_ingested():
    """컬렉션이 비어 있으면 설정된 PDF로 수집 시도."""
    client = _get_chroma_client()
    if client is None:
        return False
    try:
        collection = client.get_collection(_COLLECTION_NAME)
        if collection.count() > 0:
            return True
    except Exception:
        pass
    pdf_path = _get_pdf_path()
    if pdf_path:
        return ingest_pdf(pdf_path)
    return False


def get_context(question: str, top_k: int = _TOP_K) -> str:
    """
    질문과 관련된 문서 청크를 반환.
    컬렉션이 비어 있으면 ENGINEER_PDF_PATH 또는 data/Engineer.pdf로 수집 시도.
    """
    if not question or not question.strip():
        return ""
    if not _ensure_ingested():
        return ""
    try:
        client = _get_chroma_client()
        if client is None:
            return ""
        collection = client.get_collection(_COLLECTION_NAME)
        if collection.count() == 0:
            return ""
        q_embedding = _embed([question.strip()])
        if not q_embedding:
            return ""
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents"],
        )
        docs = results.get("documents")
        if not docs or not docs[0]:
            return ""
        return "\n\n---\n\n".join(docs[0])
    except Exception as e:
        print(f"[RAG] 검색 오류: {e}")
        return ""
