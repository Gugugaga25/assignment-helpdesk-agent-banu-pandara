"""
Modul RAG — mengubah knowledge base menjadi embedding, menyimpannya ke FAISS in-memory,
melakukan semantic search, lalu menghasilkan jawaban menggunakan Google Gemini API.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "it_helpdesk_knowledge_base.csv")
)

# Model yang telah dipastikan
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")

_client: Optional[genai.Client] = None
_index: Optional[faiss.IndexFlatIP] = None
_documents: List[Dict[str, Any]] = []
_df: Optional[pd.DataFrame] = None


def get_client() -> genai.Client:
    """Mengembalikan singleton client Google GenAI."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


def _get_embedding(text: str) -> List[float]:
    """Menghasilkan embedding vektor tunggal menggunakan model text-embedding-004."""
    client = get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def _get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Menghasilkan embedding vektor dalam batch untuk efisiensi inisialisasi index."""
    client = get_client()
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [emb.values for emb in result.embeddings]
    except Exception:
        # Fallback: jika batching gagal, buat embedding satu per satu
        return [_get_embedding(t) for t in texts]


def build_index() -> None:
    """
    Membangun FAISS IndexFlatIP secara in-memory dari file it_helpdesk_knowledge_base.csv.
    Vektor dinormalisasi L2 agar Inner Product setara dengan Cosine Similarity.
    """
    global _index, _documents, _df

    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(f"Knowledge base CSV tidak ditemukan di: {KNOWLEDGE_BASE_PATH}")

    _df = pd.read_csv(KNOWLEDGE_BASE_PATH)
    _documents = []
    text_corpus: List[str] = []

    for _, row in _df.iterrows():
        faq_id = int(row["faq_id"])
        category = str(row["category"]).strip()
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        # Representasi dokumen komprehensif untuk pencarian semantik optimal
        doc_text = f"Kategori: {category}\nPertanyaan: {question}\nJawaban: {answer}"
        text_corpus.append(doc_text)

        _documents.append(
            {
                "id": faq_id,
                "topic": category,
                "category": category,
                "question": question,
                "answer": answer,
                "content": doc_text,
            }
        )

    # Dapatkan embedding untuk seluruh baris dokumen
    raw_embeddings = _get_embeddings_batch(text_corpus)
    embeddings_np = np.array(raw_embeddings, dtype=np.float32)

    # Normalisasi L2 untuk Cosine Similarity
    faiss.normalize_L2(embeddings_np)
    dimension = embeddings_np.shape[1]

    # Inisialisasi FAISS IndexFlatIP
    _index = faiss.IndexFlatIP(dimension)
    _index.add(embeddings_np)


def ensure_index() -> None:
    """Memastikan index FAISS telah terbangun sebelum proses pencarian."""
    if _index is None or not _documents:
        build_index()


def semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Mencari top_k dokumen terdekat secara semantik dari FAISS index.
    """
    ensure_index()
    if _index is None or not _documents:
        return []

    # Embedding query dan normalisasi L2
    query_vector = np.array([_get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    k = min(top_k, len(_documents))
    distances, indices = _index.search(query_vector, k)

    results: List[Dict[str, Any]] = []
    for idx in indices[0]:
        if 0 <= idx < len(_documents):
            results.append(_documents[idx])

    return results


def generate_rag_answer(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Menghasilkan jawaban berbasis konteks hasil retrieval menggunakan model Gemini.
    Menerapkan prompt Persona-Task-Context-Format dan guardrails ketat tanpa halusinasi.
    """
    if not retrieved_docs:
        return "Maaf, saya tidak memiliki informasi tersebut dalam basis pengetahuan internal perusahaan."

    # Format potongan context
    context_blocks = []
    for doc in retrieved_docs:
        context_blocks.append(
            f"[ID FAQ #{doc['id']} - Topik: {doc['topic']}]\n"
            f"Pertanyaan: {doc['question']}\n"
            f"Jawaban: {doc['answer']}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Anda adalah Helpdesk AI Specialist yang profesional, ramah, dan faktual di lingkungan internal perusahaan.
Tugas Anda adalah menjawab pertanyaan karyawan dengan akurat HANYA berdasarkan konteks knowledge base yang dilampirkan.

Konteks Knowledge Base:
{context_str}

Pertanyaan Karyawan:
{query}

Format & Guardrails Wajib:
1. Jawab HANYA berdasarkan konteks knowledge base di atas. Jangan mengarang, berasumsi, atau menambahkan informasi di luar konteks.
2. Jika jawaban dari pertanyaan tidak tercantum dalam konteks knowledge base (termasuk pertanyaan ambigu, obrolan santai di luar konteks, atau out-of-domain seperti resep makanan, puisi, dan sejenisnya), katakan dengan sopan dan jelas bahwa Anda tidak memiliki informasi tersebut dalam basis pengetahuan internal.
3. Sampaikan jawaban dalam Bahasa Indonesia yang formal, jelas, dan langsung menjawab inti pertanyaan karyawan.
"""

    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )

    if response.text:
        return response.text.strip()
    return "Maaf, tidak dapat menghasilkan jawaban saat ini."


async def answer_from_knowledge_base(query: str) -> Dict[str, Any]:
    """
    Fungsi utama yang dipanggil router.py ketika intent = 'faq'.
    Melakukan semantic search, menghasilkan jawaban via LLM,
    dan mengembalikan struktur data lengkap (answer & sources) sesuai PRD Kasus 3.
    """
    docs = semantic_search(query, top_k=3)
    answer = generate_rag_answer(query, docs)

    sources = [{"id": doc["id"], "topic": doc["topic"]} for doc in docs]

    return {
        "answer": answer,
        "sources": sources,
    }
