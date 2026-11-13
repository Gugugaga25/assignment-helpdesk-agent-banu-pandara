"""
Router — menentukan apakah query user butuh RAG (informasi/kebijakan)
atau tool call (aksi spesifik, misal cek status tiket).
"""

import os
import re
from typing import Any, Dict, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

from app.rag import answer_from_knowledge_base
from app.tools import lookup_ticket_status, get_ticket_status

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """Mengembalikan singleton client Google GenAI."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


def extract_ticket_id(query: str) -> Optional[str]:
    """
    Mengekstrak format identifier tiket dari teks query.
    Mendukung format 'TICKET-XXXX' (case-insensitive) dan variasi lainnya.
    """
    if not query:
        return None

    # Pola utama sesuai spesifikasi PRD Section 3.2.C: r"TICKET-\d+"
    match = re.search(r"\bTICKET[-_ ]?(\d+)\b", query, re.IGNORECASE)
    if match:
        num_part = match.group(1)
        clean_num = num_part.zfill(4) if len(num_part) < 4 else num_part
        return f"TICKET-{clean_num}"

    # Pola alternatif: 'tiket 0042' atau 'ticket #0042'
    match_alt = re.search(r"\b(?:tiket|ticket)\s*#?\s*(\d+)\b", query, re.IGNORECASE)
    if match_alt:
        clean_num = match_alt.group(1).zfill(4)
        return f"TICKET-{clean_num}"

    return None


async def detect_intent(query: str) -> str:
    """
    Menentukan intent dari query user:
      - "ticket_status" -> permintaan cek status tiket, arahkan ke tool call
      - "faq"           -> pertanyaan informasi/kebijakan, arahkan ke RAG
    """
    if not query:
        return "faq"

    clean_query = query.strip()
    query_lower = clean_query.lower()
    ticket_id = extract_ticket_id(clean_query)

    # 1. Deterministic Fast-Path: Jika query memuat format TICKET-XXXX
    if re.search(r"\bTICKET[-_ ]?\d+\b", clean_query, re.IGNORECASE):
        return "ticket_status"

    # 2. Rule-based: Jika ada nomor tiket dan kata kunci aksi tiket
    ticket_action_keywords = [
        "status", "cek", "periksa", "progres", "progress",
        "lacak", "update", "apa kabar", "posisi", "tinjau"
    ]
    if ticket_id and any(kw in query_lower for kw in ticket_action_keywords):
        return "ticket_status"

    # 3. Rule-based: Frasa eksplisit pengecekan tiket
    explicit_phrases = [
        "status tiket", "cek tiket", "periksa tiket", "progres tiket",
        "progress tiket", "lacak tiket", "tiket status", "tiket saya"
    ]
    if any(phrase in query_lower for phrase in explicit_phrases):
        return "ticket_status"

    # 4. LLM Intent Classifier (Gemini) untuk query ambigu/kompleks
    try:
        client = get_client()
        prompt = f"""Anda adalah sistem klasifikasi intent AI untuk Helpdesk IT & People Operations.
Tugas Anda mengklasifikasikan pertanyaan pengguna ke dalam salah satu dari dua intent:
1. 'ticket_status' : jika pengguna ingin mengetahui, mengecek, atau melacak status, progres, atau detail dari suatu tiket layanan.
2. 'faq' : jika pengguna menanyakan panduan, SOP, kebijakan, prosedur teknis, cuti, fasilitas, atau pertanyaan umum lainnya.

Instruksi:
Jawab HANYA dengan satu kata: 'ticket_status' atau 'faq'.

Pertanyaan Pengguna: "{clean_query}"
Klasifikasi Intent:"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        predicted = response.text.strip().lower()
        if "ticket_status" in predicted:
            return "ticket_status"
        return "faq"
    except Exception:
        # Fallback aman ke faq jika koneksi LLM intent bermasalah
        return "faq"


async def handle_query(query: str) -> Dict[str, Any]:
    """
    Fungsi utama yang dipanggil oleh endpoint /ask.
    Menjalankan intent detection, lalu merutekan ke RAG atau tool call yang sesuai.
    Mengembalikan format terpadu: {"status": "...", "intent": "...", "data": ...}.
    """
    try:
        intent = await detect_intent(query)

        if intent == "ticket_status":
            ticket_id = extract_ticket_id(query)
            data = lookup_ticket_status(ticket_id)
            return {
                "status": "success",
                "intent": "ticket_status",
                "data": data,
            }

        elif intent == "faq":
            data = await answer_from_knowledge_base(query)
            return {
                "status": "success",
                "intent": "faq",
                "data": data,
            }

        else:
            return {
                "status": "error",
                "intent": "unknown",
                "data": {
                    "error_message": "Maaf, jenis permintaan Anda tidak dapat dikenali."
                },
            }

    except Exception as exc:
        return {
            "status": "error",
            "intent": "unknown",
            "data": {
                "error_message": "LLM Service Unavailable. Silakan coba kembali beberapa saat lagi."
            },
        }
