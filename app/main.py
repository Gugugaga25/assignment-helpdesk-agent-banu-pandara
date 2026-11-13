"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.router import handle_query
from app.rag import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inisialisasi FAISS index secara in-memory saat aplikasi startup
    sesuai spesifikasi PRD Section 3.2.A dan Milestone 4.
    """
    try:
        build_index()
    except Exception as exc:
        print(f"[Warning] Inisialisasi FAISS index saat startup: {exc}")
    yield


app = FastAPI(
    title="IT Helpdesk Assistant API",
    description="End-to-End RAG & AI Agent Helpdesk Microservice",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    status: str
    intent: str
    data: Any


@app.get("/health")
async def health():
    """Endpoint sederhana untuk health check service."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.
    """
    try:
        result = await handle_query(request.query)
        return AskResponse(**result)
    except Exception as exc:
        return AskResponse(
            status="error",
            intent="unknown",
            data={
                "error_message": "LLM Service Unavailable. Silakan coba kembali beberapa saat lagi."
            },
        )
