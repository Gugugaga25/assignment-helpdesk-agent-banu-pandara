"""
Modul Tools — fungsi-fungsi aksi deterministik untuk pengecekan data tiket.
Mengakses data/mock_tickets.csv secara aman tanpa pencarian semantik/probabilistik.
"""

import os
from typing import Any, Dict, Optional
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_TICKETS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "mock_tickets.csv"))

_tickets_df = None


def get_tickets_df() -> pd.DataFrame:
    """Memuat mock_tickets.csv ke dalam DataFrame Pandas secara in-memory (singleton)."""
    global _tickets_df
    if _tickets_df is None:
        if not os.path.exists(MOCK_TICKETS_PATH):
            raise FileNotFoundError(f"Mock tickets file not found at: {MOCK_TICKETS_PATH}")
        _tickets_df = pd.read_csv(MOCK_TICKETS_PATH)
        _tickets_df["ticket_id_clean"] = _tickets_df["ticket_id"].astype(str).str.strip().str.upper()
    return _tickets_df


def lookup_ticket_status(ticket_id: Optional[str]) -> Dict[str, Any]:
    """
    Mencari status tiket berdasarkan ticket_id pada data mock_tickets.csv.
    Input dibersihkan (strip spasi & standarisasi uppercase).
    Mengembalikan data tiket jika ditemukan, atau pesan informatif jika tidak ditemukan,
    tanpa melempar unhandled exception / crash HTTP 500.
    """
    if not ticket_id or not str(ticket_id).strip():
        return {
            "ticket_id": "",
            "found": False,
            "message": "Nomor tiket tidak valid atau tidak disertakan.",
        }

    clean_id = str(ticket_id).strip().upper()

    try:
        df = get_tickets_df()
        matched = df[df["ticket_id_clean"] == clean_id]

        if matched.empty:
            return {
                "ticket_id": clean_id,
                "found": False,
                "message": f"Tiket dengan ID {clean_id} tidak ditemukan dalam sistem.",
            }

        row = matched.iloc[0]
        return {
            "ticket_id": str(row["ticket_id"]),
            "status": str(row["status"]),
            "issue": str(row["subject"]),
            "updated_at": str(row["last_update"]),
            "employee_name": str(row["employee_name"]),
            "category": str(row["category"]),
            "found": True,
        }
    except Exception as exc:
        return {
            "ticket_id": clean_id,
            "found": False,
            "message": f"Terjadi kesalahan saat memeriksa tiket {clean_id}: {str(exc)}",
        }


def check_ticket_status(ticket_id: Optional[str]) -> Dict[str, Any]:
    """Alias untuk lookup_ticket_status."""
    return lookup_ticket_status(ticket_id)


def get_ticket_status(ticket_id: Optional[str]) -> Dict[str, Any]:
    """Fungsi utama sesuai spesifikasi PRD Section 3.2.B."""
    return lookup_ticket_status(ticket_id)
