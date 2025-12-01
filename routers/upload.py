from fastapi import APIRouter, UploadFile, File
import os
from utils.pdf_reader import extract_text_from_pdf
from utils.chunker import chunk_text
from core.embeddings import get_passage_embeddings
from core.vector_store import vector_db
from core.bm25_store import bm25_store

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_location)
    chunks = chunk_text(text)

    for chunk in chunks:
        emb = get_passage_embeddings(chunk)
        vector_db.add(emb, chunk)
        bm25_store.add(chunk)

    bm25_store.build()

    return {"message": "PDF processed successfully", "chunks": len(chunks)}
