import google.generativeai as genai
import os
from core.embeddings import get_query_embeddings
from core.vector_store import vector_db
from core.bm25_store import bm25_store
from core.reranker import rerank_chunks
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ.get("API_KEY"))

def rag_answer(query: str):

    # 1. Get embedding
    query_emb = get_query_embeddings(query)

    # 2. Search vector DB
    semantic_results = vector_db.search(query_emb, k=15)

    # ---------- 3. Keyword search (BM25) ----------
    keyword_results = bm25_store.search(query, k=7)

    # ---------- 4. Merge + dedupe ----------
    candidate_chunks = list(dict.fromkeys(semantic_results + keyword_results))
    
    print("*" * 80)
    print("Candidate Chunks")
    print("*" * 80)
    print(candidate_chunks)
    print("*" * 80)

    # ---------- 5. Re-rank candidates ----------
    best_chunks = rerank_chunks(query, candidate_chunks, top_k=5)

    print("*" * 80)
    print("Candidate Chunks")
    print("*" * 80)
    print(best_chunks)
    print("*" * 80)

    # 6. Build optimized context (with size limit + structure)
    context = build_context(best_chunks, max_chars=4000)

    # 3. Build prompt
    prompt = f"""
    You are a helpful assistant.
    Use only the context.

    Context:
    {context}

    Question:
    {query}
    """

    # 4. Gemini streaming
    model = genai.GenerativeModel("gemini-2.5-pro")
    stream = model.generate_content(prompt, stream=True)

    # 5. Yield chunks
    for chunk in stream:
        if chunk.text:
            yield chunk.text

def build_context(chunks: list[str], max_chars: int = 4000) -> str:
    """
    Build a compact, structured context string from selected chunks,
    respecting a max character budget.
    """
    selected_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Optional: small header for each chunk
        header = f"[Chunk {i}]\n"
        part = header + chunk + "\n\n"
        part_len = len(part)

        if total_chars + part_len > max_chars:
            break

        selected_parts.append(part)
        total_chars += part_len

    return "\n---\n".join(selected_parts)
