from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

def rerank_chunks(query: str, chunks: list[str], top_k: int = 5):
    """
    Re-rank candidate chunks based on (query, chunk) relevance score.
    Returns top_k highest relevance chunks.
    """

    if not chunks:
        return []
    
    # Prepare pairs: (query, chunk)
    pairs = [(query, chunk) for chunk in chunks]

    # Model predicts relevance scores
    scores = reranker.predict(pairs)

     # Sort by score (descending)
    sorted_pairs = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

     # Return top_k chunks only
    top_chunks = [chunk for _, chunk in sorted_pairs[:top_k]]
    return top_chunks