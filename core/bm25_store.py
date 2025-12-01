from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self):
        self.raw_chunks = []
        self.tokenzed_chunks = []
        self.bm25 = None

    def add(self, chunk_text: str):
        tokens = chunk_text.lower().split()
        self.raw_chunks.append(chunk_text)
        self.tokenzed_chunks.append(tokens)

    def build(self):
        """
        Must be called after all chunks are added.
        Creates BM25 model.
        """
        self.bm25 = BM25Okapi(self.tokenzed_chunks)
    
    def search(self, query: str, k = 5):
        """
        Keyword search.
        Returns top-k chunk texts.
        """

        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [self.raw_chunks[idx] for idx in top_indices]

bm25_store = BM25Store()