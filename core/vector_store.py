import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, embeddings, chunk):
        # Convert to shape (1, 384)
        vector = np.array(embeddings, dtype="float32").reshape(1, -1)
        self.index.add(vector)
        self.chunks.append(chunk)

    def search(self, query_embeddings, k=3):
        if len(self.chunks) == 0:
            return []
        
        # Convert to shape (1, 384)
        query_vector = np.array(query_embeddings, dtype="float32").reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)

        result = []
        print(indices)
        for idx in indices[0]:
            if idx < len(self.chunks):
                result.append(self.chunks[idx])
        
        # Remove duplicates while preserving order
        unique_results = list(dict.fromkeys(result))

        return unique_results

vector_db = VectorStore()
