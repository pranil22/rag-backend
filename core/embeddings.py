from sentence_transformers import SentenceTransformer

# BEST model for RAG retrieval
model = SentenceTransformer("intfloat/e5-base-v2")

def get_query_embeddings(text: str):
    return model.encode("query: " + text).tolist()

def get_passage_embeddings(text: str):
    return model.encode("passage: " + text).tolist()