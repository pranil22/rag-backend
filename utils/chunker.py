from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size = 800, chunk_overlap = 150):
    """
    Smart semantic-aware chunking.
    Produces well-structured chunks suitable for embeddings.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ".", "!", "?",
            " "
        ]
    )

    chunks = splitter.split_text(text)
    return chunks