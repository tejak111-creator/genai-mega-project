from app.rag.loader import load_text_files
from app.rag.chunker import chunk_document
from app.rag.embeddings import SentenceTransformerProvider
from app.rag.vector_store import FaissVectorStore

def main():
    # 1) Load
    text = load_text_files("data/sample.txt")

    # 2) Chunk with metadata
    chunks = chunk_document(text, doc_id="sample.txt", chunk_size=500, overlap=50)
    print("chunks:", len(chunks))

    # 3) Embed chunks
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    vectors = embedder.embed([c.text for c in chunks])
    print("vectors:", len(vectors), "dim:", vectors[0].shape)

    # 4) Build FAISS index
    dim = vectors[0].shape[0]
    store = FaissVectorStore(embedding_dim=dim)
    store.add(vectors, chunks)

    # 5) Search
    query = "What is this document about?"
    results = store.search(query, embedder=embedder, top_k=3)

    print("\nQUERY:", query)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | score={r.score:.4f} | doc={r.chunk.doc_id} chunk={r.chunk.chunk_id}")
        print(r.chunk.text[:300])

if __name__ == "__main__":
    main()
