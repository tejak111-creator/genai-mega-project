from app.rag.loader import load_text_files
from app.rag.chunker import chunk_document
from app.rag.embeddings import SentenceTransformerProvider
from app.rag.vector_store import FaissVectorStore
from app.rag.retriever import Retriever
from app.rag.pipeline import RagPipeline
from app.core.llm import get_provider

def main():
    text = load_text_files("data/sample.txt")
    chunks = chunk_document(text, doc_id="sample.txt")
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    vectors = embedder.embed([c.text for c in chunks])
    dim = vectors[0].shape[0]
    store = FaissVectorStore(embedding_dim=dim)
    store.add(vectors=vectors, chunks=chunks)
    retriever = Retriever(store=store, embedder=embedder)
    llm = get_provider()
    rag = RagPipeline(retriever=retriever,llm_provider=llm)
    question = "What does this document say?"
    result = rag.run(question, top_k=3)
    print("\n QUESTION:", question)
    print("\nANSWER:\n", result.answer)
    print("\n RETRIEVED:")
    for r in result.results:
        preview = r.chunk.text[:120].replace("\n", " ")
        print(f"- doc={r.chunk.doc_id} chunk={r.chunk.chunk_id} score={r.score:.4f}")
        print("  preview:", preview)

if __name__ == "__main__":
    main()