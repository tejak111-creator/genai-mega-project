from app.rag.multimodal_embeddings import CLIPEmbeddingProvider
from app.rag.vector_store import FaissVectorStore
from app.rag.models import DocumentChunk


def main():
    clip = CLIPEmbeddingProvider()

    texts = ["a cat", "a dog"]
    text_vectors = clip.embed_text(texts)

    chunks = [
        DocumentChunk(doc_id="text1", chunk_id=0, text="a cat", modality="text"),
        DocumentChunk(doc_id="text2", chunk_id=1, text="a dog", modality="text"),
    ]

    dim = text_vectors[0].shape[0]
    store = FaissVectorStore(dim)

    store.add(text_vectors, chunks)

    results = store.search("a cat", clip, top_k=1)

    print(results)


if __name__ == "__main__":
    main()