from app.rag.embeddings import DummyEmbeddingProvider

provider = DummyEmbeddingProvider()

texts = ["hello world", "machine learning"]

vectors = provider.embed(texts)

print(len(vectors))
print(vectors[0])
print(vectors[0].shape)